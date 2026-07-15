#!/usr/bin/env python3
"""
3DGS Camera Relocalization
==========================
Given a query image from unknown pose, estimates 6-DoF camera pose
by matching SuperPoint features against the trained 3DGS map.

Pipeline:
  1. Load trained 3DGS model
  2. Load COLMAP poses (ground truth for evaluation)
  3. Hold out 20 images as test set
  4. For each query image:
     a. Start from coarse pose (nearest neighbor in descriptor space)
     b. Render 3DGS from coarse pose → RGB + depth
     c. Extract SuperPoint features from rendered + query images
     d. LightGlue match → 2D-2D correspondences
     e. Back-project rendered matches to 3D using depth map
     f. PnP+RANSAC → refined 6-DoF pose
  5. Compute and report position error vs ground truth
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import struct
import numpy as np
import torch
import cv2
from PIL import Image
from argparse import ArgumentParser, Namespace
from pathlib import Path

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

# ── 3DGS imports ──────────────────────────────────────────────────────────────
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.camera_utils import loadCam
from scene.cameras import Camera
from arguments import ModelParams, PipelineParams, get_combined_args

device = torch.device("cuda")

# =====================================================================
# 1. LOAD COLMAP POSES
# =====================================================================
def qvec_to_rotmat(q):
    """COLMAP quaternion (qw,qx,qy,qz) → 3x3 rotation matrix."""
    qw,qx,qy,qz = q
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [  2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2),   2*(qy*qz-qw*qx)],
        [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
    ])

def read_images_bin(path):
    images = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            iid      = struct.unpack('<I', f.read(4))[0]
            qvec     = struct.unpack('<4d', f.read(32))
            tvec     = struct.unpack('<3d', f.read(24))
            cam_id   = struct.unpack('<I', f.read(4))[0]
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00': break
                name += c
            npts = struct.unpack('<Q', f.read(8))[0]
            f.read(24 * npts)
            R = qvec_to_rotmat(qvec)
            t = np.array(tvec)
            # Camera centre in world: C = -R^T @ t
            C = -R.T @ t
            images[name.decode()] = {'R': R, 't': t, 'C': C, 'cam_id': cam_id}
    return images

def read_cameras_bin(path):
    """Returns dict of camera_id → {fx,fy,cx,cy,W,H}."""
    cameras = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            cam_id   = struct.unpack('<I', f.read(4))[0]
            model_id = struct.unpack('<I', f.read(4))[0]
            W        = struct.unpack('<Q', f.read(8))[0]
            H        = struct.unpack('<Q', f.read(8))[0]
            # PINHOLE: fx fy cx cy
            params   = struct.unpack('<4d', f.read(32))
            cameras[cam_id] = {
                'W': W, 'H': H,
                'fx': params[0], 'fy': params[1],
                'cx': params[2], 'cy': params[3]
            }
    return cameras

# =====================================================================
# 2. LOAD 3DGS MODEL
# =====================================================================
def load_gaussians(model_path, iteration=30000):
    print(f"Loading 3DGS model from {model_path} iter {iteration}...")
    gaussians = GaussianModel(sh_degree=3)
    ply_path  = os.path.join(model_path, "point_cloud",
                              f"iteration_{iteration}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    print(f"Loaded {gaussians.get_xyz.shape[0]:,} Gaussians")
    return gaussians

# =====================================================================
# 3. RENDER FROM POSE
# =====================================================================
def make_camera(R, t, fx, fy, cx, cy, W, H):
    """Build a GTSAM-style Camera object from pose + intrinsics."""
    FoVx = 2 * np.arctan(W / (2 * fx))
    FoVy = 2 * np.arctan(H / (2 * fy))
    cam  = Camera(
        colmap_id=0, R=R, T=t,
        FoVx=FoVx, FoVy=FoVy,
        image=Image.new("RGB", (W, H), (0, 0, 0)), # <--- Changed to dummy PIL Image
        image_name="query", uid=0,
        data_device="cuda",
        resolution=(W, H),                         # <--- Changed to standard tuple
        depth_params=None,
        invdepthmap=None
    )
    return cam

def render_from_pose(gaussians, pipe, bg, R, t, fx, fy, cx, cy, W, H):
    cam = make_camera(R, t, fx, fy, cx, cy, W, H)
    with torch.no_grad():
        pkg = render(cam, gaussians, pipe, bg)
    rgb   = pkg["render"]          # (3, H, W) float [0,1]
    depth = pkg["depth"]           # (1, H, W) float metres
    return rgb, depth

# =====================================================================
# 4. FEATURE EXTRACTION HELPER
# =====================================================================
def extract_feats(extractor, img_np):
    """img_np: HxWx3 uint8 → SuperPoint features on GPU."""
    t = torch.from_numpy(img_np).float().permute(2,0,1)[None] / 255.0
    with torch.no_grad():
        feats = extractor.extract(t.to(device))
    return feats

# =====================================================================
# 5. FEATURE-MATCH RELOCALIZATION (main method)
# =====================================================================
def relocalize_feature_match(query_img_np, coarse_R, coarse_t,
                              gaussians, pipe, bg,
                              fx, fy, cx, cy, W, H,
                              extractor, matcher):
    """
    Returns (R_refined, t_refined, n_inliers) or (None, None, 0).
    """
    H_q, W_q = query_img_np.shape[:2]

    # Render from coarse pose
    rgb, depth = render_from_pose(gaussians, pipe, bg,
                                   coarse_R, coarse_t,
                                   fx, fy, cx, cy, W, H)
    rendered_np = (rgb.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    depth_np    = depth.squeeze().cpu().numpy()   # (H, W)

    # Resize query to match render resolution if needed
    if H_q != H or W_q != W:
        query_resized = np.array(
            Image.fromarray(query_img_np).resize((W, H), Image.BILINEAR))
    else:
        query_resized = query_img_np

    # Extract features
    feats0 = extract_feats(extractor, rendered_np)   # rendered (map)
    feats1 = extract_feats(extractor, query_resized)  # query

    # Match
    with torch.no_grad():
        m01     = matcher({'image0': feats0, 'image1': feats1})
        f0, f1, mc = [rbd(x) for x in [feats0, feats1, m01]]
    matches = mc['matches'].cpu().numpy()   # (M, 2)

    if len(matches) < 15:
        return None, None, 0

    kpts0 = f0['keypoints'].cpu().numpy()   # rendered keypoints
    kpts1 = f1['keypoints'].cpu().numpy()   # query keypoints

    # Build 3D-2D correspondences
    # 3D: back-project rendered keypoint through depth map
    # 2D: corresponding query keypoint
    obj_pts, img_pts = [], []
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)

    for m0, m1 in matches:
        u0 = int(round(float(kpts0[m0, 0])))
        v0 = int(round(float(kpts0[m0, 1])))
        if not (0 <= v0 < H and 0 <= u0 < W):
            continue
        Z = float(depth_np[v0, u0])
        if Z < 0.05 or Z > 50.0 or not np.isfinite(Z):
            continue
        # 3D point in camera frame
        Xc = (u0 - cx) * Z / fx
        Yc = (v0 - cy) * Z / fy
        pt_cam = np.array([Xc, Yc, Z])
        # Transform to world frame using coarse pose
        # COLMAP convention: x_cam = R @ x_world + t
        pt_world = coarse_R.T @ (pt_cam - coarse_t)
        obj_pts.append(pt_world.astype(np.float32))
        img_pts.append(kpts1[m1].astype(np.float32))

    if len(obj_pts) < 12:
        return None, None, 0

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.array(obj_pts), np.array(img_pts), K, None,
        iterationsCount=2000, reprojectionError=3.0,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP
    )
    if not success or inliers is None or len(inliers) < 10:
        return None, None, 0

    R_ref, _ = cv2.Rodrigues(rvec)
    t_ref    = tvec.flatten()
    return R_ref, t_ref, len(inliers)

# =====================================================================
# 6. RENDER-COMPARE BASELINE
# =====================================================================
def relocalize_render_compare(query_img_np, init_R, init_t,
                               gaussians, pipe, bg,
                               fx, fy, cx, cy, W, H,
                               n_iters=150, lr=5e-3):
    """
    Photometric pose refinement via gradient descent on rendered image.
    Returns (R_refined, t_refined).
    """
    import torch.nn.functional as F

    H_q, W_q = query_img_np.shape[:2]
    if H_q != H or W_q != W:
        q_np = np.array(Image.fromarray(query_img_np).resize((W,H), Image.BILINEAR))
    else:
        q_np = query_img_np
    query_t = torch.from_numpy(q_np).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0

    # Parameterise pose as axis-angle + translation
    aa  = torch.zeros(3, requires_grad=True, device=device, dtype=torch.float64)
    t_p = torch.tensor(init_t, requires_grad=True, device=device, dtype=torch.float64)
    R0  = torch.tensor(init_R, device=device, dtype=torch.float64)
    opt = torch.optim.Adam([aa, t_p], lr=lr)

    def aa_to_R(aa):
        angle = torch.norm(aa)
        if angle < 1e-6:
            return torch.eye(3, device=device, dtype=torch.float64)
        axis = aa / angle
        K = torch.zeros(3,3,device=device,dtype=torch.float64)
        K[0,1]=-axis[2]; K[0,2]=axis[1]
        K[1,0]=axis[2];  K[1,2]=-axis[0]
        K[2,0]=-axis[1]; K[2,1]=axis[0]
        return torch.eye(3,device=device,dtype=torch.float64) + \
               torch.sin(angle)*K + (1-torch.cos(angle))*(K@K)

    best_loss, best_R, best_t = 1e9, init_R.copy(), init_t.copy()

    # Create a base camera outside the loop to hold our intrinsics
    base_cam = make_camera(init_R, init_t, fx, fy, cx, cy, W, H)

    for i in range(n_iters):
        opt.zero_grad()
        R_cur = (aa_to_R(aa) @ R0).float()
        t_cur = t_p.float()
        
        # 1. Build the World-to-View matrix in PyTorch (maintaining the gradient graph)
        w2v = torch.eye(4, device=device)
        w2v[:3, :3] = R_cur
        w2v[:3, 3]  = t_cur
        
        # 2. Inject the differentiable transforms directly into our base camera
        base_cam.world_view_transform = w2v.transpose(0, 1)
        base_cam.full_proj_transform = (base_cam.world_view_transform.unsqueeze(0).bmm(base_cam.projection_matrix.unsqueeze(0))).squeeze(0)
        base_cam.camera_center = base_cam.world_view_transform.inverse()[3, :3]
        
        # 3. Call the 3DGS renderer directly (bypassing the no_grad helper)
        pkg = render(base_cam, gaussians, pipe, bg)
        rgb = pkg["render"]
        
        # 4. Compute photometric loss and backpropagate
        loss = torch.abs(rgb.unsqueeze(0) - query_t).mean()
        loss.backward()
        opt.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_R = R_cur.detach().cpu().numpy()
            best_t = t_cur.detach().cpu().numpy()

    return best_R, best_t

# =====================================================================
# 7. COARSE POSE INITIALISATION
#    Use COLMAP poses of training images as candidates.
#    Pick nearest by L2 distance in camera-centre space — simulates
#    "IMU gives rough position" in a real robot scenario.
# =====================================================================
def coarse_pose_from_nearest(query_gt_C, train_images, noise_m=0.5):
    """
    Finds training image whose camera centre is closest to query.
    Adds Gaussian noise to simulate imperfect prior.
    Returns (R_coarse, t_coarse).
    """
    best_dist, best_name = 1e9, None
    for name, data in train_images.items():
        d = np.linalg.norm(data['C'] - query_gt_C)
        if d < best_dist:
            best_dist = d
            best_name = name
    R_nn = train_images[best_name]['R']
    t_nn = train_images[best_name]['t']
    # Add noise to translation
    t_noisy = t_nn + np.random.randn(3) * noise_m
    return R_nn, t_noisy, best_name

# =====================================================================
# 8. EVALUATION
# =====================================================================
def position_error(t_est, t_gt):
    """Camera centre error in metres."""
    R_est = np.eye(3)   # placeholder — we care about translation
    C_est = -R_est.T @ t_est   # approximate
    return float(np.linalg.norm(t_est - t_gt))

# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", "-m", required=True)
    parser.add_argument("--scene_path", "-s", required=True)
    parser.add_argument("--iteration",  type=int, default=30000)
    parser.add_argument("--n_test",     type=int, default=20,
                        help="Number of held-out test images")
    parser.add_argument("--noise_m",    type=float, default=0.3,
                        help="Coarse pose noise in metres")
    args = parser.parse_args()

    # Pipeline params for 3DGS renderer
    pipe = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False  # <--- ADD THIS LINE
    )

    bg = torch.tensor([0,0,0], dtype=torch.float32, device=device)

    # Load model
    gaussians = load_gaussians(args.model_path, args.iteration)

    # Load COLMAP data
    colmap_dir = os.path.join(args.scene_path, "sparse", "0")
    all_images  = read_images_bin(os.path.join(colmap_dir, "images.bin"))
    cameras_db  = read_cameras_bin(os.path.join(colmap_dir, "cameras.bin"))

    # Camera intrinsics (assume single camera)
    cam_id   = list(all_images.values())[0]['cam_id']
    cam_info = cameras_db[cam_id]
    fx, fy   = cam_info['fx'], cam_info['fy']
    cx, cy   = cam_info['cx'], cam_info['cy']
    W,  H    = cam_info['W'],  cam_info['H']
    # Half resolution (matches training -r 2)
    fx /= 2; fy /= 2; cx /= 2; cy /= 2
    W  //= 2; H //= 2
    print(f"Camera: {W}x{H}  fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    # Split: hold out every 9th image as test
    all_names   = sorted(all_images.keys())
    test_names  = all_names[::len(all_names)//args.n_test][:args.n_test]
    train_names = [n for n in all_names if n not in set(test_names)]
    train_images = {n: all_images[n] for n in train_names}
    print(f"Train: {len(train_images)}  Test: {len(test_names)}")

    # Feature matcher
    extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
    matcher   = LightGlue(features='superpoint').eval().to(device)

    images_dir = os.path.join(args.scene_path, "images")

    # Results storage
    results = {
        'feat_match': {'errors': [], 'inliers': [], 'success': 0},
        'render_cmp': {'errors': [], 'success': 0},
        'coarse':     {'errors': []},
    }

    print(f"\nRunning relocalization on {len(test_names)} test images...\n")

    for i, name in enumerate(test_names):
        gt      = all_images[name]
        gt_R    = gt['R']
        gt_t    = gt['t']
        gt_C    = gt['C']

        # Load query image
        img_path = os.path.join(images_dir, name)
        query_np = np.array(Image.open(img_path).convert('RGB'))

        # Coarse pose: nearest training image + noise
        R_c, t_c, nn_name = coarse_pose_from_nearest(gt_C, train_images,
                                                       noise_m=args.noise_m)
        coarse_err = np.linalg.norm(t_c - gt_t)
        results['coarse']['errors'].append(coarse_err)

        print(f"[{i+1:2d}/{len(test_names)}] {name}  nn={nn_name}  "
              f"coarse_err={coarse_err:.3f}m", end="  ")

        # ── Feature-match relocalization ──────────────────────────
        R_fm, t_fm, n_inl = relocalize_feature_match(
            query_np, R_c, t_c, gaussians, pipe, bg,
            fx, fy, cx, cy, W, H, extractor, matcher
        )
        if R_fm is not None:
            err_fm = np.linalg.norm(t_fm - gt_t)
            results['feat_match']['errors'].append(err_fm)
            results['feat_match']['inliers'].append(n_inl)
            results['feat_match']['success'] += 1
            print(f"FM={err_fm:.3f}m({n_inl}inl)", end="  ")
        else:
            print(f"FM=FAIL", end="  ")

        # ── Render-compare baseline ────────────────────────────────
        R_rc, t_rc = relocalize_render_compare(
            query_np, R_c, t_c, gaussians, pipe, bg,
            fx, fy, cx, cy, W, H, n_iters=100
        )
        err_rc = np.linalg.norm(t_rc - gt_t)
        results['render_cmp']['errors'].append(err_rc)
        results['render_cmp']['success'] += 1
        print(f"RC={err_rc:.3f}m")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RELOCALIZATION RESULTS")
    print("="*60)

    def stats(errs):
        if not errs: return "no results"
        a = np.array(errs)
        return (f"median={np.median(a)*100:.1f}cm  "
                f"mean={np.mean(a)*100:.1f}cm  "
                f"max={np.max(a)*100:.1f}cm")

    print(f"Coarse prior   ({len(results['coarse']['errors'])} imgs): "
          f"{stats(results['coarse']['errors'])}")
    print(f"Feature-Match  ({results['feat_match']['success']}/{len(test_names)} success): "
          f"{stats(results['feat_match']['errors'])}")
    print(f"Render-Compare ({results['render_cmp']['success']}/{len(test_names)} success): "
          f"{stats(results['render_cmp']['errors'])}")

    if results['feat_match']['errors'] and results['coarse']['errors']:
        n = min(len(results['feat_match']['errors']),
                len(results['coarse']['errors']))
        fm  = np.median(results['feat_match']['errors'][:n])
        crs = np.median(results['coarse']['errors'][:n])
        print(f"\nFeature-Match improvement over coarse prior: "
              f"{(1-fm/crs)*100:.1f}%")

    # Save results for README
    import json
    out = {
        'coarse_median_cm':   float(np.median(results['coarse']['errors'])*100),
        'feat_match_median_cm': float(np.median(results['feat_match']['errors'])*100)
                                if results['feat_match']['errors'] else None,
        'render_cmp_median_cm': float(np.median(results['render_cmp']['errors'])*100),
        'feat_match_success_rate': results['feat_match']['success'] / len(test_names),
    }
    with open('reloc/results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to reloc/results.json")

if __name__ == "__main__":
    main()
