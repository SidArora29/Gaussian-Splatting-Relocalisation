#!/usr/bin/env python3
"""Relocalization v2 — fixed RC gradient + improved FM robustness."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import struct, json
import numpy as np
import torch
import cv2
from PIL import Image
from argparse import Namespace
from pathlib import Path
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera

device = torch.device("cuda")

# ── COLMAP readers ────────────────────────────────────────────────────────────
def qvec_to_R(q):
    qw,qx,qy,qz = q
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),   1-2*(qx**2+qz**2),  2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),    1-2*(qx**2+qy**2)]
    ])

def read_images_bin(path):
    out = {}
    with open(path,'rb') as f:
        n = struct.unpack('<Q',f.read(8))[0]
        for _ in range(n):
            iid  = struct.unpack('<I',f.read(4))[0]
            qvec = struct.unpack('<4d',f.read(32))
            tvec = struct.unpack('<3d',f.read(24))
            cam_id = struct.unpack('<I',f.read(4))[0]
            name=b''
            while True:
                c=f.read(1)
                if c==b'\x00': break
                name+=c
            npts=struct.unpack('<Q',f.read(8))[0]
            f.read(24*npts)
            R = qvec_to_R(qvec)
            t = np.array(tvec)
            out[name.decode()]={'R':R,'t':t,'C':-R.T@t,'cam_id':cam_id}
    return out

def read_cameras_bin(path):
    out={}
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        for _ in range(n):
            cid=struct.unpack('<I',f.read(4))[0]
            struct.unpack('<I',f.read(4))   # model_id
            W=struct.unpack('<Q',f.read(8))[0]
            H=struct.unpack('<Q',f.read(8))[0]
            p=struct.unpack('<4d',f.read(32))
            out[cid]={'W':W,'H':H,'fx':p[0],'fy':p[1],'cx':p[2],'cy':p[3]}
    return out

# ── Renderer ──────────────────────────────────────────────────────────────────
def render_rgbd(gaussians, pipe, bg, R, t, fx, fy, cx, cy, W, H):
    FoVx = 2*np.arctan(W/(2*fx))
    FoVy = 2*np.arctan(H/(2*fy))
    cam  = Camera(
                  resolution=(W,H),
                  colmap_id=0, R=R.transpose(), T=t,
                  FoVx=FoVx, FoVy=FoVy,
                  depth_params=None,
                  image=Image.fromarray(np.zeros((H,W,3), dtype=np.uint8)),
                  invdepthmap=None,
                  image_name="q", uid=0,
                  data_device="cuda")
    with torch.no_grad():
        pkg = render(cam, gaussians, pipe, bg)
    return pkg["render"], pkg["depth"]   # (3,H,W), (1,H,W)

def render_np(gaussians, pipe, bg, R, t, fx, fy, cx, cy, W, H):
    rgb, depth = render_rgbd(gaussians, pipe, bg, R, t, fx, fy, cx, cy, W, H)
    return (rgb.permute(1,2,0).cpu().numpy()*255).astype(np.uint8), \
           depth.squeeze().cpu().numpy()

# ── Feature helpers ───────────────────────────────────────────────────────────
def to_tensor(img_np):
    return torch.from_numpy(img_np).float().permute(2,0,1)[None]/255.0

def extract(extractor, img_np):
    with torch.no_grad():
        return extractor.extract(to_tensor(img_np).to(device))

# ── Feature-Match relocalization ─────────────────────────────────────────────
def relocalize_fm(query_np, R_c, t_c, gaussians, pipe, bg,
                  fx, fy, cx, cy, W, H, extractor, matcher,
                  min_inliers=10):
    # Resize query to render resolution
    q = np.array(Image.fromarray(query_np).resize((W,H),Image.BILINEAR))

    ren_np, dep_np = render_np(gaussians, pipe, bg, R_c, t_c,
                                fx, fy, cx, cy, W, H)

    f0 = extract(extractor, ren_np)
    f1 = extract(extractor, q)
    with torch.no_grad():
        m01 = matcher({'image0':f0,'image1':f1})
        f0, f1, mc = [rbd(x) for x in [f0, f1, m01]]
    matches = mc['matches'].cpu().numpy()

    if len(matches) < min_inliers:
        return None, None, len(matches)

    kp0 = f0['keypoints'].cpu().numpy()
    kp1 = f1['keypoints'].cpu().numpy()
    K   = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float32)

    obj, img = [], []
    for m0, m1 in matches:
        u,v = int(round(float(kp0[m0,0]))), int(round(float(kp0[m0,1])))
        if not (0<=v<H and 0<=u<W): continue
        Z = float(dep_np[v,u])
        if Z < 0.05 or Z > 50 or not np.isfinite(Z): continue
        pt_cam   = np.array([(u-cx)*Z/fx, (v-cy)*Z/fy, Z])
        pt_world = R_c.T @ (pt_cam - t_c)
        obj.append(pt_world.astype(np.float32))
        img.append(kp1[m1].astype(np.float32))

    if len(obj) < min_inliers:
        return None, None, len(obj)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.array(obj), np.array(img), K, None,
        iterationsCount=3000, reprojectionError=4.0,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP)

    if not ok or inliers is None or len(inliers) < min_inliers:
        return None, None, 0

    R_ref, _ = cv2.Rodrigues(rvec)
    return R_ref, tvec.flatten(), len(inliers)

# ── Render-Compare baseline (FIXED — no detach in loss path) ─────────────────
def relocalize_rc(query_np, R_c, t_c, gaussians, pipe, bg,
                  fx, fy, cx, cy, W, H, n_iters=200, lr=2e-3):
    """
    FIX vs v1: use direct numpy pose → render → numpy loss.
    Gradient-free optimisation using Nelder-Mead on 6-DOF pose.
    More reliable than autograd through the renderer for this use case.
    """
    from scipy.optimize import minimize

    q = np.array(Image.fromarray(query_np).resize((W,H),Image.BILINEAR),
                 dtype=np.float32) / 255.0  # (H,W,3)

    # Parameterise: 3 axis-angle + 3 translation delta
    def aa_to_R(aa):
        angle = np.linalg.norm(aa)
        if angle < 1e-8: return np.eye(3)
        ax = aa/angle
        K  = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
        return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)

    call_count = [0]
    def objective(x):
        call_count[0] += 1
        daa = x[:3]; dt = x[3:]
        R_try = aa_to_R(daa) @ R_c
        t_try = t_c + dt
        ren_np, _ = render_np(gaussians, pipe, bg,
                               R_try, t_try, fx, fy, cx, cy, W, H)
        ren = ren_np.astype(np.float32)/255.0
        return float(np.abs(ren - q).mean())

    x0     = np.zeros(6)
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': n_iters, 'xatol':1e-4,
                               'fatol':1e-5, 'disp':False})
    daa, dt = result.x[:3], result.x[3:]
    R_ref = aa_to_R(daa) @ R_c
    t_ref = t_c + dt
    return R_ref, t_ref

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", "-m", required=True)
    ap.add_argument("--scene_path", "-s", required=True)
    ap.add_argument("--iteration",  type=int, default=30000)
    ap.add_argument("--n_test",     type=int, default=20)
    ap.add_argument("--noise_m",    type=float, default=0.3)
    ap.add_argument("--skip_rc",    action="store_true",
                    help="Skip render-compare (slow, ~3 min/image)")
    args = ap.parse_args()

    pipe = Namespace(convert_SHs_python=False,
                     compute_cov3D_python=False,
                     debug=False,
                     antialiasing=False)
    bg   = torch.tensor([0,0,0], dtype=torch.float32, device=device)

    # Load model
    gaussians = GaussianModel(sh_degree=3)
    ply = os.path.join(args.model_path,"point_cloud",
                       f"iteration_{args.iteration}","point_cloud.ply")
    gaussians.load_ply(ply)
    print(f"Loaded {gaussians.get_xyz.shape[0]:,} Gaussians")

    # COLMAP
    sp = os.path.join(args.scene_path,"sparse","0")
    all_imgs = read_images_bin(os.path.join(sp,"images.bin"))
    cams_db  = read_cameras_bin(os.path.join(sp,"cameras.bin"))
    cam_id   = list(all_imgs.values())[0]['cam_id']
    ci       = cams_db[cam_id]
    # Half-res to match training -r 2
    fx,fy    = ci['fx']/2, ci['fy']/2
    cx,cy    = ci['cx']/2, ci['cy']/2
    W,H      = ci['W']//2, ci['H']//2
    print(f"Camera: {W}x{H}  fx={fx:.1f}")

    # Test/train split
    names      = sorted(all_imgs.keys())
    step       = max(1, len(names)//args.n_test)
    test_names = names[::step][:args.n_test]
    train_imgs = {n:all_imgs[n] for n in names if n not in set(test_names)}
    print(f"Train:{len(train_imgs)}  Test:{len(test_names)}")

    extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
    matcher   = LightGlue(features='superpoint').eval().to(device)
    imgs_dir  = os.path.join(args.scene_path,"images")

    res = {'coarse':[], 'fm':[], 'rc':[], 'fm_inliers':[]}

    for i, name in enumerate(test_names):
        gt  = all_imgs[name]
        q_np = np.array(Image.open(os.path.join(imgs_dir,name)).convert('RGB'))

        # Coarse: nearest training image + noise
        dists = {n: np.linalg.norm(d['C']-gt['C'])
                 for n,d in train_imgs.items()}
        nn    = min(dists, key=dists.get)
        R_c   = train_imgs[nn]['R'].copy()
        t_c   = train_imgs[nn]['t'] + np.random.randn(3)*args.noise_m
        coarse_err = float(np.linalg.norm(t_c - gt['t']))
        res['coarse'].append(coarse_err)

        print(f"[{i+1:2d}/{len(test_names)}] {name}  "
              f"coarse={coarse_err*100:.1f}cm", end="")

        # Feature-match
        R_fm, t_fm, n_inl = relocalize_fm(
            q_np, R_c, t_c, gaussians, pipe, bg,
            fx, fy, cx, cy, W, H, extractor, matcher, min_inliers=8)
        if R_fm is not None:
            err = float(np.linalg.norm(t_fm - gt['t']))
            res['fm'].append(err)
            res['fm_inliers'].append(n_inl)
            print(f"  FM={err*100:.1f}cm({n_inl}inl)", end="")
        else:
            print(f"  FM=FAIL({n_inl})", end="")

        # Render-compare (slow — skip with --skip_rc for quick test)
        if not args.skip_rc:
            R_rc, t_rc = relocalize_rc(
                q_np, R_c, t_c, gaussians, pipe, bg,
                fx, fy, cx, cy, W, H, n_iters=150)
            err_rc = float(np.linalg.norm(t_rc - gt['t']))
            res['rc'].append(err_rc)
            print(f"  RC={err_rc*100:.1f}cm", end="")

        print()

    # Summary
    def med(lst):
        return f"{np.median(lst)*100:.1f}cm" if lst else "n/a"

    print("\n" + "="*55)
    print("RESULTS (median position error)")
    print("="*55)
    print(f"Coarse prior:    {med(res['coarse'])}  (n={len(res['coarse'])})")
    print(f"Feature-Match:   {med(res['fm'])}  "
          f"(success={len(res['fm'])}/{len(test_names)}, "
          f"avg_inliers={np.mean(res['fm_inliers']):.0f})")
    if res['rc']:
        print(f"Render-Compare:  {med(res['rc'])}  (n={len(res['rc'])})")

    out = {
        'coarse_median_cm':   float(np.median(res['coarse'])*100),
        'fm_median_cm':       float(np.median(res['fm'])*100) if res['fm'] else None,
        'rc_median_cm':       float(np.median(res['rc'])*100) if res['rc'] else None,
        'fm_success_rate':    len(res['fm'])/len(test_names),
        'fm_avg_inliers':     float(np.mean(res['fm_inliers'])) if res['fm_inliers'] else 0,
    }
    os.makedirs('reloc', exist_ok=True)
    with open('reloc/results_v2.json','w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved reloc/results_v2.json")

if __name__ == "__main__":
    main()
