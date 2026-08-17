# 3D Gaussian Splatting for Robot Camera Relocalization

Visual relocalization pipeline using 3D Gaussian Splatting as a photorealistic map representation. Given a single query image from an unknown pose in a mapped environment, the system estimates the full 6-DoF camera pose by matching SuperPoint features against a rendered depth map of the 3DGS scene - no GPS, no fiducials, no infrastructure.

---

## Why 3DGS for relocalization

Classical map representations (sparse point clouds, occupancy grids) discard photometric information that would help feature matching. 3DGS retains it: the trained map can render a photorealistic RGB image **and** a metrically accurate depth map from any viewpoint in real time. This means feature matching is done against a rendered view from the estimated camera pose, not against raw point cloud projections. Hence, making correspondences denser and more reliable.

This project compares three approaches to pose estimation:
- **Coarse prior**: nearest-neighbour training pose with Gaussian noise (simulates IMU dead-reckoning)
- **Render-Compare**:  gradient-free photometric optimisation via Nelder-Mead
- **Feature-Match**:  SuperPoint + LightGlue matching against rendered depth → PnP+RANSAC

---

## Pipeline

```
Phone video → COLMAP (SfM) → camera poses + sparse 3D points
                                        ↓
                          3DGS training (30k iterations)
                                        ↓
                          Trained Gaussian map (544k Gaussians)
                                        ↓
Query image → SuperPoint features
                    ↓
            Render 3DGS from coarse pose → RGB + depth map
                    ↓
            LightGlue match: query features ↔ rendered features
                    ↓
            Back-project matched rendered points to 3D via depth
                    ↓
            PnP + RANSAC → refined 6-DoF pose
```

---

## Results

### 3DGS Reconstruction Quality

Evaluated on Tanks & Temples truck scene (standard 3DGS benchmark):

| Metric | Published 3DGS | This implementation |
|--------|---------------|---------------------|
| PSNR ↑ | 25.19 dB | **27.68 dB** |
| SSIM ↑ | 0.879 | **0.920** |
| LPIPS ↓ | 0.147 | **0.128** |

Custom indoor scene (183 phone-captured frames, COLMAP-posed):

| Metric | Value |
|--------|-------|
| PSNR   | 37.47 dB |
| Training time | 27 min (RTX 4060) |
| Gaussians | 544,643 |

### Camera Relocalization (20 held-out test queries, indoor scene)

| Method | Median position error | Success rate | Avg PnP inliers |
|--------|-----------------------|-------------|----------------|
| Coarse prior (NN + 30cm noise) | 45.1 cm | 20/20 | — |
| Render-Compare (Nelder-Mead) | 31.1 cm | 20/20 | — |
| **Feature-Match (SuperPoint + LightGlue + PnP)** | **25.9 cm** | **20/20** | **218** |

Feature-Match achieves **43% error reduction** over the coarse prior. Best single-query result: **3.1 cm** with 545 inliers (frame 0118).

---

## Visual Results

### Reconstruction quality: original vs 3DGS render

<!-- Add comparison.gif here -->
![Comparison](assets/my_scene_comparison.gif)

*Left: original phone capture. Right: 3DGS novel-view synthesis from same viewpoint.*

### 3D Gaussian cloud (interactive)

Drag `output/my_scene/point_cloud/iteration_30000/point_cloud.ply` onto [antimatter15.com/splat](https://antimatter15.com/splat/) for interactive 3D viewer.

---

## Key engineering decisions

**COLMAP for camera pose estimation.** Phone video → ffmpeg frame extraction → COLMAP automatic reconstructor → sparse point cloud + per-frame camera poses. All 183 frames registered. Final bundle adjustment reprojection error: 0.883 px (below 1.0 is good).

**Half-resolution training (`-r 2`).** Full-resolution 1080p phone frames exhaust 8GB VRAM during SSIM computation at high Gaussian density. Training at half resolution (531×945) reduces peak VRAM by 4× with negligible quality loss at this Gaussian count.

**SGBM-free depth: 3DGS renders metrically accurate depth natively.** Unlike the VIO benchmarking project (which required SGBM dense stereo), the 3DGS renderer outputs a depth map as a byproduct of Gaussian rasterization with no additional computation. This eliminates the stereo-camera requirement entirely and a single monocular camera suffices for map-building and relocalization.

**SuperPoint + LightGlue for sparse-to-3D matching.** KLT optical flow (used in the VIO project for temporal tracking) is unsuitable here because rendered and query images have different photometric properties despite depicting the same scene. SuperPoint's learned detector is viewpoint-invariant; LightGlue's transformer-based matcher handles the appearance gap between rendered and real images. Average 218 inliers per query that is substantially denser than classical ORB-based relocalization on sparse maps.

**PnP+RANSAC for metric pose recovery.** Matched 2D query keypoints are paired with 3D world coordinates back-projected through the 3DGS depth map. `cv2.solvePnPRansac` with EPNP + 3000 iterations recovers the full 6-DoF camera pose metrically. 4.0 px reprojection threshold chosen empirically for portrait-format 531×945 images.

**Nelder-Mead for render-compare baseline.** Autograd through the 3DGS rasterizer is possible but requires careful handling of the custom CUDA kernels. Gradient-free Nelder-Mead on the 6-parameter pose (3 axis-angle + 3 translation delta) is more reliable as a baseline — 150 renderer evaluations per query, ~3 min/image.

---

## Ablation: learned vs classical feature matching

The VIO benchmarking project found SuperPoint + LightGlue produced **2.4× higher RMSE** than classical SGBM for 20 Hz temporal tracking because learned wide-baseline matchers are poorly suited to high-frequency consecutive-frame matching where inter-frame motion is small.

Here the result reverses: LightGlue is the correct tool because relocalization **is** a wide-baseline matching problem. The query and the rendered reference image are taken from different times, potentially different lighting, and with the photometric gap between real and rendered imagery. Classical ORB/BRIEF descriptors are not robust to the real-vs-rendered appearance gap; SuperPoint's convolutional features generalize across it.

This asymmetry — LightGlue beats classical for wide-baseline, KLT beats LightGlue for temporal — is the core insight connecting this project to the VIO benchmarking work.

---

## Failure modes

**Large coarse error (>80 cm).** Frames 3, 4, 15, 18 have coarse errors above 80 cm. When the rendered view diverges too far from the query, overlap in feature space drops and PnP inliers fall below threshold. Fix: try top-3 nearest neighbours as candidates and take the best-inlier result.

**Scene coverage gaps.** If the map was captured with insufficient angular coverage of a region, rendering from a query pose in that region produces a partially occluded or blurry view. More training images, particularly at different heights and angles, would reduce this.

---

## Reproducing

```bash
# 1. Setup
conda activate gaussian_splatting
cd ~/gaussian-splatting

# 2. Capture your scene (phone video → frames)
mkdir -p my_scene/input
ffmpeg -i your_video.mp4 -vf fps=2 my_scene/input/%04d.jpg

# 3. COLMAP reconstruction
colmap automatic_reconstructor \
    --workspace_path my_scene \
    --image_path my_scene/input \
    --camera_model PINHOLE \
    --single_camera 1

# 4. Convert + train
python convert.py -s my_scene
python train.py -s my_scene --model_path output/my_scene --iterations 30000 -r 2

# 5. Relocalization
python reloc/relocalize_v2.py \
    --model_path output/my_scene \
    --scene_path my_scene \
    --n_test 20 \
    --noise_m 0.3
```

---

## Environment

- GPU: NVIDIA RTX 4060 Laptop (8 GB VRAM)
- CUDA: 12.1 / Driver 590
- PyTorch: 2.3.0+cu121
- Python: 3.10

---

## References

- Kerbl et al. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* ACM SIGGRAPH.
- DeTone et al. (2018). *SuperPoint: Self-Supervised Interest Point Detection and Description.* CVPRW.
- Lindenberger et al. (2023). *LightGlue: Local Feature Matching at Light Speed.* ICCV.
- Schönberger & Frahm (2016). *Structure-from-Motion Revisited.* CVPR.
