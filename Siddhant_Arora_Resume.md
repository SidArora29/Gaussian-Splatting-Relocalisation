**SIDDHANT ARORA**

\+91 8130840105  |  [siddhantarora396@gmail.com](mailto:siddhantarora396@gmail.com)  |  [linkedin.com/in/siddhantarora369](https://linkedin.com/in/siddhantarora369)  |  [github.com/SidArora29](https://github.com/SidArora29)

**EDUCATION**

**University of Michigan, Ann Arbor**  —  MS Robotics  						    *Aug 2026 \- May 2028*

*Coursework: Math for Robotics · Self-Driving Cars · Action and Perception*

**Vellore Institute of Technology**  —  BTech CSE (AI & Robotics)  |  CGPA 9.18/10 | 3.93/4.0    		      *Aug 2021 \- Sep 2025*

**PROFESSIONAL EXPERIENCE**

**WISERLI**  |  Robotics Software Engineer  —  *Nagpur, IN  						          Jul 2025 \- Jul 2026*

* **VIO on legged hardware:** Deployed a **Visual-Inertial Odometry** pipeline on a **Jetson Orin** quadruped with **Intel RealSense D457** via IMU preintegration and camera-IMU extrinsic calibration — cutting trajectory RMSE **34%** over baseline VSLAM across featureless corridors where classical SLAM fails outright.

* **Loop closure in GPS-denied navigation:** Eliminated external positioning dependency by designing a keyframe-based visual relocalization algorithm with pose-graph loop closure — cutting localization failure rate **67%** and sustaining autonomous traversal across corridors exceeding **200 m** without GPS anchor.

* **Real-time IBVS on embedded hardware:** Closed the perception-control loop at **35 Hz** with **sub-12 ms** end-to-end latency by implementing **Image-Based Visual Servoing** for moving obstacle tracking on Jetson Orin — validated under multi-obstacle conditions at operational robot speeds.

**MITACS Globalink**  |  Research Scholar  —  *Montreal, CA   						     May 2024 \- Aug 2024*

* Reduced inter-robot trajectory tracking RMSE **28%** and cut protective formation convergence from **10s to 6s** by architecting a decentralized 4-robot coordination system via real-time sensor fusion and a dynamic Voronoi boundary algorithm.

* Achieved zero-collision autonomous flight across **12 outdoor deployments** in GPS-degraded environments by designing path planning algorithms for the **DJI Mini 4 Pro** fusing LiDAR, stereo depth, and IMU measurements.

**Dassault Systèmes**  |  Deep Learning Intern  —  *Gurugram, IN  					      Aug 2023 – Jan 2024*

* Deployed **YOLO-NAS** warehouse monitoring at **96.7% detection accuracy**; refined **MoveNet \+ BlazePose** pose estimation, improving keypoint localization **10%** across 17 joint landmarks for ergonomic safety compliance.

**ACADEMIC PROJECTS**

**3D Gaussian Splatting for Robot Camera Relocalization**  [\[GitHub\]](https://github.com/SidArora29/vio-benchmark)                                                                                        *Jun 2026 \- Jul 2026*

* Trained a **544k-Gaussian 3DGS scene map** from 183 phone-captured frames (COLMAP SfM) achieving a **PSNR** of **37.47 dB.**  
* Built a **6-DoF relocalization pipeline** matching **SuperPoint+LightGlue** features against rendered depth maps via **PnP+RANSAC**, reaching **25.9 cm median error** across **20** queries achieving a **43% improvement over coarse prior.**

**Custom Stereo VIO System — GTSAM Factor Graph \+ ISAM2**  [\[GitHub\]](https://github.com/SidArora29/gtsam-stereo-vio) 					      *May 2026 \- Jun 2026*

* Built stereo VIO from scratch using **GTSAM factor graphs**, on-manifold **IMU preintegration**, and **ISAM2** — reaching **0.997 m ATE RMSE** on EuRoC MH\_01\_easy, within 5× of VINS-Mono on identical ground-truth and evo\_ape harness.

* Conducted a controlled frontend ablation integrating **SuperPoint \+ LightGlue** as a temporal tracker replacement — quantifying **2.4× higher RMSE** vs classical SGBM at 20 Hz and tracing failure to non-deterministic per-frame keypoint selection, characterising when learned matchers help vs hurt in a production VIO stack.

**Mira: Autonomous Underwater Vehicle**  [\[GitHub\]](https://github.com/SidArora29/mira-auv)                                                                                                                    *Oct 2022 \- Jun 2024*

* Built stereo visual odometry (**CalypsoVO**) with a custom learned feature descriptor \+ 1-point RANSAC — cutting cumulative drift **40%** vs ORB-SLAM2 on turbid featureless sequences; developed GAN-based image restoration correcting colour casts and turbidity to recover **20%** downstream feature detection accuracy.

* Delivered **sub-3 cm** docking alignment under active current disturbances via cascaded PID \+ ArUco fiducials. 

* **Led an 18-member team to 2nd globally and 1st in Asia** at Tau Autonomy Challenge 2024, Norway.

**SKILLS**

**State Estimation & SLAM:** Visual-Inertial Odometry (VIO), Visual SLAM, 3D Scene Reconstruction, Monocular Depth Estimation, GTSAM, ISAM2, IMU Preintegration, Factor Graphs, EKF/UKF, Camera-IMU Calibration, Stereo Rectification, SGBM, Pose-Graph Optimization, IBVS

**Learned Perception:** SuperPoint, LightGlue, Neural Radiance Fields (NeRF), 3D Gaussian Splatting, YOLO-NAS, MoveNet, U-Net, ResNet, GAN-based Image Restoration, Neural Depth Estimation, PyTorch, TensorFlow, OpenCV

**Robotics Systems:** ROS 1/2, NVIDIA Jetson Orin, Intel RealSense D457, LiDAR, MEMS IMUs, Stereo Cameras, Embedded Edge Deployment, Real-Time Systems, C++, Python

**Evaluation & Tools:** evo (Trajectory Evaluation), EuRoC MAV Dataset, Docker, Git, Linux, Isaac Sim, Gazebo

**Awards:** 2nd globally / 1st Asia — Tau Autonomy Challenge 2024 (Norway, 15+ teams)  |  India representative — CERN Beamline for Schools (top 300+ globally)  |  Vice Captain, Team Dreadnought Robotics (100+ members, ROS2 training)