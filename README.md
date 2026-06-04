<div align="center">

# HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos

> **Based on**: [ThunderVVV/HaWoR](https://github.com/ThunderVVV/HaWoR) - CVPR 2025 Highlight
>
> **Paper**: [arXiv:2501.02973](https://arxiv.org/abs/2501.02973)
>
> **Original Authors**: Jinglei Zhang, Jiankang Deng, Chao Ma, Rolandos Alexandros Potamias

This is our customized fork of HaWoR, adapted for the **Ego-Video-to-SIM** pipeline.

</div>

## What's Changed from Upstream

| Feature | Description |
|---|---|
| `demov2.py` | Enhanced reconstruction pipeline with rich visualization outputs |
| `hawor_detect_wrapper.py` | Wrapper script for hand detection + mask generation (called by HaworArmSegmenter) |
| `assess/` | Test suites: sanity, synthetic, and example-data tests |
| `tools/` | Quality evaluation, video rendering, reconstruction viewing, verification visualization |

## demov2: Enhanced Reconstruction Pipeline

`demov2.py` is the core entry point for our pipeline. Compared to the original `demo.py`, it generates **three types of visualization** and **one reconstruction data file**:

### Output Structure

```
output/<video_name>/
├── vis_cam_<start>_<end>/          # Camera-view overlay
│   ├── 000000.png                  # Hand mesh projected onto original frames
│   ├── 000001.png
│   └── ...
├── vis_world_<start>_<end>/        # World-view 3D rendering
│   ├── 000000.png                  # Top-down 3D scene with hand meshes + camera trajectory
│   ├── 000005.png
│   └── world_view.mp4              # Full world-view video
├── vis_verify/                     # Verification visualization
│   ├── hand_0/                     # Left hand verification
│   │   ├── frame_0000.png          # Composite: original + overlay + depth + info panel + trajectory
│   │   ├── frame_0005.png
│   │   └── verify.mp4
│   └── hand_1/                     # Right hand verification
│       ├── frame_0000.png
│       └── verify.mp4
└── reconstruction/                 # Reconstruction data
    └── hawor_results_<start>_<end>.npz
```

### Output Details

| Output | Content | Purpose |
|---|---|---|
| **vis_cam/** | Original frame + semi-transparent hand mesh overlay | Verify hand projection aligns with video |
| **vis_world/** | 3D scene: ground plane (checkerboard) + camera trajectory (pyramids) + hand meshes | Observe hand motion in world coordinates |
| **vis_verify/** | 4-panel composite: original / overlay / depth map / info+trajectory | Detailed per-frame verification with depth values, focal length, SLAM scale |
| **reconstruction/*.npz** | `pred_trans`, `pred_rot`, `pred_hand_pose`, `pred_betas`, `pred_valid`, `R_c2w`, `t_c2w`, `img_focal`, `img_center`, `slam_scale` | Complete reconstruction data for downstream use |

### Coordinate System

```
MANO output (SLAM coordinate, Y-up)
    → R_x = diag(1,-1,-1) flip
    → World space (OpenCV coordinate, Y-down)
    → R_w2c transform
    → Camera space
    → Perspective projection → 2D pixels
```

### Usage

```bash
# Basic usage
python demov2.py --video_path ./example/video_0.mp4

# With custom focal length
python demov2.py --video_path ./example/video_0.mp4 --img_focal 600.0

# With custom model weights
python demov2.py --video_path ./example/video_0.mp4 \
    --checkpoint ./weights/hawor/checkpoints/hawor.ckpt \
    --infiller_weight ./weights/hawor/checkpoints/infiller.pt
```

## hawor_detect_wrapper

A lightweight wrapper that runs hand detection + mask generation, designed to be called by `HaworArmSegmenter` via subprocess:

```bash
conda run -n hawor python hawor_detect_wrapper.py --video_path <video> [--img_focal <focal>]
```

Output: `model_masks.npy` in the sequence folder.

## Tools

| Tool | Description |
|---|---|
| `tools/vis_verify.py` | Standalone verification visualization |
| `tools/render_video.py` | Render reconstruction results as video |
| `tools/view_reconstruction.py` | Interactive 3D reconstruction viewer |
| `tools/evaluate_quality.py` | Quantitative quality evaluation |

## Installation

See upstream [HaWoR](https://github.com/ThunderVVV/HaWoR) for installation instructions.

Additional dependencies:
```bash
pip install easydict
```

## Acknowledgement

This project is built upon [HaWoR](https://github.com/ThunderVVV/HaWoR). We thank the original authors for their excellent work.

## Citation

```bibtex
@article{zhang2025hawor,
      title={HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos},
      author={Zhang, Jinglei and Deng, Jiankang and Ma, Chao and Potamias, Rolandos Alexandros},
      journal={arXiv preprint arXiv:2501.02973},
      year={2025}
}
```
