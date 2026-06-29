# demov2.py 文档

## 使用方法

### 基本用法

```bash
cd /path/to/HaWoR
conda activate hawor

python demov2.py --video_path <视频路径>
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--video_path` | str | `example/video_0.mp4` | **必填**，输入视频文件路径 |
| `--img_focal` | float | None | 手动指定焦距（像素）。不指定则自动搜索或从est_focal.txt读取 |
| `--input_type` | str | `file` | 输入类型，目前只支持 `file` |
| `--checkpoint` | str | `./weights/hawor/checkpoints/hawor.ckpt` | HAWOR模型权重路径 |
| `--infiller_weight` | str | `./weights/hawor/checkpoints/infiller.pt` | Infiller模型权重路径 |

### 示例

```bash
# 基本用法：自动处理视频
python demov2.py --video_path /path/to/video.mp4

# 手动指定焦距（适用于已知相机内参的情况）
python demov2.py --video_path /path/to/video.mp4 --img_focal 1056

# 使用自定义权重路径
python demov2.py --video_path /path/to/video.mp4 \
    --checkpoint ./weights/hawor/checkpoints/hawor.ckpt \
    --infiller_weight ./weights/hawor/checkpoints/infiller.pt
```

### 输出

所有输出统一存放在 `HaWoR/output/<video_name>/` 目录下，`<video_name>` 为视频文件名（不含扩展名）。

运行完成后会打印完整的输出摘要，包括数据来源、数据文件列表和可视化文件列表。

### 运行时间

| 阶段 | 首次运行 | 有缓存 |
|------|---------|--------|
| 检测+跟踪 | ~2min | 跳过 |
| 运动估计 | ~1min | 跳过 |
| SLAM | ~3min | 跳过 |
| 焦距搜索 | ~5min（仅默认值600时触发） | 跳过 |
| Infiller | ~30s | 跳过 |
| 渲染 | ~2min | ~2min |
| **总计** | **~10min** | **~2min** |

缓存机制：如果中间数据（tracks、SLAM、cam_space等）已存在，对应阶段会自动跳过。如需重新运行某个阶段，删除对应的中间目录即可。

### 读取输出数据

```python
import numpy as np, joblib, torch

output_dir = 'output/hoi4d'

# 1. 读取世界空间手部参数（最常用）
ws = joblib.load(f'{output_dir}/world_space_res.pth')
pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = ws
# pred_trans:     (2, N, 3)  手部平移 [0=左, 1=右]
# pred_rot:       (2, N, 3)  全局旋转（轴角）
# pred_hand_pose: (2, N, 45) 关节旋转（轴角，15关节×3）
# pred_betas:     (2, N, 10) MANO形状参数
# pred_valid:     (2, N)     有效帧标记

# 2. 读取SLAM相机轨迹
slam = dict(np.load(f'{output_dir}/SLAM/hawor_slam_w_scale_0_600.npz', allow_pickle=True))
traj = slam['traj']        # (N, 7) [tx, ty, tz, qx, qy, qz, qw]
scale = float(slam['scale'])
focal = float(slam['img_focal'])
center = slam['img_center'] # [cx, cy]

# 3. 读取合并的重建结果
res = np.load(f'{output_dir}/reconstruction/hawor_results_0_600.npz', allow_pickle=True)
print(list(res.keys()))
# ['pred_trans', 'pred_rot', 'pred_hand_pose', 'pred_betas', 'pred_valid',
#  'R_c2w', 't_c2w', 'img_focal', 'img_center', 'slam_scale', 'start_idx', 'end_idx']

# 4. 使用 load_slam_cam 加载相机位姿
from lib.eval_utils.custom_utils import load_slam_cam
R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(f'{output_dir}/SLAM/hawor_slam_w_scale_0_600.npz')

# 5. 使用 MANO 生成手部 mesh
from hawor.utils.process import run_mano
outputs = run_mano(pred_trans[1:2], pred_rot[1:2], pred_hand_pose[1:2], betas=pred_betas[1:2])
vertices = outputs['vertices']  # (1, N, 778, 3)
joints = outputs['joints']      # (1, N, 21, 3)
```

---

## Pipeline 概述

demov2.py 是 HaWoR 手部重建的完整 pipeline，从视频输入到3D手部重建和可视化。

### Pipeline 流程

```
输入视频
  │
  ├─ Stage 1: detect_track_video()
  │   ├─ ffmpeg 提取帧 → extracted_images/
  │   ├─ 检测手部 → model_boxes.npy
  │   └─ 跟踪手部 → model_tracks.npy, model_masks.npy, frame_chunks_all.npy
  │
  ├─ Stage 2: hawor_motion_estimation()
  │   ├─ 读取帧块，HAWOR模型推理
  │   ├─ 输出相机空间手部参数 → cam_space/
  │   └─ 返回 img_focal（焦距）
  │
  ├─ Stage 3: hawor_slam()
  │   ├─ DROID-SLAM 估计相机轨迹
  │   ├─ 自动搜索焦距（如果 est_focal.txt 为默认值600）
  │   └─ 输出 → SLAM/hawor_slam_w_scale_*.npz
  │       ├─ traj: (N, 7) 相机轨迹 [tx, ty, tz, qx, qy, qz, qw]
  │       ├─ scale: 单目深度缩放因子
  │       ├─ img_focal: 焦距
  │       └─ img_center: 主点
  │
  ├─ Stage 4: hawor_infiller()
  │   ├─ cam2world_convert: 相机空间 → 世界空间（使用SLAM相机位姿）
  │   ├─ Transformer infiller: 填充未检测到手的帧
  │   └─ 输出 → world_space_res.pth
  │       ├─ pred_trans: (2, N, 3) 手部平移
  │       ├─ pred_rot: (2, N, 3) 手部全局旋转（轴角）
  │       ├─ pred_hand_pose: (2, N, 45) 手部关节旋转（轴角）
  │       ├─ pred_betas: (2, N, 10) MANO形状参数
  │       └─ pred_valid: (2, N) 有效帧标记
  │
  ├─ Stage 5: MANO前向传播 + 坐标变换
  │   ├─ run_mano / run_mano_left: 生成手部顶点和关节
  │   ├─ R_x变换: SLAM坐标系(Y-up) → OpenCV坐标系(Y-down)
  │   └─ 计算 R_w2c, t_w2c 用于投影
  │
  └─ Stage 6: 可视化渲染
      ├─ vis_cam: 相机视角叠加（手部mesh投影回原始帧）
      ├─ vis_world: 3D世界视角（俯瞰场景）
      └─ vis_verify: 验证视图（叠加+深度+轨迹+信息面板）
```

### 函数调用说明

| 函数 | 来源 | 说明 |
|------|------|------|
| `detect_track_video(args)` | `scripts/scripts_test_video/detect_track_video.py` | 视频帧提取、手部检测与跟踪 |
| `hawor_motion_estimation(args, start, end, seq_folder)` | `scripts/scripts_test_video/hawor_video.py` | HAWOR模型推理，输出相机空间手部参数 |
| `hawor_slam(args, start, end)` | `scripts/scripts_test_video/hawor_slam.py` | DROID-SLAM相机轨迹估计 |
| `hawor_infiller(args, start, end, frame_chunks)` | `scripts/scripts_test_video/hawor_video.py` | 相机空间→世界空间转换 + 缺失帧填充 |
| `load_slam_cam(slam_path)` | `lib/eval_utils/custom_utils.py` | 加载SLAM相机位姿(R_c2w, t_c2w, R_w2c, t_w2c) |
| `run_mano(trans, rot, pose, betas)` | `hawor/utils/process.py` | MANO右手前向传播，输出vertices和joints |
| `run_mano_left(trans, rot, pose, betas)` | `hawor/utils/process.py` | MANO左手前向传播 |
| `get_mano_faces()` | `hawor/utils/process.py` | 获取MANO面片索引 |
| `cam2world_convert(R_c2w, t_c2w, data, handedness)` | `lib/eval_utils/custom_utils.py` | 相机空间→世界空间坐标转换 |

### 输出目录结构

```
output/<video_name>/
├── SLAM/                              # SLAM相机轨迹数据
│   └── hawor_slam_w_scale_0_N.npz     # 相机位姿 + scale + 焦距 + 主点
├── cam_space/                         # 相机空间手部参数
│   ├── 0/                             # 左手
│   │   └── start_end.json             # {init_trans, init_root_orient, init_hand_pose, init_betas}
│   └── 1/                             # 右手
│       └── start_end.json
├── tracks_0_N/                        # 检测与跟踪数据
│   ├── model_boxes.npy                # 手部边界框
│   ├── model_masks.npy                # 手部mask
│   ├── model_tracks.npy               # 手部跟踪
│   └── frame_chunks_all.npy           # 帧块划分
├── extracted_images/                  # 提取的视频帧
│   ├── 0000.jpg
│   ├── 0001.jpg
│   └── ...
├── world_space_res.pth                # 世界空间手部参数（核心输出）
├── est_focal.txt                      # 焦距估计值
├── vis_cam_0_N/                       # 相机视角叠加图
├── vis_world_0_N/                     # 3D世界视角
│   └── world_view.mp4
├── vis_verify/                        # 验证视图
│   ├── hand_0/
│   │   └── verify.mp4
│   └── hand_1/
│       └── verify.mp4
└── reconstruction/                    # 重建结果
    └── hawor_results_0_N.npz          # 所有数据合并
```

---

## Q&A - 关键问题

### Q1: 相机外参和手部数据是固定的还是从视频估计的？
全部从视频估计，没有任何固定/硬编码的数据：
- 相机外参 (R_c2w, t_c2w): SLAM从视频估计
- 手部参数 (pred_trans, pred_rot, pred_hand_pose, pred_betas): MANO从视频推理
- 焦距 (img_focal): SLAM通过重投影误差搜索得到（或从est_focal.txt读取）
- 主点 (cx, cy): SLAM输出（或图像中心）
- SLAM scale: SLAM估计的单目深度缩放因子

### Q2: World View中的相机轨迹是真实的吗？
是的。World View中的金字塔标记使用SLAM估计的真实相机位姿(R_c2w, t_c2w)。
World View的观察相机是虚拟的（用于3D可视化），但场景中的相机轨迹和手部位置都是真实数据。
相机移动不多是正常的——取决于视频内容。

### Q3: SLAM scale不等于1有问题吗？
没问题。SLAM scale是单目SLAM的固有特性，用于将相对深度转换为绝对尺度。
scale的值取决于SLAM的初始化和焦距估计。焦距越准确，scale越接近1，
但scale≠1不代表结果不正确。关键是手部投影是否与视频对齐。

### Q4: 坐标系变换链是什么？
MANO输出(世界空间, SLAM坐标系) → R_x翻转(Y/Z) → 世界空间(OpenCV坐标系)
→ R_w2c变换 → 相机空间 → 透视投影 → 2D像素
其中 R_x = diag(1,-1,-1) 将SLAM的OpenGL坐标系(Y-up)转为OpenCV坐标系(Y-down)

### Q5: 焦距如何确定？
优先级: 命令行参数 > est_focal.txt > SLAM焦距搜索(重投影误差最小化)
如果est_focal.txt包含默认值600但图像尺寸暗示焦距应更大，SLAM会自动搜索。
