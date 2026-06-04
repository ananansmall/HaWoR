# HaWoR 手部重建 - 使用指南

## 目录结构

运行 `demov2.py` 后，每个视频示例文件夹的结构如下：

```
example/beizi/
├── extracted_images/          # 从视频提取的帧图像 (0000.jpg, 0001.jpg, ...)
├── est_focal.txt              # 估算的焦距 (像素)
├── tracks_0_184/              # 检测与跟踪结果 (帧 0~183)
│   ├── model_tracks.npy       # 手部跟踪轨迹
│   ├── model_masks.npy        # 手部掩码 (用于 SLAM 遮挡)
│   ├── model_boxes.npy        # 检测框 (可能为空)
│   └── frame_chunks_all.npy   # 帧分块信息
├── cam_space/                 # 相机空间手部参数 (每只手一个 JSON)
│   └── 1/                     # 手的 ID (0=左手, 1=右手)
│       └── 6_183.json         # {init_trans, init_root_orient, init_hand_pose, init_betas}
├── SLAM/                      # DROID-SLAM 相机轨迹
│   └── hawor_slam_w_scale_0_184.npz
├── reconstruction/            # ★ 世界坐标系重建结果
│   └── hawor_results_0_184.npz
├── world_space_res.pth        # 世界空间重建结果 (经 R_x 变换, 旧格式)
├── vis_cam_0_183/             # ★ 相机视角叠加渲染 (手部网格叠加在原始帧上)
│   ├── 000000.png             # 每帧一张 PNG
│   ├── 000001.png
│   └── ...
├── vis_world_0_183/           # ★ 世界视角 3D 场景渲染
│   ├── world_view.mp4         # 3D 场景视频
│   ├── 000000.png             # 每5帧一张 PNG
│   └── ...
└── vis_verify/                # ★ 可视化验证输出 (深度+轨迹+信息)
    └── hand_1/
        ├── verify.mp4         # 验证视频
        └── frame_*.png        # 逐帧验证图
```

---

## ★ 数据与视频帧的对应关系

这是理解数据真实性的关键。所有数据文件的帧索引都对应 `extracted_images/` 中的图像编号。

### 帧编号对应表

| 数据文件 | 帧索引含义 | 示例 |
|----------|-----------|------|
| `extracted_images/0006.jpg` | 全局帧号 = 6 | 第 7 帧图像 (从 0 开始) |
| `SLAM/.../hawor_slam_w_scale_0_184.npz` | `traj[i]` 对应帧号 `i` (0~183) | `traj[6]` = 帧 6 的相机位姿 |
| `cam_space/1/6_183.json` | 文件名 `6_183` 表示帧 6~183 | `init_trans[0, i]` 对应帧 `6+i` |
| `model_tracks.npy` | 每个条目有 `frame` 字段 | `tracks[1][i]["frame"]` = 帧号 |
| `model_masks.npy` | `masks[i]` 对应帧 `i` | `masks[6]` = 帧 6 的手部掩码 |
| `world_space_res.pth` | `pred_trans[hand, i]` 对应帧 `i` | `pred_trans[1, 6]` = 帧 6 的右手平移 |

### 具体对应示例 (beizi)

```
帧 0~5:   无手部检测 (cam_space 不包含这些帧)
帧 6:     手部首次检测 (track_id=10000, 仅此 1 帧)
帧 7~183: 手部持续跟踪 (track_id=1, 共 177 帧)
帧 6~183: cam_space/1/6_183.json 中 init_trans[0, 0~177] 依次对应
          init_trans[0, 0] → 帧 6
          init_trans[0, 1] → 帧 7
          ...
          init_trans[0, 177] → 帧 183
```

### 读取代码模板

```python
import json, numpy as np, torch
import sys; sys.path.insert(0, '.')
from hawor.utils.process import run_mano
from hawor.utils.rotation import rotation_matrix_to_angle_axis

# 1. 加载相机空间手部参数
with open('example/beizi/cam_space/1/6_183.json') as f:
    cam = json.load(f)
start_frame, end_frame = 6, 183
num_hand_frames = end_frame - start_frame + 1  # 178

# 2. 获取某帧的手部数据
target_frame = 50  # 想看帧 50
idx = target_frame - start_frame  # = 44
trans_50 = np.array(cam['init_trans'])[0, idx]       # 帧 50 的手腕平移
orient_50 = np.array(cam['init_root_orient'])[0, idx] # 帧 50 的手腕旋转

# 3. 运行 MANO 获取 3D 网格
data = {k: torch.tensor(v) for k, v in cam.items()}
root_aa = rotation_matrix_to_angle_axis(data["init_root_orient"])
pose_aa = rotation_matrix_to_angle_axis(data["init_hand_pose"])
outputs = run_mano(data["init_trans"], root_aa, pose_aa, betas=data["init_betas"])
verts = outputs['vertices'][0].cpu().numpy()  # (178, 778, 3)
joints = outputs['joints'][0].cpu().numpy()    # (178, 21, 3)

# 4. 投影到 2D 图像
focal = 600.0
cx, cy = 640.0, 360.0  # 1280x720 图像中心
j50 = joints[idx]  # 帧 50 的 21 个关节
u = focal * j50[:, 0] / j50[:, 2] + cx
v = focal * j50[:, 1] / j50[:, 2] + cy
# (u, v) 就是关节在图像上的像素坐标

# 5. 加载 SLAM 相机轨迹
from lib.eval_utils.custom_utils import load_slam_cam
R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(
    'example/beizi/SLAM/hawor_slam_w_scale_0_184.npz')
# R_c2w[50] = 帧 50 的 camera-to-world 旋转矩阵
# t_c2w[50] = 帧 50 的 camera-to-world 平移向量
```

---

## 全部数据文件详细说明

### 1. `est_focal.txt`

纯文本文件，包含一个浮点数——相机焦距（像素）。

```
600
```

**来源优先级**: 命令行 `--img_focal` > `est_focal.txt` > 默认值 600

**影响范围**: MANO 深度估计 (`tz = 2*focal/bbox_size`)、SLAM 内参、渲染投影

---

### 2. `extracted_images/`

从视频提取的帧图像，文件名格式 `{帧号:04d}.jpg`。

- `0000.jpg` = 第 0 帧
- `0001.jpg` = 第 1 帧
- ...

**帧号是所有数据文件的索引基准。**

---

### 3. `tracks_0_184/` - 检测与跟踪结果

文件夹名 `0_184` 表示覆盖帧 0~183（共 184 帧）。

#### `model_tracks.npy`

手部跟踪轨迹，`dict` 类型，key 为 track_id。

```python
tracks = np.load('tracks_0_184/model_tracks.npy', allow_pickle=True).item()

# tracks[track_id] = list of dict, 每个 dict 包含:
#   "frame": int        - 帧号 (对应 extracted_images/)
#   "det": bool         - 该帧是否检测到手
#   "det_box": list     - 检测框 [[x1, y1, x2, y2, conf], ...]
#   "det_handedness":   - 左/右手信息

# beizi 示例:
# tracks[10000] = 1 帧 (帧 6, 初始检测)
# tracks[1] = 177 帧 (帧 7~183, 持续跟踪)
```

**track_id 含义**:
- `10000` = 初始检测帧（仅 1 帧，用于初始化跟踪器）
- `1` = 右手持续跟踪
- `0` = 左手持续跟踪

#### `model_masks.npy`

手部掩码，用于 SLAM 遮挡（避免手部区域干扰特征匹配）。

```python
masks = np.load('tracks_0_184/model_masks.npy', allow_pickle=True)
# shape: (184, 720, 1280), dtype: bool
# masks[i] = 帧 i 的手部掩码 (True = 手部区域)
```

**与视频帧对应**: `masks[i]` 对应 `extracted_images/{i:04d}.jpg`

#### `model_boxes.npy`

检测框，可能为空（被 `model_tracks.npy` 替代）。

#### `frame_chunks_all.npy`

帧分块信息，`defaultdict` 类型，用 `joblib` 保存。

```python
import joblib
chunks = joblib.load('tracks_0_184/frame_chunks_all.npy')
# chunks[1] = [[6, 7, 8, ..., 183]]  - track_id=1 的帧列表
```

---

### 4. `cam_space/{hand_id}/{start}_{end}.json`

相机空间的手部 MANO 参数，由 `hawor_motion_estimation` 生成。

- `hand_id`: 0=左手, 1=右手
- `start_end`: 帧范围 (对应 extracted_images/ 中的帧号)

```python
import json
with open('cam_space/1/6_183.json') as f:
    data = json.load(f)

# data 包含 (shape 中的 T = end-start+1 = 178):
# init_trans:       (1, T, 3)       - 相机空间手腕平移 [tx, ty, tz]
#                                    tz > 0 表示手在相机前方
#                                    tz 越小 = 手越近
# init_root_orient: (1, T, 3, 3)    - 手腕旋转矩阵
# init_hand_pose:   (1, T, 15, 3, 3) - 15个关节的旋转矩阵
# init_betas:       (1, T, 10)      - MANO shape 参数
```

**与视频帧对应**: `data['init_trans'][0, i]` 对应帧 `start + i`

**坐标系**: 相机空间 (OpenCV 约定: x-right, y-down, z-forward)

**深度含义**: `tz` 值代表手腕到相机的距离（米），由弱透视模型计算：
```
tz = 2 * focal / (bbox_size * scale_factor)
```
- `focal=600` 时，一个 200px 的手 bbox → `tz ≈ 2*600/200 = 6.0` (约 6 米)
- 实际 tz 范围: 0.12~0.31 (beizi 示例，手很近)

---

### 5. `SLAM/hawor_slam_w_scale_{start}_{end}.npz`

DROID-SLAM 输出的相机轨迹和深度信息。

```python
import numpy as np
slam = dict(np.load('SLAM/hawor_slam_w_scale_0_184.npz', allow_pickle=True))

# slam 包含:
# traj:       (184, 7)  - 每帧的 [tx, ty, tz, qx, qy, qz, qw]
#                         这是 camera-to-world (c2w) 位姿
#                         tx,ty,tz = 相机在世界空间的位置
#                         qx,qy,qz,qw = 四元数表示的相机朝向
# scale:      scalar    - Metric3D 估计的米制尺度因子
#                         0.44 表示 SLAM 原始坐标乘以 0.44 得到米制坐标
# img_focal:  scalar    - SLAM 使用的焦距 (600.0)
# img_center: (2,)      - 图像中心 [640.0, 360.0]
# tstamp:     (15,)     - SLAM 关键帧的时间戳 (帧号)
# disps:      (15, 328, 584) - 关键帧的逆深度图
#                            328x584 是 SLAM 内部分辨率
#                            值越大 = 距离越近
```

**与视频帧对应**: `traj[i]` 对应帧 `i`（即 `extracted_images/{i:04d}.jpg`）

**c2w 四元数转旋转矩阵**:

```python
import torch
from hawor.utils.rotation import quaternion_to_matrix

# traj 中的四元数顺序是 [qx, qy, qz, qw]
# quaternion_to_matrix 需要 [qw, qx, qy, qz]
R_c2w = quaternion_to_matrix(
    torch.from_numpy(traj[:, [3,0,1,2]])  # [qx,qy,qz,qw] → [qw,qx,qy,qz]
)
```

**disps (逆深度图) 可视化**:

```python
import cv2
# disps[0] 对应 tstamp[0] 帧的逆深度
disp = slam['disps'][0]  # (328, 584)
depth = 1.0 / (disp + 1e-6)  # 转为深度
depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
cv2.imwrite('keyframe_depth.png', (depth_norm * 255).astype(np.uint8))
```

---

### 6. `world_space_res.pth`

世界空间的手部重建结果（经 `R_x = diag(1,-1,-1)` 变换后），用 `joblib` 保存。

```python
import joblib
ws = joblib.load('world_space_res.pth')
# ws 是一个 list, 包含 5 个 tensor:

# ws[0] pred_trans:     (2, 184, 3)  - 世界空间手腕平移 [左手, 右手]
#                                    已乘以 R_x 变换
# ws[1] pred_rot:       (2, 184, 3)  - 世界空间手腕旋转 (angle-axis)
# ws[2] pred_hand_pose: (2, 184, 45) - 15个关节的 angle-axis (15*3=45)
# ws[3] pred_betas:     (2, 184, 10) - MANO shape 参数
# ws[4] pred_valid:     (2, 184)     - 是否有效帧 (bool)
```

**与视频帧对应**: `pred_trans[hand, i]` 对应帧 `i`

**坐标系**: 渲染世界空间 (经 R_x 变换: y-up, z-backward)

**pred_valid 含义**: `pred_valid[1, i] = True` 表示帧 `i` 检测到右手

**注意**: `pred_valid` 不完全可靠。infiller 会为两只手都生成数据，
即使只检测到一只手。未检测到的手的数据是垃圾数据（trans 接近零）。
请以 `cam_space/` 目录中实际存在的 hand_id 为准。

```python
# 获取右手有效帧号
valid_frames = torch.where(ws[4][1])[0].cpu().numpy()
# beizi: valid_frames ≈ [6, 7, 8, ..., 183]

# 判断实际检测到的手 (更可靠)
import os
cam_dir = 'example/beizi/cam_space'
detected_hands = set(int(d) for d in os.listdir(cam_dir)
                     if os.path.isdir(os.path.join(cam_dir, d)))
# beizi: {1}  (只有右手)
```

---

### 7. `reconstruction/hawor_results_{start}_{end}.npz`

`demov2.py` 生成的完整重建结果，包含世界坐标系下的所有数据。

```python
import numpy as np
res = dict(np.load('reconstruction/hawor_results_0_184.npz', allow_pickle=True))

# res 包含:
# pred_trans:     (2, 184, 3)  - 世界空间手腕平移 (经 R_x 变换)
# pred_rot:       (2, 184, 3)  - 世界空间手腕旋转 (angle-axis)
# pred_hand_pose: (2, 184, 45) - 15个关节的 angle-axis
# pred_betas:     (2, 184, 10) - MANO shape 参数
# pred_valid:     (2, 184)     - 是否有效帧 (注意: 不完全可靠, 见上文)
# R_c2w:          (184, 3, 3)  - camera-to-world 旋转矩阵 (经 R_x 变换)
# t_c2w:          (184, 3)     - camera-to-world 平移向量 (经 R_x 变换)
# img_focal:      scalar       - 焦距
# start_idx:      scalar       - 起始帧号
# end_idx:        scalar       - 结束帧号
```

---

### 8. `vis_cam_{start}_{end}/` - 相机视角叠加渲染

由 `demov2.py` 生成，手部网格直接投影到原始视频帧上。

- 每帧一张 PNG 图片，文件名 `{帧号:06d}.png`
- 右手 = 绿色网格，左手 = 橙色网格
- 只渲染 `cam_space/` 中实际检测到的手（不会出现垃圾数据）

**用途**: 验证手部 2D 投影是否与视频中的手对齐

---

### 9. `vis_world_{start}_{end}/` - 世界视角 3D 场景渲染

由 `demov2.py` 生成，从侧视相机看整个 3D 场景。

- `world_view.mp4` - 3D 场景视频
- 每5帧一张 PNG 图片

**场景内容**:
- 棋盘格地面 (灰色)
- 相机标记 (金字塔形，绿色/蓝色面表示相机朝向)
- 右手网格 (紫色)
- 左手网格 (橙色)

**用途**: 查看整体 3D 场景布局、相机运动轨迹、手在世界空间中的位置

---

### 10. `vis_verify/` - 可视化验证输出

由 `demov2.py` 或 `tools/vis_verify.py` 生成，用于验证数据与视频的对齐关系。

每帧验证图包含 5 个区域：

```
┌──────────┬──────────┬──────────┬────────┬──────────────┐
│          │          │          │        │              │
│  原始图像 │ 手部叠加  │ 深度图   │ 深度   │  相机轨迹    │
│          │ (绿色网格)│ (热力图)  │ 信息   │  (XZ俯视)   │
│          │          │          │ 面板   │  绿点=当前帧  │
│          │          │          │        │              │
└──────────┴──────────┴──────────┴────────┴──────────────┘
```

**深度信息面板内容**:
- 当前帧号
- 手腕深度 (tz): 手腕到相机的距离 (米)
- 五个指尖深度: Thumb/Index/Middle/Ring/Pinky 的 z 值
- 深度范围: min/max/span (反映手的厚度)
- 焦距和尺度因子

---

## 核心脚本说明

### 1. `demov2.py` - 主重建脚本

完整的 HaWoR 手部重建流程：检测 → 跟踪 → 运动估计 → SLAM → Infiller → 渲染。

**使用纯 OpenCV 渲染，不需要 X11 显示服务器，可以在任何服务器上运行。**

```bash
# 基本用法 (自动生成三种渲染输出)
conda run -n hawor python demov2.py --video_path example/beizi

# 指定焦距
conda run -n hawor python demov2.py --video_path example/beizi --img_focal 1280

# 对新视频运行
conda run -n hawor python demov2.py --video_path example/7.mp4
```

**自动生成的输出**:

| 输出目录 | 内容 | 用途 |
|----------|------|------|
| `vis_cam_{start}_{end}/` | 每帧 PNG (手部叠加在原始帧上) | 验证 2D 对齐 |
| `vis_world_{start}_{end}/` | PNG + MP4 (3D 场景: 手+地面+相机标记) | 查看整体场景 |
| `vis_verify/` | PNG + MP4 (叠加+深度+轨迹+信息面板) | 综合验证 |
| `reconstruction/` | NPZ (完整重建数据) | 后续使用 |

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video_path` | `example/video_0.mp4` | 视频文件路径或示例文件夹路径 |
| `--img_focal` | `None` (自动读取 est_focal.txt 或默认 600) | 相机焦距 (像素) |
| `--checkpoint` | `./weights/hawor/checkpoints/hawor.ckpt` | HaWoR 模型权重 |
| `--infiller_weight` | `./weights/hawor/checkpoints/infiller.pt` | Infiller 权重 |

---

### 2. `tools/vis_verify.py` - ★ 可视化验证脚本

**最常用的验证工具**。直接用 OpenCV 投影 MANO 网格到原始图像上，
同时显示深度信息和相机轨迹，无需 aitviewer。

```bash
# 生成验证图片 + 视频 (每 5 帧取一帧)
conda run -n hawor python tools/vis_verify.py --seq example/beizi

# 只生成图片 (跳过视频，更快)
conda run -n hawor python tools/vis_verify.py --seq example/beizi --skip_video

# 指定帧号
conda run -n hawor python tools/vis_verify.py --seq example/beizi --frames 10 30 50 70

# 调整取样间隔
conda run -n hawor python tools/vis_verify.py --seq example/beizi --step 10
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seq` | `example/beizi` | 示例文件夹路径 |
| `--frames` | `None` (自动按 step 取帧) | 指定要可视化的帧号 |
| `--skip_video` | `False` | 跳过视频生成，只生成图片 |
| `--step` | `5` | 图片模式下每隔几帧取一帧 |

**输出位置**: `{seq}/vis_verify/hand_{id}/`

---

### 3. `align_hawor_to_ras.py` - 对齐到 ReplicateAnyScene

将 HaWoR 世界坐标系的手部重建结果对齐到 ReplicateAnyScene 场景坐标系。

```bash
conda run -n ReplicateAnyScene python align_hawor_to_ras.py \
    --ras_output ./outputs/your_scene \
    --hawor_reconstruction ./HaWoR/example/beizi/reconstruction/hawor_results_0_184.npz \
    --hawor_slam ./HaWoR/example/beizi/SLAM/hawor_slam_w_scale_0_184.npz \
    --output_dir ./aligned_output
```

**对齐原理 (分解式)：**

```
总变换: p_ras = s * R_total @ p_hawor + t

分解为:
1. R_axis = [[1,0,0],[0,0,1],[0,-1,0]]  (轴约定: OpenCV y-down/z-forward → z-up)
2. R_residual ≈ I 或用 Umeyama 估计     (残差旋转)
3. t = RAS第一帧相机位置                  (原点对齐)
4. s = 从深度图或相机轨迹估计             (尺度对齐)
```

**输出文件：**

| 文件 | 说明 |
|------|------|
| `alignment_transform.npz` | 对齐变换参数 (scale, rotation, translation) |
| `aligned_hawor_hands.npz` | 对齐后的手部数据 (可直接用于 RAS 场景) |

---

## 坐标系说明

HaWoR 涉及多个坐标系的转换：

```
MANO 相机空间 (y-down, z-forward)
    │
    │ cam2world_convert (使用 SLAM c2w)
    ▼
SLAM 世界空间 (y-down, z-forward)
    │
    │ R_x = diag(1, -1, -1)
    ▼
渲染世界空间 (y-up, z-backward)  ← demov2.py / world_space_res.pth 的坐标系
    │
    │ R_axis = [[1,0,0],[0,0,1],[0,-1,0]]
    ▼
RAS 场景空间 (z-up)  ← align_hawor_to_ras.py 输出的坐标系
```

**重要**: `world_space_res.pth` 中的 `pred_trans` 已经过 `R_x = diag(1,-1,-1)` 变换。
如果需要原始 SLAM 坐标系的数据，请从 `SLAM/hawor_slam_*.npz` 和 `cam_space/` 重新计算。

---

## 焦距说明

焦距 (`img_focal`) 影响三个关键环节：

1. **MANO 3D 重建**: `tz = 2 * focal / bbox_size`，焦距直接影响深度估计
2. **SLAM 轨迹**: DROID-SLAM 使用焦距计算相机内参
3. **渲染投影**: 投影公式 `u = focal * x / z + cx`

**当前默认值**: 600 (硬编码后备值)

**推荐值**: `max(image_height, image_width)`
- 1280×720 图像 → focal = 1280
- 640×480 图像 → focal = 640

**修改焦距需要重新运行整个 pipeline**:

```bash
# 1. 更新焦距
echo "1280" > example/beizi/est_focal.txt

# 2. 清除旧数据
rm -rf example/beizi/SLAM/
rm example/beizi/tracks_0_184/frame_chunks_all.npy
rm -rf example/beizi/cam_space/

# 3. 重新运行
conda run -n hawor python demov2.py --video_path example/beizi --img_focal 1280
```

---

## 常用操作模板

### 模板 1: 从视频开始完整重建

```bash
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR

conda run -n hawor python demov2.py \
    --video_path example/your_video.mp4 \
    --img_focal 1280
```

运行后自动生成：
- `vis_cam_*/` — 相机视角叠加 (验证 2D 对齐)
- `vis_world_*/` — 3D 场景渲染 (查看整体布局)
- `vis_verify/` — 综合验证 (深度+轨迹+信息)
- `reconstruction/` — 完整重建数据

### 模板 2: 验证重建结果是否与视频对齐

```bash
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR

# 快速验证 (只生成图片)
conda run -n hawor python tools/vis_verify.py --seq example/beizi --skip_video

# 完整验证 (生成图片 + 视频)
conda run -n hawor python tools/vis_verify.py --seq example/beizi

# 检查特定帧
conda run -n hawor python tools/vis_verify.py --seq example/beizi --frames 10 30 50 70 --skip_video
```

验证图说明：
- **原始图像**: 视频原始帧
- **手部叠加 (绿色)**: MANO 网格投影到图像上，如果绿色手与图像中的手重合 → 对齐正确
- **深度图 (热力图)**: 蓝色=远，红色=近，如果深度分布合理 → 深度预测正确
- **深度信息面板**: 查看 tz 值是否合理 (手通常在 0.2~2.0 米)
- **相机轨迹 (绿点=当前帧)**: 查看相机运动是否合理

### 模板 3: 查看已有重建数据

```bash
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR

conda run -n hawor python -c "
import numpy as np, torch, json, joblib, sys, os
sys.path.insert(0, '.')

# 加载相机空间数据
cam_dir = 'example/beizi/cam_space'
for hid in os.listdir(cam_dir):
    for fname in os.listdir(os.path.join(cam_dir, hid)):
        if fname.endswith('.json'):
            with open(os.path.join(cam_dir, hid, fname)) as f:
                cam = json.load(f)
            trans = np.array(cam['init_trans'])
            hand = '右手' if hid == '1' else '左手'
            print(f'{hand}: 帧 {fname.replace(\".json\",\"\")}, 共 {trans.shape[1]} 帧')
            print(f'  深度范围: {trans[0,:,2].min():.3f} ~ {trans[0,:,2].max():.3f} 米')

# 加载 SLAM 数据
slam = dict(np.load('example/beizi/SLAM/hawor_slam_w_scale_0_184.npz', allow_pickle=True))
print(f'SLAM 尺度因子: {slam[\"scale\"]:.4f}')
print(f'SLAM 焦距: {slam[\"img_focal\"]}')
"
```

### 模板 4: 将手部对齐到 RAS 场景

```bash
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src

conda run -n ReplicateAnyScene python align_hawor_to_ras.py \
    --ras_output ./outputs/your_scene \
    --hawor_reconstruction ./HaWoR/example/beizi/reconstruction/hawor_results_0_184.npz \
    --hawor_slam ./HaWoR/example/beizi/SLAM/hawor_slam_w_scale_0_184.npz \
    --output_dir ./aligned_output \
    --combine_glb
```

### 模板 5: 用对齐后的手部数据生成 MANO 网格

```python
import numpy as np
import torch
import sys
sys.path.insert(0, '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR')
from hawor.utils.process import run_mano, run_mano_left, get_mano_faces

# 加载对齐后的数据
aligned = dict(np.load('./aligned_output/aligned_hawor_hands.npz', allow_pickle=True))

# 生成右手网格 (frame 0)
pred_trans = torch.from_numpy(aligned['pred_trans'])
pred_rot = torch.from_numpy(aligned['pred_rot'])
pred_hand_pose = torch.from_numpy(aligned['pred_hand_pose'])
pred_betas = torch.from_numpy(aligned['pred_betas'])

# 右手 (index 1), 第 0 帧
result = run_mano(pred_trans[1:2, 0:1], pred_rot[1:2, 0:1],
                  pred_hand_pose[1:2, 0:1], betas=pred_betas[1:2, 0:1])
vertices = result['vertices'][0, 0].cpu().numpy()  # (778, 3)
faces = get_mano_faces()

# 保存为 OBJ
import trimesh
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh.export('./aligned_output/right_hand_frame0.obj')
```

---

## 渲染输出对比

| 输出 | 视角 | 背景 | 内容 | 用途 |
|------|------|------|------|------|
| `vis_cam_*/` | 原始相机 | 原始视频帧 | 手部网格叠加 | 验证 2D 对齐 |
| `vis_world_*/` | 固定侧视 | 棋盘格地面 | 手+相机标记+地面 | 查看整体 3D 场景 |
| `vis_verify/` | 原始相机 | 原始视频帧 | 叠加+深度+轨迹+信息 | 综合验证 |

**推荐**: 先看 `vis_cam_*/` 确认 2D 对齐，再看 `vis_verify/` 检查深度，最后看 `vis_world_*/` 理解 3D 场景布局。
