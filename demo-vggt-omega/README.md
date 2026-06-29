# HaWoR + VGGT-Omega Reconstruction Pipeline

用 VGGT-Omega 替换 DROID-SLAM 的相机估计，坐标系与 ReplicateAnyScene mainv2.py 一致。

## 与 demov2.py 的核心区别

| 项目 | demov2.py | demo_vggt_omega.py |
|------|-----------|---------------------|
| 相机来源 | DROID-SLAM | VGGT-Omega |
| 尺度因子 | SLAM + Metric3D 估计 | 1.0 (VGGT-Omega metric depth) |
| 焦距来源 | SLAM 估计 / 手动指定 | VGGT-Omega FoV 推算 |
| Room Alignment | 无 | 可选 (需 ReplicateAnyScene) |
| 帧数限制 | 无 (SLAM 处理全部帧) | max_frames 限制 + SLERP 插值 |
| 输出目录 | `output/<video_name>/` | `output/<video_name>_vggt-omega/` |

## Pipeline 流程

```
1. detect_track_video          → 手部检测 & 追踪 (与 demov2 相同)
2. VGGT-Omega camera           → 相机位姿估计 (替换 SLAM)
   ├─ 加载 VGGT-Omega 模型
   ├─ 采样 max_frames 帧 (均匀采样)
   ├─ 推理 → extrinsics (w2c) + intrinsics + depth
   ├─ [可选] Room Alignment (SAM3 + floor/wall)
   ├─ w2c → c2w 转换
   └─ SLERP 插值到全部帧
3. hawor_motion_estimation     → 手部运动估计 (与 demov2 相同)
4. vggt_omega_infiller         → cam2world + 缺失帧填充 (修改版)
5. MANO forward + R_x 变换     → 手部顶点 (与 demov2 相同)
6. 可视化 + 保存结果
```

## 坐标系说明

- VGGT-Omega 输出: OpenCV convention (Y-down, Z-forward), w2c 格式
- 转换为 c2w 后传入 cam2world_convert (与 SLAM 的 c2w 用法一致)
- 可视化时应用 R_x = diag(1,-1,-1) 转换为 Y-up (与 demov2 一致)
- Room Alignment (可选): 通过 SAM3 检测 floor/wall, 对齐到 Z-up 坐标系 (floor at z=0)

## 使用方法

### 基本用法

```bash
cd /path/to/HaWoR
conda activate hawor
python demo-vggt-omega/demo_vggt_omega.py \
    --video_path /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/assets/example/hoi4d.mp4
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video_path` | `example/video_0.mp4` | 输入视频路径 |
| `--img_focal` | None | 手动指定焦距 (覆盖 VGGT-Omega 估计值) |
| `--checkpoint` | `./weights/hawor/checkpoints/hawor.ckpt` | HAWOR 模型权重 |
| `--infiller_weight` | `./weights/hawor/checkpoints/infiller.pt` | Infiller 模型权重 |
| `--max_frames` | 20 | VGGT-Omega 最大处理帧数 (47GB GPU 建议 20) |
| `--enable_room_alignment` | False | 启用 Room Alignment |

### Room Alignment (可选)

```bash
python demo-vggt-omega/demo_vggt_omega.py \
    --video_path /path/to/video.mp4 \
    --enable_room_alignment
```

需要 ReplicateAnyScene 在 `../ReplicateAnyScene/` 目录下可用。

### 手动指定焦距

```bash
python demo-vggt-omega/demo_vggt_omega.py \
    --video_path /path/to/video.mp4 \
    --img_focal 600.0
```

## 输出结构

```
output/<video_name>_vggt-omega/
├── vggt_omega_cam/                    # VGGT-Omega 相机数据
│   ├── vggt_omega_cam.npz             # R_c2w, t_c2w, img_focal, img_center
│   ├── intrinsic.txt                  # 3x3 内参矩阵
│   └── extrinsic_0.txt ...           # 每帧 4x4 外参矩阵
├── cam_space/                         # 相机空间手部参数
├── tracks_0_N/                        # 检测 & 追踪结果
├── world_space_res.pth                # 世界空间手部参数
├── est_focal.txt                      # 焦距估计
├── vis_cam_0_N/                       # 相机视角叠加 (手部 mesh)
│   ├── 000000.png
│   └── ...
├── vis_world_0_N/                     # 3D 世界视角 (需 demov2 可用)
├── vis_verify/                        # 验证图像 (需 demov2 可用)
└── reconstruction/
    └── hawor_results_0_N.npz          # 合并结果
        ├── pred_trans (2, N, 3)       # 手部平移
        ├── pred_rot (2, N, 3)         # 手部旋转 (axis-angle)
        ├── pred_hand_pose (2, N, 45)  # 手部姿态
        ├── pred_betas (2, N, 10)      # MANO betas
        ├── pred_valid (2, N)          # 有效帧标记
        ├── R_c2w (N, 3, 3)           # 相机到世界旋转
        ├── t_c2w (N, 3)              # 相机到世界平移
        ├── img_focal                  # 焦距
        ├── img_center (2,)            # 主点
        ├── slam_scale                 # 尺度因子 (=1.0)
        └── camera_source              # "vggt_omega"
```

## 依赖

### 必需

- HaWoR 环境 (hawor conda env, PyTorch 1.13+)
- VGGT-Omega 模型权重: `/mnt/data/lza/models/vggt_omega/vggt_omega_1b_512.pt`
- VGGT-Omega 代码: `../vggt-omega/`

### PyTorch 兼容性

VGGT-Omega 官方要求 PyTorch >= 2.3，但本脚本已内置兼容层:

- **`F.scaled_dot_product_attention`**: PyTorch 2.0+ 才有，本脚本在导入 VGGT-Omega 前自动 monkey-patch 一个等效实现
- **`torch.autocast`**: PyTorch 1.10+ 已支持，HaWoR 环境 (1.13) 可正常使用
- **无需升级 PyTorch**，不影响 HaWoR 原有代码运行

已验证环境:
- PyTorch 1.13.0+cu117: ✓ 导入、模型实例化、前向推理均正常
- PyTorch 2.x: ✓ 原生 SDPA，无需 patch

### 可选

- ReplicateAnyScene (Room Alignment): `../ReplicateAnyScene/`
- SAM3 模型权重 (Room Alignment 需要)

## 关键技术细节

### 帧数插值

VGGT-Omega 有 max_frames 限制 (默认 160)。当视频帧数超过此限制时:
1. 均匀采样 max_frames 帧进行推理
2. 旋转: SLERP (球面线性插值) 插值四元数
3. 平移: 线性插值
4. 边界帧: 使用最近的 VGGT-Omega 帧的位姿

### 焦距转换

VGGT-Omega 在 512x512 分辨率下估计内参。转换为原始分辨率:
```
fov_w = 2 * arctan(W_vggt / (2 * fx_vggt))
fx_orig = (orig_W / 2) / tan(fov_w / 2)
```

### 尺度因子

VGGT-Omega 的深度预测使用 exp 激活，输出始终正值。与 SLAM 不同，不需要额外的尺度估计 (Metric3D)，slam_scale 固定为 1.0。

## 测试

```bash
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR
python demo-vggt-omega/demo_vggt_omega.py \
    --video_path /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/assets/example/hoi4d.mp4 \
    --max_frames 160
```

输出目录: `output/hoi4d_vggt-omega/`
