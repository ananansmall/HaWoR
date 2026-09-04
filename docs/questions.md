# HaWoR 问题与解答汇总

本文档记录项目开发过程中的问题与解答。

---

## 一、VGGT-Omega 模型相关

### Q1: VGGT-Omega 两个模型权重有什么用？分别在哪个文件夹？对于 predict 和 demo-vggt-omega 来说有什么区别？

**回答**:

模型权重位于 `/mnt/data/lza/models/vggt_omega/`，有两个文件：

| 模型文件 | 大小 | 分辨率 | Text Alignment | 用途 |
|---------|------|--------|---------------|------|
| `vggt_omega_1b_512.pt` | 4.3 GB | 512×512 | ❌ 无 | **标准模型**: 相机位姿 + 深度预测 |
| `vggt_omega_1b_256_text.pt` | 5.0 GB | 256×256 | ✅ 有 | **文本对齐变体**: 额外输出 text_alignment_embedding |

**架构差异**:
- 两者共享相同的 Aggregator 骨干 (patch_size=16, embed_dim=1024, 24层交替注意力)
- 两者都有 CameraHead (9D pose_enc) 和 DenseHead (depth + depth_conf)
- `vggt_omega_1b_256_text.pt` 额外包含 `TextAlignmentHead`, 输出 `predictions["text_alignment_embedding"]`

**对 predict 和 demo-vggt-omega 的影响**:
- `demo-vggt-omega/demo_vggt_omega.py` 当前使用 `vggt_omega_1b_512.pt` (CHECKPOINT_PATH 硬编码)
- `ReplicateAnyScene/src/vggt_omega_predict.py` 同样使用 `vggt_omega_1b_512.pt`
- **512 模型是推荐选择**: 更高分辨率 (512 vs 256) → 更精确的相机位姿和深度
- **256_text 模型用于**: 需要文本-3D对齐的场景 (如 ReplicateAnyScene 的文本驱动物体检索), demo-vggt-omega 不需要此功能

**加载方式差异**:
```python
# 512 标准模型 (当前使用)
model = VGGTOmega().eval()  # enable_alignment=False (默认)
state_dict = torch.load("vggt_omega_1b_512.pt", ...)

# 256 文本对齐模型
model = VGGTOmega(enable_alignment=True).eval()
state_dict = torch.load("vggt_omega_1b_256_text.pt", ...)
# 额外输出: predictions["text_alignment_embedding"]
```

**结论**: 对于 demo-vggt-omega 的手部重建任务, `vggt_omega_1b_512.pt` 是正确选择, 512 分辨率提供更精确的相机位姿。`vggt_omega_1b_256_text.pt` 的文本对齐功能在手部重建中不需要。

---

### Q2: VGGT-Omega 输出 (`output/7_vggt-omega`) 与原始 DROID-SLAM 输出 (`example/7`) 的核心数据差异有多大？

**回答**:

两套输出使用**相同的输入视频**，但相机估计方法不同：
- `example/7`：使用 **DROID-SLAM** 估计相机位姿
- `output/7_vggt-omega`：使用 **VGGT-Omega** 估计相机位姿

#### 1. 焦距 (Focal Length) — ⚠️ 差异很大

| 指标 | VGGT-Omega | DROID-SLAM (原始) |
|------|-----------|-------------------|
| 焦距 | **905.5** | **600.0** |
| 差异 | +305.5 (**+50.9%**) | — |

VGGT-Omega 估计的焦距比 DROID-SLAM 高出约 51%，这意味着 VGGT-Omega 认为相机视场角 (FoV) 更窄。焦距直接影响 3D→2D 投影，50% 的差异会导致手部 mesh 在图像上的投影大小显著不同。

#### 2. 相机旋转 (R_c2w) — ✅ 差异较小

| 指标 | 值 |
|------|---|
| 平均角度差 | **2.21°** |
| 最大角度差 | **4.25°** |
| 中位角度差 | **2.07°** |
| <5° 的帧 | **113/113 (100%)** |

旋转估计非常一致，所有帧的角度差都在 5° 以内，这在多视图几何中属于合理范围。

#### 3. 相机平移 (t_c2w) — ⚠️ 差异显著（尺度不同）

| 指标 | VGGT-Omega | DROID-SLAM |
|------|-----------|------------|
| 平移范数均值 | **0.0534** | **0.0156** |
| 平移范数范围 | [0.0003, 0.1122] | [0.0015, 0.0266] |
| **尺度比 (VGGT/Orig)** | **均值 3.12** | — |

VGGT-Omega 的相机平移量约为 DROID-SLAM 的 **3 倍**。这是因为 VGGT-Omega 直接输出 metric depth（绝对尺度），而 DROID-SLAM 输出的是相对尺度（需要后续 scale 对齐）。VGGT-Omega 的 `slam_scale=1.0`（无缩放），而 DROID-SLAM 通常需要 scale recovery。

#### 4. 手部姿态 (Hand Pose) — ✅ 差异较小

| 参数 | 绝对差均值 | 绝对差最大值 |
|------|----------|------------|
| pred_trans (平移) | 0.0138 | 0.0788 |
| pred_rot (旋转) | 0.0129 | 0.0956 |
| pred_hand_pose (关节) | 0.0002 | 0.0026 |
| pred_betas (形状) | 0.0004 | 0.0074 |

手部关节角 (hand_pose) 和形状参数 (betas) 几乎一致（差异 <0.01），说明 HaWoR 的手部检测和运动估计阶段不受相机估计方法影响。平移和旋转的差异主要来自 cam2world 转换时使用了不同的相机位姿。

#### 5. cam_space 数据 — ✅ 完全一致

cam_space 中的手部参数（相机坐标系下）完全相同，因为这是 HaWoR 检测阶段的输出，与相机估计方法无关。

#### 总结

| 数据类型 | 差异程度 | 原因 |
|---------|---------|------|
| 焦距 | ⚠️ **大** (50.9%) | VGGT-Omega 和 DROID-SLAM 对 FoV 的估计不同 |
| 相机旋转 | ✅ 小 (平均 2.2°) | 两种方法对相机朝向的估计基本一致 |
| 相机平移 | ⚠️ **显著** (3x 尺度) | VGGT-Omega 输出 metric depth，DROID-SLAM 输出相对尺度 |
| 手部关节/形状 | ✅ 很小 | HaWoR 检测阶段与相机估计无关 |
| cam_space | ✅ 完全一致 | 同上 |

**核心结论**：差异主要集中在**焦距**和**平移尺度**上，这是两个方法本质不同导致的。旋转和手部姿态本身差异不大。焦距 50% 的差异是最大的问题——它会直接影响手部 mesh 投影到图像上的大小是否准确。

---

### Q3: VGGT-Omega 焦距能否固定？Scale recovery 后与 ReplicateAnyScene 对比如何？

**回答**:

#### 1. VGGT-Omega 焦距能否固定？

**不能直接固定。** VGGT-Omega 的焦距由 `CameraHead` 神经网络完全估计，编码在 `pose_enc` 的第 7、8 维（FoV），没有配置接口可以注入已知焦距。

焦距计算链路：
```
图像特征 → CameraHead 神经网络 → pose_enc[..., 7:9] (FoV) → fx = (W/2) / tan(fov_w/2)
```

可行的替代方案（按难度递增）：

| 方案 | 做法 | 优缺点 |
|------|------|--------|
| **A. 后处理替换** | 解码后用已知焦距覆盖 intrinsic 中的 fx/fy | 最简单，但深度-焦距耦合：模型估计的深度 d 和焦距 f 是配合的，替换焦距后深度需按 `f_known/f_est` 缩放 |
| **B. 修改 pose_enc** | 推理后将 pose_enc[..., 7:8] 替换为已知 FoV 对应值再解码 | 同样有深度耦合问题 |
| **C. 训练时约束** | 修改 CameraHead 或损失函数，加入焦距约束 | 最彻底，但需重新训练 |

**关键问题**：VGGT-Omega 的深度估计和焦距是**耦合**的。模型可能在"错误的"焦距下估计了"配合的"深度，使得 3D 重建自洽。如果只替换焦距而不调整深度，反投影出的 3D 点会出错。

#### 2. 三方焦距对比（加入 ReplicateAnyScene）

视频分辨率 1920×1080，RAS 使用 512×512 采样：

| 方法 | 焦距 (原始分辨率) | 与 RAS 差异 |
|------|-----------------|------------|
| **VGGT-Omega** | **905.5** | **2.0%** ✅ |
| **ReplicateAnyScene** | **888.1** | — (基准) |
| **DROID-SLAM** | **600.0** | **32.4%** ❌ |

**VGGT-Omega 的焦距估计与 ReplicateAnyScene 非常接近（仅差 2%）**，而 DROID-SLAM 偏差 32.4%。这说明 VGGT-Omega 的焦距估计实际上更准确。

#### 3. 三方相机轨迹对比

**旋转 (R_c2w) — 与 RAS 对比：**

| 方法 | 平均角度差 | 最大角度差 |
|------|----------|----------|
| **VGGT-Omega vs RAS** | **0.59°** ✅ | **1.02°** |
| **DROID-SLAM vs RAS** | **2.50°** | **4.11°** |
| VGGT vs DROID | 2.17° | 3.83° |

VGGT-Omega 的旋转估计与 RAS 高度一致（平均仅 0.59°），远优于 DROID-SLAM。

**平移 (t_c2w) — Procrustes 对齐到 RAS：**

| 方法 | Scale 因子 | 残差均值 | 残差最大值 |
|------|-----------|---------|----------|
| **VGGT-Omega → RAS** | **0.9271** | **0.0065** ✅ | **0.0120** |
| **DROID-SLAM → RAS** | **2.8575** | **0.0136** | **0.0230** |

关键发现：
- VGGT-Omega 对齐到 RAS 的 scale 因子为 **0.93**（接近 1.0），说明 VGGT-Omega 的绝对尺度与 RAS 几乎一致，**不需要 scale recovery**
- DROID-SLAM 对齐到 RAS 的 scale 因子为 **2.86**，说明 DROID-SLAM 的尺度偏小，需要约 2.86x 的缩放才能与 RAS 对齐
- VGGT-Omega 对齐后的残差也更小（0.0065 vs 0.0136）

#### 4. 总结

| 对比项 | VGGT-Omega | DROID-SLAM |
|--------|-----------|------------|
| 焦距准确性 | ✅ 与 RAS 仅差 2% | ❌ 与 RAS 差 32% |
| 旋转准确性 | ✅ 与 RAS 平均 0.59° | 与 RAS 平均 2.5° |
| 尺度 | ✅ 接近 metric（scale≈0.93） | ❌ 需 scale recovery（scale≈2.86） |
| 对齐后残差 | ✅ 更小 (0.0065) | 较大 (0.0136) |

**结论**：
1. VGGT-Omega 焦距**不能固定**，但其估计的焦距（905.5）与 RAS（888.1）非常接近，实际上比 DROID-SLAM 更准确
2. VGGT-Omega **不需要 scale recovery**——它的尺度已经接近 metric（与 RAS 的 scale 因子为 0.93），而 DROID-SLAM 需要 ~2.86x 的 scale recovery
3. VGGT-Omega 的相机轨迹整体上比 DROID-SLAM 更接近 ReplicateAnyScene 的结果

---

## 二、手部参数与位置生成过程

### Q4: HaWoR 手部关键点参数、位置是怎么得到的？整个数据流是什么？

**回答**:

HaWoR 的手部重建分 4 个阶段：检测跟踪 → 相机系 MANO 估计 → cam→world 变换 + 补全 → MANO 正演输出。

#### 1. 手部检测与跟踪（`detect_track_video.py` + `lib/pipeline/tools.py`）

- **检测器**：YOLO（`./weights/external/detector.pt`，ultralytics），置信度阈值 `thresh=0.2`。
- **跟踪器**：YOLO 内置 `model.track(..., persist=True)`，本质是 **ByteTrack**。
- **左右手判定**：`results[0].boxes.cls` 输出 0=left、1=right。
- **输出**：
  - `model_boxes.npy`：实际是空占位（代码中 `boxes_ = []` 后未填充）。
  - `model_tracks.npy`：是 `dict {track_id: [subj...]}`，每个 `subj` 含 `frame, det, det_box(xyxy+conf), det_handedness(0/1)`。后续在 `hawor_motion_estimation` 中按 handedness 投票合并为左右两条轨（`final_tracks = {0: left, 1: right}`）。
- 精度依赖该 YOLO 检测器；阈值 0.2 较低以保证召回。

#### 2. 相机系 MANO 估计（`hawor_video.py:hawor_motion_estimation` → `lib/models/hawor.py:inference/forward`）

- 把轨按连续性切成 chunks（`parse_chunks`，min_len=16），每 16 帧一个 batch 喂入 **HAWOR 模型**（ViT backbone + Transformer mano_head）。
- 网络回归三组输出：
  - `pred_pose`（rot6d）→ `pred_rotmat` (T,16,3,3)：第 0 个是 root_orient，后 15 个是手指关节。
  - `pred_shape` (T,10)：MANO 形状参数 betas。
  - `pred_cam` (s, tx, ty)：弱透视相机参数。
- **`init_hand_pose` 45 维** = 15 个手指关节 × 3 轴角（MANO 标准自由度：5 指各 3 关节）。
- **`init_betas` 10 维** = MANO 标准形状空间维度。
- **训练损失**（`compute_loss`）：2D 关键点重投影 L1 + 3D 关键点 L1 + MANO 参数（global_orient/hand_pose/betas）L1，无显式正则项。
- **弱透视投影 `tz = 2·focal/(scale·200)`**（`get_trans`，hawor.py:500）：
  - `b = scale * 200`：`scale` 是检测框归一化大小，200 是训练时约定的"还原到像素"常数（bbox 短边归一化基准）。
  - `tz = 2*focal/b` 来自弱透视相机模型：物体像高 = `focal·H_real/Z`，令 `H_real = b/s`（其中 s 是网络预测的缩放），反解得 `Z = 2·focal·s/b`，代码中 `bs = b*s`，故 `tz = 2·focal/bs`。
  - tx/ty 同理做中心点偏移补偿：`tx_full = tx + 2(cx-img_cx)/bs`。

#### 3. cam→world 变换 + 补全（`custom_utils.py:cam2world_convert` + `hawor_video.py:hawor_infiller`）

- **cam→world 变换**（注意不是简单 `R@t+t`）：
  1. 先在相机系用 `run_mano` 正演得到 root joint 位置 `root_loc`（B,T,3）。
  2. `offset = init_trans - root_loc`（MANO 的 transl 与 root joint 的固定偏移，与旋转无关）。
  3. `init_trans_world = R_c2w @ root_loc + t_c2w + offset`（`einsum("tij,btj->bti", R_c2w_sla, root_loc) + t_c2w_sla`）。
  4. 旋转：`R_world = R_c2w @ R_cam`（再转轴角）。
- c2w 来自 SLAM（`hawor_slam_w_scale.npz`），`t_c2w` 已乘 `scale`（SLAM 尺度恢复），`R_c2w` 由四元数 (wxyz→xyzw) 转矩阵。
- **Infiller 补全**：`TransformerModel`（horizon=120, d_model=384, 8 层），输入 154 维 = 2 手 × (3 trans + 10 betas + 96 rot6d=16关节×6)。补全的是 **检测丢失帧**（`pred_valid=False`）。
  - 流程：转 canonical 系（对齐到窗口首帧）→ slerp/lerp 初始化缺失帧 → rot6d → Transformer 推理 → 转回 world 系。
  - 输出 `pred_trans(2,T,3)`、`pred_rot(2,T,3 轴角)`、`pred_hand_pose(2,T,45)`、`pred_betas(2,T,10)`。

#### 4. MANO 正演 + 输出（`demov2.py` + `hawor/utils/process.py:run_mano`）

- 输入 `pred_trans, pred_rot(aa), pred_hand_pose(15×3 aa), pred_betas(10)` → MANO 正演 → `joints(T,21,3)`、`vertices(T,778+,3)`、`faces`。
- **`hawor_results.npz` 字段**：`pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid, R_c2w, t_c2w, img_focal, start_idx, end_idx`。
- **`pred_valid`**：检测+跟踪成功帧初始为 1；infiller 处理后所有缺失帧也被置为 1（`pred_valid[:, start:end] = 1`）。
- **最终坐标系**：原始 **SLAM World (OpenCV 系：x-right, y-down, z-forward)**，因为 SLAM 输出的是 OpenCV 相机系，cam2world 直接用其 c2w 转过去仍是 OpenCV 系。
- **为什么不是 OpenGL**：SLAM (DROID) 默认输出 OpenCV 系。`demov2.py` 在渲染阶段才用 `R_x = diag(1,-1,-1)` 把 y/z 翻转成 OpenGL (y-up, z-back) 用于可视化；但保存到 npz 的 `pred_trans/pred_rot/R_c2w/t_c2w` 仍是原始 OpenCV 系。

#### 数据流总结

```
视频 → ffmpeg 抽帧(30fps)
   → YOLO 检测+ByteTrack 跟踪 → model_tracks.npy (左右手两轨)
   → HAWOR 模型 (ViT+Transformer) → 相机系 MANO 参数 (cam_space/*.json)
       ├── pred_rotmat (16,3,3) → init_root_orient(3,3) + init_hand_pose(15,3,3)
       ├── pred_shape (10)
       └── pred_cam (s,tx,ty) → get_trans → init_trans (3) [弱透视]
   → cam2world_convert (用 SLAM 的 R_c2w, t_c2w) → 世界系参数
   → Infiller Transformer 补全遮挡/丢失帧
   → world_space_res.pth: pred_trans(2,T,3), pred_rot(2,T,3), pred_hand_pose(2,T,45), pred_betas(2,T,10), pred_valid(2,T)
   → MANO 正演 → vertices/joints
   → hawor_results.npz (OpenCV 世界系)
```

**关键点**：每个手部参数的来源——
- `pred_rot/hand_pose`：HAWOR 网络直接回归（rot6d→rotmat→aa），训练监督是 2D/3D 关键点 + MANO 参数 L1。
- `pred_trans`：网络回归 `pred_cam(s,tx,ty)`，经弱透视公式 `tz=2f/(s·b)` 还原到相机系 3D 平移，再用 SLAM c2w 转世界系。
- `pred_betas`：网络直接回归 10 维形状。
- `pred_valid`：检测命中=1，infiller 补全帧也被置 1。

---

## 三、mano_ras_reconstruction（RAS depth 相关重建）

### Q5: mano_ras_reconstruction 脚本在哪、生成的 npz 与 hawor_results 有什么区别？为什么右手也 valid？demo 和 demov2 有什么区别？

**回答**:

#### 1. 文件位置

- 脚本：`HaWoR/mano_ras_reconstruction.py`（已从 test/ 移到主目录，因 `test/_DATA` 不存在，MANO 模型路径依赖 HERE 定位）。
- 输出：`HaWoR/output/<video>/reconstruction/mano_ras_reconstruction_<start>_<end>.npz`。

#### 2. 两个 npz 的字段对比

| hawor_results（原，参数+SLAM 相机） | mano_ras_reconstruction（新，RAS depth 反投影结果） |
|---|---|
| pred_trans/pred_rot/pred_hand_pose/pred_betas (MANO 参数) | 无参数，只有前向结果 |
| R_c2w/t_c2w（HaWoR SLAM 世界系） | fx/fy/cx/cy、depth_scale（RAS depth 内参） |
| img_focal、start_idx、end_idx | start_idx/end_idx、hawor_frames/ras_frames/frame_ratio |
| pred_valid (2,T) | ri_to_hw (T,2)：内容反查帧映射 |
| — | verts_ras_{left,right} (T,778,3)：RAS depth 相机系顶点（查不到处 NaN） |
| — | joints_ras_{left,right} (T,16,3)：J_regressor 从 NaN-safe 顶点回归 |
| — | verts_cam_{left,right}：同批顶点的 HaWoR 相机系参考值 |
| — | valid_{left,right} (T,) |

配套性分析：新 npz 自洽（几何结果+内参+帧映射即可解释全部数据）。若下游需要可再加：(a) 用 RAS `extrinsics/*.txt` 每帧外参转到 RAS 世界系；(b) 21 关节格式（现 16 MANO 原生关节，缺 5 指尖，指尖可由已知顶点索引直接取）。按"缺失就空着"原则暂不加。

补充（demov2 全部产物清单，以 output/7/ 为例）：`hawor_results npz` 是 demov2 **唯一的重建数据文件**；其余全是可视化/中间产物——vis_cam(74M)/vis_world(17M)/vis_verify(60M)/combined_render(14M) 视频、cam_space（真检出的相机系参数 json）、SLAM、tracks、extracted_images、est_focal/mask_focal、world_space_res.pth、run_demov2.log。两者关系是**互补而非替代**：hawor_results=MANO 参数源+HaWoR SLAM 世界系（可重新前向/转到 RGB 世界系）；mano_ras_reconstruction=RAS depth 相机系的几何结果（778 顶点+16 关节）。注意关节格式差异：demov2 生态用 21 关节（run_mano 输出 16+5 指尖），新 npz 只有 16。

#### 3. 为什么右手也是 valid（明明只有左手）

两层原因：

1. **hawor_results 里右手的 pred_valid 本身就是全 True**——但这是 infiller 的副作用。视频 7 实际只检出左手（`example/7/cam_space/` 只有手 0 目录），右手全程缺失 → 整段作为缺失 gap 喂给 infiller → `hawor_video.py:378` `pred_valid[:, filling_net_start:filling_net_end] = 1` 把填充区间**两只手都**标记为 valid。所以右手的 MANO 参数全是 Transformer 网络脑补的，不是真实检测。
2. **新 npz 的 valid_right 只表示"投影在图内且查到 depth>0"**。背景像素也有深度值，所以几乎每帧都 True，它不代表"RAS 拍到了右手"。

#### 4. demo 和 demov2 的区别

- **demo.py**（115 行）：核心 pipeline——detect_track_video → hawor_motion_estimation → hawor_slam → hawor_infiller → MANO 正演 → aitviewer 可视化（world/cam 两模式）。不保存 npz。
- **demov2.py**（~1100 行）：核心 pipeline 与 demo 完全相同（demov2.py:726 注释 "identical to demo.py"），额外增加：
  - 日志落盘 run_demov2.log（demov2.py:695-722）
  - 强制 focal 重搜索，失败回退 600（demov2.py:747-771）
  - 保存 hawor_results npz（demov2.py:928 附近）
  - 多种可视化产物：vis_cam / vis_world / vis_verify / 相机轨迹图 / RAS depth_with_mano 叠加（demov2.py:1089-1095）
- 两者都接受任意 mp4（--video_path），detect_track_video 自动抽帧+YOLO 检测+ByteTrack 跟踪；区别只在 demov2 输出更多中间产物并强制重估 focal。

---

### Q6: infiller 模块到底是什么处理逻辑？为什么整段没检测到也能"编"出一只手？

**回答**:

#### 本质

infiller 是一个**双手操作先验 Transformer**（[infiller/lib/model/network.py](TransformerModel)：horizon=120 帧、d_model=384、8 层、输入 154 维 = 2 手 × (trans 3 + betas 10 + rot6d 96)）。它在双手操作序列上训练，学的是"两只手一起干活时的协同运动规律"。设计目的是补**短时遮挡**，但代码里没有任何"这只手从未被检测到"的防护，所以单手视频会被它凭空生成一只完整的幻觉手。

#### 处理流程（[hawor_video.py:238-381](../scripts/scripts_test_video/hawor_video.py)）

1. **真实检测入数组**（278-302）：每只手的跟踪 chunk 从 `cam_space/<idx>/*.json` 读出，cam→world 后写入 `pred_*`，`pred_valid[idx, chunk] = 1`。
2. **找缺失 gap**（308-312）：对每只手（**右手先处理**，`for idx in [1, 0]`）取 `missing = ~pred_valid[idx]`，切 gap。
3. **找锚帧 + 定窗口**（316-323）：每个 gap 从首帧向前找"**双手都 valid**"的最近帧（`pred_valid[:, f].sum() != 2` 则继续前移）；从锚帧起取 ≤120 帧窗口。
4. **预处理**（[filling_utils.py:146-251](filling_preprocess)）：窗口内双手参数转到 canonical 系（以各手窗口首帧的根关节位姿为原点）；缺失帧用 lerp/slerp 插值初始化；rot6d 编码；拼成 (T, 154)。
5. **Transformer 推理**（360）：valid 帧通过 data_mask 作为上下文，模型输出整个窗口**双手**的补全参数。
6. **只替换缺失位置**（369-372）：`filling_seq['trans'][~seq_valid] = output[~seq_valid]` 等，只覆盖无效帧。
7. **置 valid 的"副作用"**（374-378）：写回后 `pred_valid[:, filling_net_start:filling_net_end] = 1`——**区间内两只手全部标记有效**，下游无法区分真实检测和模型生成。

#### 为什么视频 7 整段被"编"

- 右手全程零检测 → `missing` 全 True → 整个视频是一个 gap；
- 锚帧搜索在 frame 0 就退出（前面没有帧）→ 窗口 `[0, 120)` 覆盖全部 113 帧，一次推理搞定；
- 右手输入全零（canonical 系下插值无有效点可插，保持零），左手真实运动作为条件 → Transformer 按"左手在做 X 时右手通常做 Y"的先验**生成**一只假右手；
- 结果写进 `hawor_results` 且 `pred_valid[1]` 全 True——这就是新 npz 里右手数据的来源。

#### 结论

`pred_valid=True ≠ 真实检测`，它可能是 infiller 补全后统一置位的。要区分只能回查 `cam_space/<hand>/` 目录（有 json 才是真检测过）。

---

### Q7: mano_ras_reconstruction 为什么是相机坐标系？为什么 16 关节缺 5 指尖？帧数跟原视频还是 depth？有没有平滑？

**回答**:

1. **坐标系**：depth 反投影（X=(u-cx)·z/fx）天然在 depth 相机系。已按 [ReplicateAnyScene/mainv2.py](../ReplicateAnyScene/mainv2.py) 的约定（mainv2.py:438 `"extrinsics_convention": "w2c": p_cam = R @ p_world + t`）读 RAS `extrinsics/<ri>.txt`（4×4 w2c）反变换：`p_world = R_w2c^T @ (p_cam - t_w2c)`。新增输出字段 `verts_ras_world_{side}` (T,778,3)、`joints_ras_world_{side}` (T,21,3)。

2. **16 vs 21 关节**：MANO 原生模型只有 **16 个关节**（J_regressor 是 (16,778)：wrist+15 手指关节），5 个指尖不是独立关节。HaWoR 的做法（[lib/models/mano_wrapper.py:20-38](../lib/models/mano_wrapper.py)）：从网格上取 5 个固定顶点当指尖（本环境 smplx vertex_ids['mano'] = thumb744/index320/middle443/ring554/pinky671），cat 到 16 关节后按 OpenPose 顺序重排成 21。新 npz 已照抄此逻辑，`joints_ras_*`/`joints_ras_world_*` 均为 21 关节。

3. **帧数与平滑**：T = hawor_results 的帧数 = **原视频抽帧数**（30fps；视频7 T=113，121 T=600），不是 depth 帧数。逐帧完全独立处理：MANO 前向→投影→查 depth→反投影，**无任何平滑/插值/滤波**（demov2 没有的都没加）。唯一的对应关系是视频帧 i → RAS depth 帧 ri：内容反查优先（缩略图 MSE），fallback 线性比例；映射关系存在 `ri_to_hw` 里。注意 demov2 参数层自带的 infiller 补全仍在（这是 hawor_results 本身就有的，非本脚本添加）。

补充（NaN 帧误判 bug 说明，与"7 有没有右手"无关）：`verts_to_joints` 是矩阵乘法，某帧顶点全 NaN 时 verts_safe 把 NaN 替换成 0 → joints 回归出全 0 向量（不是 NaN）→ `~np.isnan()` 判不出 → 无效帧被误标 valid。实测 7 号右手第 112 帧即此情况（修复前误报 113/113，修复后正确 112/113）。该坑对左手同样存在：任何反投影全失败的帧都会被旧代码显示成坐标原点 [0,0,0] 且标记有效。右手"存在"本身是 infiller 脑补问题（见 Q6），本次未改动。

---

### Q8: 更新后两个 npz 的区别？demov2 还生成了什么有用的东西（track 等）？

**回答**:

#### 两个 npz（更新后）

| | hawor_results（demov2） | mano_ras_reconstruction（新） |
|---|---|---|
| 本质 | MANO **参数源** + SLAM 相机 | 前向后的**几何结果** |
| 坐标系 | HaWoR SLAM 世界系（RGB 视频） | RAS depth 相机系 + RAS 世界系（verts/joints 各一份） |
| 几何 | 无（需自己前向） | verts (T,778,3) + joints (T,**21**,3 OpenPose 序) |
| RAS 信息 | 完全没有 | 内参 fx/fy/cx/cy、depth_scale、ri_to_hw 帧映射 |
| valid 语义 | infiller 置位（真检出+脑补都算） | depth 反投影逐帧真实成功与否 |

关节格式已对齐（都是 21/OpenPose）。参数要重投影时用 `R_c2w/t_c2w + img_focal`；几何直接用新 npz。

#### demov2 其他产物里真正有用的（以 output/7/tracks_0_113/ 为例）

- `model_tracks.npy`：YOLO+ByteTrack 检测记录（每帧 bbox、置信度、左右手判定）→ 可按置信度过滤低质量帧
- `model_masks.npy` (T,H,W)：RGB 手掩码 → 可验证投影是否落在 RGB 手区域
- `model_verts.npy` / `model_joints.npy`：运动估计阶段的相机系 MANO（仅真检出的手）
- `frame_chunks_all.npy`：每只手的连续检测段结构
- `cam_space/<hand>/<chunk>.json`：**唯一能区分"真检出"vs"infiller 脑补"的记录**（有 json 才是真检测过；7 号只有 cam_space/0/0_112.json）
- `SLAM/hawor_slam_w_scale_*.npz`：traj(113,7)、scale、img_focal、img_center、disps——hawor_results 已含 c2w，但 scale 只在这里
- `est_focal.txt` / `mask_focal.txt`：焦距估计（分支 C 投影用的 600 即来自此）

其余 vis_cam / vis_world / vis_verify / combined_render 均为纯可视化，无数据价值。

补充（tracks_0_113 目录的用途）：它是 `hawor_motion_estimation` 阶段（检测+跟踪+第一遍 MANO 估计）的中间缓存目录（[hawor_video.py:68-236](../scripts/scripts_test_video/hawor_video.py)）。作用有二：(1) **断点缓存**——`frame_chunks_all.npy / model_masks.npy / model_verts.npy / model_joints.npy` 四件齐全时直接加载返回，跳过昂贵的 HAWOR 模型推理（hawor_video.py:69-73）；(2) **下游数据源**——`model_tracks.npy` 被 hawor_video.py:53 重读用于切 chunk；`frame_chunks_all.npy` 传入 infiller 确定"真检出帧"；`model_masks.npy` 被 demov2.py:750 用于焦距搜索和 combined_render。想知道某帧真检出与否及置信度查 `model_tracks.npy`；区分真检出 vs 脑补结合 `cam_space/<hand>/` 是否有 json。

---

### Q9: demov2 比 demo 多生成了哪些？哪些真有用哪些只是参考？

**回答**: 三类：

- **A 纯可视化（仅参考，可删，~165M/视频）**：`vis_cam_*`（RGB叠网格，demov2.py:846-900）、`vis_verify`（诊断拼图，仅真检出手，908-913）、`vis_world_*`（世界系渲染，903-906）、`combined_render`（合成视频，936+）、RAS `depth_with_mano`（1089-1095）。确认用户判断正确。
- **B 数据产物（有用）**：`hawor_results npz`（核心）、`SLAM/*.npz`（traj/scale/focal）、`cam_space/*.json`（真检出记录）、`est_focal.txt/mask_focal.txt`。
- **C 中间缓存（流水线读回，不能删）**：`tracks_*/` 四件套（命中跳过 HAWOR 推理；masks 供 focal 搜索）、`extracted_images/`（infiller 输入）、`world_space_res.pth`（与 hawor_results 重复的 dump，56K）、`run_demov2.log`。

补充（tracks 目录体积构成）：98% 是 `model_masks.npy`（视频7: 224M/228M，视频121: 1.2G/1.2G，T×1080×1920 bool 掩码，仅焦距搜索和 combined_render 用）。真正有数据价值的 `model_tracks.npy`（检测置信度）+ `frame_chunks_all.npy` 只有几十 KB。注意 hawor_video.py:69-70 缓存检查将 model_masks.npy 列为必需文件，直接删除会使缓存失效、触发 HAWOR 推理重跑；精简需改代码。

实施（可视化分离）：demov2.py 4 处输出路径改为 `output/<video>/visualization/` 子目录（vis_cam :847、vis_world :904、combined_render :938、vis_verify :195），数据产物与缓存路径不变。已有输出已归位。今后清理"没用的"只需 `rm -rf output/<video>/visualization/`。RAS 侧 depth_with_mano 未动（在 ReplicateAnyScene 目录，旧脚本 mano_ras_3d_from_vis.py 还依赖它）。

实施（--data_only 开关）：demov2.py 新增 `--data_only`（默认 False）。开启后跳过 vis_cam/vis_world/vis_verify/combined_render 四个可视化出口（含跳过 224MB model_masks.npy 加载），**depth_with_mano 始终输出（用户需要）**，数据产物照常。用法：`python demov2.py --video_path xxx.mp4 --data_only`。教训：对同一文件并行发多个 SearchReplace 会互相覆盖（vis_world 路径曾被回退），同文件编辑必须串行。


---





### Q10: 调用 demov2 有哪些方式？

**回答**: 唯一入口 `python demov2.py`，必须在 HaWoR 目录下运行（checkpoint/infiller 是相对路径）。参数：`--video_path`（输入 mp4）、`--data_only`（精简模式，跳过 vis_cam/vis_world/vis_verify/combined_render，depth_with_mano 永远生成）、`--img_focal`（手动焦距覆盖）、`--input_type`（默认 file）、`--checkpoint/--infiller_weight`（权重路径）。输出固定到 `HaWoR/output/<视频文件名去后缀>/`。要生成 depth_with_mano 需 RAS 侧已有 `ReplicateAnyScene/output_v2/<同名>_vggt_omega/{depth,intrinsics,extrinsics}`（demov2.py:506-510 自动推断），缺失时该步报错但被吞掉不影响主流程。运行环境：`/mnt/data/lza/conda_envs/hawor/bin/python`。

补充（体积）：纯数据极小——视频7(113帧) ≈19M、视频121(600帧) ≈43M（约0.1MB/帧，npz 主导）；depth_with_mano 另加 10-20M。总占用的大头是可视化(~40%)和 tracks 缓存(~50%，其中 model_masks.npy 占 98%)。`--data_only` 后单视频产出约 30-65M。

---

### Q11: depth_with_mano 里的 MANO 点会不会有问题？有平滑插值吗？

**回答**: 无平滑无插值——hawor_video.py 全文唯一插值是 :140 的 interpolate_bboxes（补检测框，非 MANO 参数）。渲染用的 saved verts(model_verts.npy) 是 HAWOR 网络逐帧独立拟合的原始输出（121=438 帧非零）；infiller 补全参数(552 帧的 pred_*)只存 npz、不参与渲染，全零帧直接跳过不画。姿态质量上限=网络固有拟合精度（遮挡/模糊帧略差），管线未引入额外失真；投影位置由自检保证（100% 落入检测掩码）。

### Q12: demov2 现在整个逻辑流程？

**回答**: 六阶段——①YOLO+ByteTrack 检测跟踪(tracks 缓存, 伪相机系 focal=600 的 masks/verts) → ②HAWOR 运动估计(逐帧独立拟合, 无平滑) → ③强制焦距搜索(每视频独立, 121=1152/7=960) → ④SLAM 世界系+c2w+scale → ⑤infiller 世界系补全+hawor_results npz(pred_*为世界系) → ⑥RAS depth_with_mano 叠加(始终执行, saved verts@600 投影)+内置自检(关节命中率≥70% OK)。--data_only 可关掉四个可视化(vis_cam/vis_world/vis_verify/combined_render 归入 visualization/)。

### Q13: demov2 整体管线 + depth_with_mano 对齐管线流程图

**回答**: 管线A(demov2六阶段): ①YOLO+ByteTrack(tracks缓存,伪相机f=600的masks/verts) → ②HAWOR逐帧拟合 → ③强制焦距搜索(est_focal) → ④SLAM世界系c2w+scale → ⑤infiller补全+hawor_results(npz pred_*=世界系) → ⑥RAS叠加(始终)。可视化默认跳过(--with_vis 才生成)。管线B(depth_with_mano): mask_focal.txt选render_focal → saved verts覆盖率≥20%走主路@render_focal(否则run_mano兜底@img_focal) → ras_to_hw内容反查帧对齐 → 逐帧守卫绘制(pred_valid∧非NaN∧非全零)@1920x1080等比缩放到688x384 → 输出png+mp4 → 自检关节命中率>=70%。对齐原理: verts与masks同相机系同源(投影必中), depth与RGB同视角(像素等比缩放即对齐)。

### Q14: depth_with_mano 各概念详解(focal作用/saved verts/守卫/投影算法)

**回答**: focal=3D转屏幕的比例(u=f·X/Z+cx), verts在哪系生成就必须用哪系的f投影(mask_focal=600); saved verts=运动估计阶段HAWOR拟合存的778顶点(tracks_*/model_verts.npy), 与检测掩码同源故必中; img_focal+WARNING仅当mask_focal.txt丢失时顶替并提示风险, 不落盘冒充真值; 守卫=pred_valid∧非NaN∧非全零, 挡infiller虚标帧和全零帧(防关节聚画图心); 投影两步: 3D@f=600→1920x1080像素→乘(688/1920,384/1080)等比到depth像素(depth与RGB同视角只差分辨率)。兜底分支已从主流程图移除。

### Q15: 3D坐标重建管线(mano_ras_reconstruction)

**回答**: 管线C——①smplx前向(npz参数→778顶点+pred_trans) → ②分支C投影(f=600@1920x1080缩放到688x384, 与depth_with_mano同款) → ③查RAS深度并反投影(X=(u-cx)z/fx..., z=0/出界置NaN) → ④extrinsics w2c反变换转世界系 → ⑤J_regressor 16关节+5指尖顶点=OpenPose 21 → ⑥全NaN帧invalid。输出mano_ras_reconstruction_*.npz: verts/joints_ras_(相机系+世界系)+valid+帧映射+内参。与管线B方向相反(B是3D画到图上, C是从图上查回3D), 深度值直接决定Z。

### Q16: MANO前向通俗解释 + 管线C大白话版

**回答**: MANO=数字手办(778顶点网格模板), 两组旋钮: 姿态(15指关节角+腕朝向)+手型(betas10维); "前向"=拧旋钮摆姿势(LBS蒙皮自动), 输出778个3D顶点。管线C大白话: 读npz里的旋钮读数 → MANO摆出手 → 投影到depth图定位(f=600两步缩放) → 逐点查depth灰度×0.001=真实距离(米, 查不到置NaN) → 像素+深度反算3D(X=(u-cx)Z/fx) → extrinsics转世界系+J_regressor提21关节。意义: npz原本Z是网络估的, 此管线后每点深度都是depth图量出来的真值。

### Q17: 总结文档重写(可读性)

**回答**: 按反馈重写 demov2_pipeline_summary.md——开头新增11条术语表(verts/伪相机/缓存命中/MANO/c2w等); 管线A五阶段统一"干什么/输入/方法/输出/缓存"四要素格式并列明具体输出文件与形状; 管线B按步骤1-6展开(mask_focal选择逻辑/img_focal顶替+WARNING含义/帧对齐方法/前置检查三条件及原因/两步投影公式带数值例子); "守卫"更名为"绘制前三项前置检查"; 英文术语后均跟中文解释。
