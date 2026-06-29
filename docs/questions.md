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
