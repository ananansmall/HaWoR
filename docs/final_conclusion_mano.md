# depth_with_mano / npz 正确性最终结论 + MANO 参数理解

> 整理自排查对话，对象：视频 7、视频 116、视频 121，HaWoR `output/` 与 RAS `output_v2/`。
> 结论标注了【最终定论】【修正】【已证实】。
> 2026-08-29 复核：用 121 数据实测重验证了 depth_with_mano 生成配方与坐标系结论，
> 修正了本文多处旧描述（见各处【2026-08-29 修正】与文末"五、复核补充"）。

---

## 一、最终定论：depth_with_mano 与相关 npz 是否对

### 1. 视频 116 的 depth_with_mano ——【正确】

验证对象：`ReplicateAnyScene/output_v2/116_C5_CellPhone_185deg_vggt_omega/depth_with_mano/*.png`（右手绿色 mesh）。

判据（非循环验证，用的是深度分离 + 图像直方图）：
- green 区域 depth 集中在 **0.81–1.07m**，正是主物体（手机+手）所在深度层
- 全图有效 depth 0.35–1.5m，手是最近主体，mesh 铺开与背景深度**一致**，无错位
- `mask_focal=600`，`depth_with_mano` 生成和反投影都用 F=600，自洽

结论：**数据可用，depth_with_mano 是正确的。**

### 2. 无法用“绿色区域占比/深度跨度”判断错位 【已修正】

之前用“green 占 42% 图像 / 深度跨度 0.64m → 疑似错位”的判断是**错误的**。
原因：本场景手拿手机极度贴近相机，手本身就是大面积近距主体，mesh 投影大、深度跨度大是**正常现象**，不等于错位。
已修正：只有当 mesh 深度与**相邻背景**深度明显不一致时，才是真正的错位。

### 3. 焦距 F=600 vs SLAM 焦距 【已修正，非 bug】【2026-08-29 修正数值】

`mask_focal.txt=600` 是 MANO 顶点的“出生焦距”（阶段②伪相机系）。
`hawor_results.npz` 的 `img_focal` 是 SLAM 焦距（**116 实测=1056，121 实测=1152**，SLAM 自动搜索后的值；旧文写 960 有误），**两套独立坐标系**。
`depth_with_mano` 生成用的是 F=600（匹配 mask_focal），所以反投影也应用 F=600。
之前判断“F=600 与 SLAM 焦距混用导致 68px 偏差”是**错误的**——两者根本不是同一时刻用的值。

实测佐证（121）：depth 视图 `intrinsic.txt` 的 fx≈391 直接投影**不匹配**原 overlay；
按"F=600@1920×1080 投影后按 688/1920、384/1080 缩放”（2dscale）投影与原 overlay 边界完全重合
→ RAS depth 视图与源视频同视角，2dscale 假设对本批数据成立，`intrinsic.txt` 的 391 不是生成用的焦距。

### 4. 脚本关系 ——【正确理解，无需删除】【2026-08-29 更新状态】

| 脚本 | 输入 | 方法差异 | 定位/现状 |
|---|---|---|---|
| demov2.py | RGB 视频 | 只生成 `hawor_results_*.npz`（参数+相机）+ vis_cam/vis_world/vis_verify | 上游，现版不产 depth_with_mano。**旧版曾有 `generate_ras_depth_vis` 生成 depth_with_mano，后被删且未提交 git**（`git log -S` 无记录）→ `run_ras_depth_vis.py` 现调用它已失效（AttributeError），勿再使用 |
| project 生成 depth_with_mano | `tracks_*/model_verts.npy` | F=mask_focal(600)@1920×1080 投影后 2D 缩放到 688×384；帧映射 ratio=hawor帧/RAS帧 自动推断（7:1.0，121:6.0）；跳过未跟踪全零帧 | 生成正确可视化（配方已于 2026-08-29 用 121 实测复现确认，见"五"） |
| mano_ras_reconstruction.py | hawor_results.npz | smplx 前向 + F=600 投影查 depth | v1；帧映射**内容反查**（`build_ri_to_hw`，线性 ratio 仅 fallback），旧文"帧映射线性"说法已过时 |
| mano_ras_3d_from_vis.py | model_verts / 颜色提取 + RAS depth | 帧映射内容反查 + 插值 + 验证 | v2，单手持。⚠️ 其 mesh 颜色提取 fallback 阈值宽松，在未跟踪帧会把 JET 色带绿色背景误提为 mesh（`mesh_3d` 出现数万垃圾点）；主输出 `joints_3d_ras` 不受影响 |

- v1 与 v2 不冲突：v1 保留“778 顶点 + RAS 世界系 + 双手”能力；v2 帧对齐更准。
- 视频 116 v2 输出：600 帧 mesh z=[0.386,1.137]，21 关节无 NaN，插值误差 mean 7.3mm，数据良好。
- ⚠️ `mano_ras_3d_from_vis.py` 的 `mano_2d` 硬编码 RF=600，只对 mask_focal=600 的视频（7/15/116/121）正确；
  151/hoi4d 的 mask_focal=1056，若要对它们跑 RAS 对齐需先把 RF 改为读 `tracks_*/mask_focal.txt`。

---

## 二、MANO 参数拆解

### 1. 三组参数 【2026-08-29 修正坐标系】

| 参数 | npz key | 维度 | 含义 |
|---|---|---|---|
| shape（形） | `pred_betas` | (2,T,10) | 手型大小胖瘦，全程基本恒定 |
| global_orient（腕） | `pred_rot[0]` | (2,T,3) | 手腕根关节旋转 |
| hand_pose（指） | `pred_hand_pose` | (2,T,45) | 15 指关节轴角，45=15×3 |
| trans（位） | `pred_trans` | (2,T,3) | **SLAM 世界系**平移（`cam2world_convert` 之后；OpenCV 约定）。旧文"相机空间位移"不准确 |
| valid（掩码） | `pred_valid` | (2,T) | 检测跟踪命中=1；infiller 补**中间空洞**帧也置 1；**尾部**未跟踪段保持 0（如 121 的 552–599 共 48 帧） |

`pred_rot` 只存根关节，`pred_hand_pose` 存剩余 15 关节，合计 16 关节旋转。

坐标系补充说明（易混点）：
- npz 里 `pred_*` 是世界系；但对 121/116 这类**静止相机**视频，SLAM 世界系≈第 0 帧相机系，
  数值上与相机系几乎重合——这解释了为何“世界系参数”用 F=600 伪相机投影仍然 work。
- npz 里 `R_c2w/t_c2w` **是 `R_x=diag(1,-1,-1)` 翻转后的**（demov2.py:562-563 先翻转、:676 后保存；
  实测 `npz['R_c2w'] == R_x @ R_slam`），与 `questions.md`"仍是原始 OpenCV 系"的旧描述相反，
  且与未翻转的 `pred_*` 约定不同——下游同时用两者放手工件时要注意。

### 2. betas（shape）每一维

10 维是手型 PCA 系数，乘在 `shapedirs`(778×10×3) 上：
- betas[0]：主干，控制整体粗细/宽窄（正胖负瘦）
- betas[1]：手掌长 vs 手指长比例
- 其余：指尖长、指头粗细、掌厚等
- 基线 0 = 均值手型；视频 116 右手 betas 多为负（如 -0.8, -1.08, -1.5）→ 比均值瘦长、手指偏细

### 3. layer 计算链（从参数到 778 顶点）

```
Step1 axis-angle → 旋转矩阵 (Rodrigues, aa_to_rotmat)
       global_orient(1,3,3) + 15指关节(15,3,3)
Step2 shape 形变 (blend shapes) —— 只改顶点，不改骨骼
       v_shaped = v_template + Σ betas[i]·shapedirs[:,i,:]
Step3 pose 形变 + LBS 蒙皮 —— 带手型摆姿态
       J = J_regressor @ v_shaped          (778→16关节)
       v_posed = v_shaped + Σ pose_feature·posedirs
       顶点 = Σ_j W·(Rⱼ⊗Tⱼ)·v_posed        (按关节权重加权)
Step4 加位移 + 提取关节
       最终顶点 = v_skinned + pred_trans   (778,3)
       顶点坐标系跟随 pred_trans：喂 cam_space 参数→相机系顶点；
       喂 npz 世界系参数→世界系顶点
       关节 21 = J_regressor@顶点 + 5指尖   (OpenPose 序)
```

要点：`shapedirs` 只形变顶点，关节位置不变 → 手变胖但骨骼不动，仅表面蒙皮变化。
这也是 HaWoR 每个 chunk 只回归一次 betas、每帧只变 pose/trans 的原因。

### 4. HaWoR 代码落点
- `run_mano`（右手）：`infiller/hand_utils/process.py:6`
- `run_mano_left`（左手）：`process.py:82`，加载 `_DATA/data_left/mano_left`，并修左手 shapedirs x 分量符号 bug（`:106-107`），避免左右手镜像导致手型翻转
- `MANO.query`：`lib/models/mano_wrapper.py:42`，直接传旋转矩阵 + `pose2rot=False` 跳过内部 axis-angle 转换

---

## 五、2026-08-29 复核记录（121 实测）【已证实】

对象：`ReplicateAnyScene/output_v2/121_C5_CellPhone_161deg_vggt_omega`（depth 688×384 共 100 帧，
fx≈391.4）+ `HaWoR/output/121_C5_CellPhone_161deg`（mask_focal=600，tracks (2,600,778,3)，仅右手）。

### 1. 121 的 depth_with_mano（8月25日生成的那批 png）——【正确】

生成配方实锤（重投影完全复现）：
- 顶点源：`tracks_0_600/model_verts.npy` 右手（mask 相机系，出生焦距 F=mask_focal=600）
- 投影：F=600 @1920×1080（cx=960,cy=540）→ 2D 乘 (688/1920, 384/1080) 缩放到 depth 视图（2dscale / branch C）
- 帧映射：hawor帧 = ras帧 × (hawor帧数/RAS帧数)，121 为 ×6.0
- 判据一（复现）：10 个采样帧重投影 mask 与原绿色 mesh IoU 0.67–0.75，放大目视边界**完全重合**
  （差值全部来自原图叠加的黄色关节点 + 抗锯齿）
- 判据二（落点）：绿色 mesh 与 RAS depth 手部近距连通域的包含率 **0.91–1.00**（8/8 采样帧），
  mesh 区域深度 0.57–0.62m，正落在手的近距层（周边背景 0.63–0.76m）
- 反证：直接用 `intrinsic.txt`（fx=391）投影与原 overlay 完全不匹配 → 排除该路径

### 2. 缺 mesh 的帧不是 bug

100 帧中 0–9、45–54、92–99 无 mesh：对应 hawor 帧在 `model_verts` 里为**全零（未跟踪，非 NaN）**，
z=0 无法投影，原生成正确跳过。其中 92–99 ↔ hawor 552–599，正好也是 `pred_valid` 无效段。

### 3. 121 的 `mano_ras_3d.npz`（v2 输出）——【正确，附使用注意】

- 100 帧 RAS、joints 有效 73/100，插值 vs 参考自洽误差 median **4.5mm**（mean 12.1mm，少量快速帧拉高），
  关节 z 0.50–1.04m，未跟踪帧正确置 NaN → **主输出 `joints_3d_ras` 可放心用**
- ⚠️ `mesh_3d` 在未跟踪帧是垃圾：颜色提取 fallback（`g > max(b,r)+30` 阈值太宽松）把 JET 色带绿色
  背景误提为 mesh，每帧 5–6.6 万点铺满全图（正常帧恰为 778 点、跨度 ±5cm）。
  用 `mesh_3d` 时按 `len==778`（或与 `valid` 对齐）过滤；修复方向：fallback 改严格绿色
  （`r<70 & b<70 & g>150`）或 verts 无效时直接置空

### 4. 工具与后续

- 复核脚本：`test/verify_depth_with_mano_121.py`（hawor env 运行），判据可复用到任意视频：
  **重投影 IoU>0.6 + mesh 落在手部近距层** 即为正确
- `run_ras_depth_vis.py` 已失效（引用被删的 `demov2.generate_ras_depth_vis`，且该函数从未提交 git）；
  如需恢复 depth_with_mano 生成，按上文"五.1"配方重写即可（约 50 行：模型顶点→2dscale 投影→fillPoly）