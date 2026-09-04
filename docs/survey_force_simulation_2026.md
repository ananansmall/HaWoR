# 2026 手物交互「力估计 + 物理仿真」调研报告

> 调研日期：2026-08-28
> 目标：为 HaWoR（手物 in-the-wild 重建，输出 MANO 手网格）评估一条可落地的扩展方向 —— 「手物视频 → 力估计 → 物理仿真」完整 pipeline 的创新点可行性。
> 说明：所有源码可用性结论均为当天在本地环境实测（含网络连通性测试），非仅凭论文推断。

---

## 1. 背景

HaWoR 体系以 MANO 手网格 + 物体重建为核心，输入单目/多目视频，输出手物 3D 重建与逐帧运动。本调研关注两条与其天然衔接的 2026 年新方向：

1. **力/压力估计**：在已有 MANO 网格上额外回归逐顶点接触力/压力（无需力传感器）。
2. **物理仿真（inverse physics / Real2Sim）**：从视频恢复可交互、可施加外力的物理世界。

---

## 2. 力/压力估计方向（2026）

### 2.1 HOPE：Hand-Object Pressure Estimation from Monocular Videos
- 来源：arXiv 2608.06192，2026-08-06，首尔大学（Subin Jeon / Byungjun Kim / Hanbyul Joo）
- 项目页：https://subin6.github.io/page-hope
- **方法要点**：
  - 输入单目视频片段，在 MANO 手网格上输出**逐顶点法向压力 + 接触**（每帧 `p ∈ R^{V}`, `c ∈ [0,1]^V`），输出空间与物体形状、传感器布局无关。
  - 将三种异构标注统一到手顶点空间：触觉手套压力、平面传感器压力、距离型手物接触，用裸手接触数据正则化无压力标签的区域。
  - 提出 **VertexFormer**（顶点锚定视频 transformer）：每个顶点是一个持久 token，聚合视觉特征与手部姿态随时间变化。
  - **contact-gated pressure head**：`p̂ = ĉ ⊙ p̃`，把「无接触则压力为 0」的物理先验写进架构。
- 数据：OpenTouch、PressureVisionDB、DexYCB/ARCTIC（仅接触标注）。
- **与 HaWoR 的直接关系**：论文明确写道 "DINOv3 patch tokens and **HaWoR-derived vertex tokens** are fused via interleaved spatial and temporal attention" —— **HOPE 的力估计头直接消费 HaWoR 的重建输出**。
- 源码状态：❌ **未开源**（项目页 Code/Model 均为 "soon"）。

### 2.2 EgoPHI：Estimating Contact and Force from Egocentric Vision
- 来源：arXiv 2608.13014，**ECCV 2026**，ETH Zürich（Christian Holz 组）
- 项目页：https://siplab.org/projects/EgoPHI
- **方法要点**：
  - 从单张自视 RGB 图 + 物体几何，联合估计手与关节物体网格上的**稠密 3D 接触图 + 3D 力分布**（逐顶点）。
  - 核心思路：**用物理仿真 pipeline 给现有手物数据集（ARCTIC）生成逐顶点力真值**，解决力标注不可规模化的问题；再学视觉→力映射，做 sim-to-real。
  - 构造了两个真实力感物体，8 名受试者采集验证，验证 sim-to-real 迁移。
  - 评估：ARCTIC（域内）、H2O（域外），比基线误差降低约 2N。
- 源码状态：⚠️ 论文声明开源（`github.com/eth-siplab/EgoPHI`），但本地实测 raw `main`/`master` 均 404，**仓库目前未真正上传代码**（可能尚在整理，需持续跟踪）。

### 2.3 EgoTactile：Learning Grasp Pressure for Everyday Objects from Egocentric Video
- 来源：清华深圳国际研究生院（Qingmin Liao 组），2026-01 前后
- 项目页：https://egotactile.github.io ，代码：https://github.com/Russell-Zeng/EgoTactile
- **方法要点**：
  - **EgoPressureDiff**：从单段自视 RGB 视频生成动态全手压力图（像素对齐的回归方法受自遮挡/物理歧义限制，改用生成式框架）。
  - 配套 V2P 数据（visual-to-pressure）与真实压力真值。
- 源码状态：✅ **已开源**（含 train_hand.py / inference_hand.py / requirements.txt / command 脚本）。

### 2.4 TouchAnything / EgoTouch（数据集 + 基线）
- 来源：arXiv 2605.13083，哈工大（深圳）+ 美团机器人研究院
- 项目页：https://jianyi2004.github.io/TouchAnything-Website/ ，代码：https://github.com/Jianyi2004/TouchAnything（MIT）
- **方法要点**：
  - **EgoTouch**：首个大规模多视角自视触觉数据集。208 个操作任务 / 1,891 段 episode / 1000 物体 / 2M 帧，同步提供：自视 + 双手腕三视角视频、双手 42 关节 3D 手姿态、可穿戴传感器采集的**连续压力图**。
  - **TouchAnything**：多视角视觉→触觉预测基线（共享视觉编码 + 跨视角注意力 + view dropout）。加手腕视角可使 Contact IoU 提升 5.0%、Volumetric IoU 提升 6.1%。
- 源码状态：✅ **已开源**（MIT，含 configs/scripts/src + environment.yaml + 数据已上传）。

### 2.5 相关参考（非 2026）
- **PressureVision**（ECCV 2022，facebookresearch）：单张 RGB → 平面压力图，开源但局限于仪器化平面、单帧。
- **EgoPressure**：最接近 HOPE 的前作，构建于现成 MANO 重建之上，但逐帧估计裸手对平面压力。
- **UniHOPE**（2025，CUHK）：同名不同物，是手物**位姿**估计，与力无关，勿混淆。

---

## 3. 视频 → 物理仿真方向（2026）

### 3.1 LaGSplat：Latent Lagrangian Gaussian Splatting
- 来源：arXiv 2608.16324，2026-08-17，CEA（法国原子能委员会）
- 项目页：https://louenpottier.github.io/lagsplat.html
- **方法要点**：
  - 从一/少数单目视频推断受物理支配的动力学；推理时用户可对拍摄物体（刚体/可变形）施加任意**未见过**的外力，实时观察响应。
  - 低维隐状态 q 同时充当「耗散性 Lagrangian 的广义坐标」与「3DGS 解码器的条件变量」；显式随物运动的点基元使图像力 f 能通过 Jacobian `J(q)^T f` 回拉进运动方程。
  - 假设耗散 Euler-Lagrange 方程，用有界、合理的响应换取对未见外力的泛化。
- 源码状态：❌ **未开源**（预印本 + 交互 demo，无代码链接）。

### 3.2 MonoPhysics：Estimating Geometry, Appearance, and Physical Parameters from Monocular Videos
- 来源：arXiv 2605.30320，2026-05-28，UNC（Roni Sengupta / Meta）
- **方法要点**：单目视频中对**可变形物体**用可微分 MPM 仿真 + 3DGS 联合优化几何、外观、物理参数（杨氏模量等）。三个视觉-物理桥：全局尺度对齐、物理感知几何细化、可微位置图。
- 单目物理参数估计精度可匹敌 12 相机多视角基线。
- 源码状态：❌ **未开源**（无代码链接）。

### 3.3 OOVW / OVOW：One Video, One World（ECCV 2026）
- 来源：arXiv 2606.31388，清华 + 中科大 + SparcAI，ECCV 2026
- 项目页：https://onevideooneworld.github.io ，代码：https://github.com/SparcAI-Inc/OVOW
- **方法要点**：单目视频 → 实例级可仿真 4D Mesh 世界；估计地面/重力，修正悬浮/穿透/支撑；可重摆物体让物理引擎重算。
- 源码状态：✅ 仓库存在（已给出链接）；本地 github 主站不通，未进一步拉取验证。

### 3.4 其他相关
- **Agentic Real2Sim**（arXiv 2607.19190）：VLM 智能体把真实交互录制转成可运行物理仿真剧本，UBC/Columbia 等，无明确开源。
- **Phys2Real**（arXiv 2510.11689，Stanford）：VLM 物理参数先验 + 在线自适应做 sim-to-real，无明确开源。
- **LagrangianSplats**（SIGGRAPH 2026，北大）：无散度流体重建，已开源 `github.com/taoningxiao/LagrangianSplats`（流体方向，非手物）。
- **PIDG / Physics-Informed Deformable GS**（USTC）：可变形高斯粒子 + Cauchy 动量约束，已开源（流体/可变形方向）。

---

## 4. 关键发现

> **HOPE 官方架构 = 视频 → HaWoR 重建 → 逐顶点力估计**。这证明「HaWoR 作为估力上游」是 2026 年主流论文认可的架构，整合方向已被验证，而非臆想。

---

## 5. 源码可用性实测表（2026-08-28 本地实测）

| 模型 | 方向 | 源码 | 备注 |
|---|---|---|---|
| HOPE | 力估计 | ❌ 未开源 | Code/Model "soon" |
| EgoPHI | 力估计 | ⚠️ 名义开源空仓 | raw 404，待跟踪 |
| EgoTactile | 力估计 | ✅ 已开源 | 清华深研院，可跑 |
| TouchAnything | 力估计 | ✅ 已开源 MIT | 含 EgoTouch 数据 |
| LaGSplat | 物理仿真 | ❌ 未开源 | 预印本 |
| MonoPhysics | 物理仿真 | ❌ 未开源 | 预印本 |
| OVOW | 物理仿真 | ✅ 仓库存在 | github 主站不通未验证 |
| PressureVision | 力估计(参考) | ✅ 已开源 | 2022，平面压力 |

---

## 6. 本地环境评估（2026-08-28 实测）

- GPU：4×RTX A6000（48G/卡），共享占用，每卡空闲约 14~28G。
- hawor 环境：`/mnt/data/lza/conda_envs/hawor`，Python 3.10、torch 1.13.0+cu117、pytorch3d 0.7.4、smplx 0.1.28、mmcv-full 1.7.2、trimesh —— 手物重建栈完整，足以承载「HaWoR 重建 → 力回归头」的推理与轻量训练。
- 内存 251G（可用 ~180G），磁盘 ~2.5T 可用。
- **网络瓶颈**：
  - `github.com` 主站直连超时（无代理、无 git url 改写）→ **git clone github 仓库不可行**。
  - `raw.githubusercontent.com` ✅、`pypi.org` ✅、`hf-mirror.com` ✅（已配置 `HF_ENDPOINT=https://hf-mirror.com`）。
  - 结论：可 `pip install`、可从 HF 镜像下载权重；拉 github 源码需走 raw 单文件、镜像或代理。

---

## 7. 整合可行性结论

1. **方向成立**：HOPE 官方即采用「HaWoR → 力」架构；物理仿真侧（LaGSplat/MonoPhysics）证明单目视频可做交互式物理仿真。三者整合成「手物视频 → 力 → 仿真」是自洽且有论文支撑的叙事。
2. **现阶段无法复现论文原版**：HOPE/LaGSplat/MonoPhysics 未开源，EgoPHI 空仓；github 主站不通。
3. **可先落地的替代路径**：
   - 用 HaWoR 现有重建 + 自建逐顶点压力回归头（参考 HOPE 的 VertexFormer + contact-gated head 设计）；
   - 压力真值来源二选一：a) 开源数据集 EgoTouch/TouchAnything 或 EgoTactile 的 V2P 数据；b) 仿 EgoPHI，用物理仿真（现成可微 MPM，如 PhysRig/PhysDreamer 思路）给手物接触生成力真值。
   - 物理仿真输出侧：可先用现成开源组件（MPM/PhysRig 类）做「给定力 → 物体响应」仿真，验证闭环。
4. **创新点定位建议**：把「in-the-wild 手物视频 + 可解释的逐顶点力 + 可交互物理仿真」三者统一，正是 HOPE（只到压力）与 LaGSplat（只到仿真）各自缺一半的完整闭环。

---

## 8. 参考链接

- HOPE 项目页：https://subin6.github.io/page-hope / 论文 https://arxiv.org/abs/2608.06192
- EgoPHI 项目页：https://siplab.org/projects/EgoPHI / 论文 https://arxiv.org/abs/2608.13014
- EgoTactile：https://egotactile.github.io / https://github.com/Russell-Zeng/EgoTactile
- TouchAnything/EgoTouch：https://jianyi2004.github.io/TouchAnything-Website/ / https://github.com/Jianyi2004/TouchAnything
- LaGSplat：https://louenpottier.github.io/lagsplat.html / 论文 https://arxiv.org/abs/2608.16324
- MonoPhysics：论文 https://arxiv.org/abs/2605.30320
- OVOW：https://onevideooneworld.github.io / 论文 https://arxiv.org/abs/2606.31388
- PressureVision：https://github.com/facebookresearch/PressureVision
- PhysRig（可微 MPM 参考）：https://github.com/haoz19/PhysRig


---

## 9. 与 HaWoR + ReplicateAnyScene 的结合：数据生成方案（重点）

### 9.1 项目目标回顾

Ego-Video-to-SIM：从**自视手物交互视频** →（HaWoR 世界系 MANO 手重建）+（ReplicateAnyScene 场景/物体 GLB/6D pose/物理仿真）→ **LeRobot v3 兼容的机器人可执行数据集**。
本方案在既有链路上**新增一条「接触/力」通道**，生成可用于训练「估力模型」（HOPE/EgoPHI 式）的数据，这正是用户强调的**关键：生成数据**。

### 9.2 结合点（代码层面，已核实）

| 组件 | 位置 | 输入 → 输出 | 与「接触/力」的关系 |
|---|---|---|---|
| HaWoR 手重建 | `HaWoR/demov2.py`、`hawor_detect_wrapper.py` | 视频 → MANO 世界系参数 `pred_trans/rot/pose/betas` + 21 关节点 + 顶点网格 | **估力的载体**（HOPE 官方即用 HaWoR vertex tokens） |
| ReplicateAnyScene 场景 | `ReplicateAnyScene/mainv2.py`（Stage1-5） | 视频 → GLB + SAM3 masks + 场景图 | 物体几何用于「几何接触」判定 |
| 运动耦合抓取检测 | `object_tracking/03_grasp_controller.py` | 物体轨迹 + 手部顶点 → `gripper_timeline` / `grasp_poses` / `interaction_segments` | **已实现**手物接触的运动学判定（V-Dreamer 思路） |
| 物理仿真 | `object_tracking/simulation/run_simulation.py` | `--hawor_npz` + `--action_json` + `--scene_glb` → `sim_results.npz` + `verification.json` | 物理仿真可**生成接触力真值**（EgoPHI 思路），已接通 GalaxeaManipSim |

### 9.3 真值来源（2026-08-29 本地实测可获取性）

| 来源 | 数据 | 代码 | 本地可下载 | 结论 |
|---|---|---|---|---|
| **EgoTouch**（TouchAnything 数据集） | 三视角视频 + **真实 162 传感器逐帧压力** + 左右手接触标注 + 手姿态 | TouchAnything 已开源 MIT | ✅ hf-mirror 实测 200 可下 | **唯一选择（数据源）** |
| **OpenTouch**（MIT） | 真实力手套压力 + 自视视频 | 已开源 | ❌ 数据在 Google Drive，本地不可达 | 排除（仅代码可下） |
| **EgoTactile**（清华深） | 真实压力手套（162 传感器） | EgoPressureDiff 已开源 | ✅ hf-mirror 可下 | **仅作估力方法参考，不选作数据源** |
| GRAB/ARCTIC/DexYCB | 仅解析/几何接触 | 有 | - | 非真实力，不满足「要有接触的」 |

### 9.3.1 明确选择（唯一推荐，不再二选一）

**数据源与代码主线 = TouchAnything / EgoTouch**，理由（均为 2026-08-29 实测）：

1. **数据 in-the-wild 且贴合 HaWoR**：EgoTouch 覆盖 Home/Office/Outdoor/Retail/Workbench 五类真实场景、208 任务/1891 段、三视角（自视+双手腕），正是 HaWoR 的自视视频输入形态。
2. **代码自带「传感器→MANO 顶点映射」**（见 §9.4.1）：仓库里已有 `ta_to_mano_mapping_*.json`、`pressure_position_mapping_*.json`、MANO 模板 obj、`pressure_map.py` —— 本方案 §9.5 的"核心改造"**无需自写，直接复用**。
3. **接触真值齐全**：`manual_contact_annotation.json`（左右手二元接触）+ `jq_pressure.json`（162 传感器压力）。
4. **基线轻量**：TouchAnything 是标准 enc-dec 训练管线（`train.py`），无 diffusion/SVD 大权重依赖。

**EgoTactile（EgoPressureDiff）不选作数据源的原因**：数据集 63 物体、受控 green-screen 采集（非 in-the-wild）；代码混杂大量无关人脸识别框架代码（`core/helper/` 的 iresnet/glint360k 等）；依赖 Stable Video Diffusion（SVD）backbone 权重，推理/训练重。仅其「视频→动态压力」diffusion 思路可作为估力模型的方法参考。

### 9.4 EgoTouch 数据格式（已实测下载样例）

单个 episode（如 `Home/arrange_pillow/20260412_101136_379/`）：
```
chest.mp4 / left.mp4 / right.mp4      ← 自视 + 双手腕三视角视频（HaWoR 输入）
jq_pressure.json                      ← 逐帧真实压力，每行 {ts, frame_index,
                                         sensor_left:[162], sensor_right:[162],
                                         quat_left, quat_right}
pressure_grids.npz                    ← 压力网格
manual_contact_annotation.json        ← {left_contact: bool, right_contact: bool, ...} 二元接触
wilor_hands.json / rokoko_hands.json  ← 手部姿态（部分帧为 None）
vive_poses.json                       ← 相机/设备位姿
visualization.mp4                     ← 可视化
```

### 9.4.1 TouchAnything 代码关键文件（已实测 github api 完整文件树）

**传感器→MANO 映射（可直接复用，无需自写）：**
- `src/resources/pressure_mappings/pressure_position_mapping_left.json` / `right.json`：162 传感器空间位置
- `scripts/tools/mano_visualization/ta_to_mano_mapping_left_visual.json` / `right_visual.json`：触觉阵列(TA)→MANO 顶点映射
- `src/resources/mano_right_neutral_subdiv.obj`：MANO 模板网格
- `src/utils/pressure_map.py`：压力→网格工具
- `src/utils/vis_pressure.py` / `visualization.py`：可视化

**模型与训练/推理：**
- `src/models/touch_anything.py`：基线模型（多视角视觉→触觉）
- `scripts/core/inference_from_video.py`：视频→触觉推理入口
- `scripts/core/inference_tactile_parallel_mano_style.py`：MANO 风格触觉推理
- `scripts/core/train.py`、`convert_to_hdf5.py`、`create_dataset_split.py`
- `configs/touchanything_with_glove_aug_wilor.yaml`：训练配置
- `environment.yaml`：环境定义

### 9.5 核心改造：162 传感器 → MANO 顶点映射（复用 TouchAnything 自带映射）

真实压力传感器是「手套坐标」，估力模型要的是「MANO 顶点坐标」的压力。TouchAnything 已提供映射资源：
```
EgoTouch jq_pressure（162 传感器 + quat 手姿态）
   → 对齐 HaWoR 世界系 MANO 顶点（复用 §9.2 各组件）
   → 用 ta_to_mano_mapping_*.json + pressure_position_mapping_*.json 映射到 MANO 顶点
   → 生成 p_vertex ∈ R^(V) × T（逐顶点压力）+ c_vertex ∈ [0,1]^V（接触掩码）
```
- 传感器→顶点映射**直接复用 TouchAnything 仓库文件**（§9.4.1），仅需把手姿态对齐到 HaWoR 输出坐标系。
- 此映射即 HOPE 论文「把触觉手套/平面传感器/距离接触三种标注统一到手顶点空间」的现成实现。

### 9.6 三种接触/力真值生成路线

| 路线 | 真值来源 | 用途 | 成本 |
|---|---|---|---|
| **A. 真实压力监督（推荐先做）** | EgoTouch 真实 162 传感器压力 → MANO 顶点映射（§9.5） | 训练 HOPE 式逐顶点压力头 | 低，数据已可下 |
| **B. 几何接触** | HaWoR 手网格 + 物体 GLB 网格距离/穿透；`grasp_controller` 运动耦合切分接触时间段 | 接触掩码 `c_vertex` | 低，代码已有 |
| **C. 物理仿真生成力（EgoPHI 思路）** | 把 HaWoR 手 + 物体 GLB 放入 SAPIEN（`run_simulation.py` 已通 GalaxeaManipSim），跑物理接触导出逐顶点接触力 | 合成力真值，可任意扩展物体/场景/抓取 | 中高 |

### 9.7 端到端数据生成 pipeline（代码骨架）

```python
# data_gen_pipeline.py —— 一个视频 → 一个「接触/力」标注 episode
def video_to_force_episode(video_path, output_dir, use_touch_gt=True):
    # 1. HaWoR 手重建（world 系 MANO）
    hawor_npz = run_hawor(video_path)                    # demov2.py 输出 pred_trans/rot/pose/betas
    mano_verts, mano_joints = forward_mano(hawor_npz)    # 顶点网格 + 21 关节点

    # 2. ReplicateAnyScene 物体/场景重建（SAM3 分割 + GLB + 6D pose）
    glb, obj_poses = run_replicate_any_scene(video_path)

    # 3. 接触/力真值（三选一或组合）
    if use_touch_gt and has_touch_gt(video_path):          # 路线 A：真实压力监督
        touch = load_touch_gt(video_path)                  # jq_pressure.json + quat
        p_vertex, c_vertex = map_sensors_to_mano_vertices(
            touch, mano_verts, ta_to_mano_mapping)         # 复用 §9.4.1 映射文件
    else:                                                  # 路线 B：几何接触
        c_vertex = geometric_contact(mano_verts, glb, obj_poses, threshold_mm=5)
        p_vertex = None

    # 4. 接触时间段切分（复用运动耦合，可选）
    segments = grasp_controller_segments(mano_joints, obj_poses)   # 03_grasp_controller.py

    # 5. 输出 episode
    write_force_episode(output_dir, {
        "video": video_path, "mano": hawor_npz,
        "obj_glb": glb, "obj_poses": obj_poses,
        "contact_vertex": c_vertex,          # (V, T) bool / float
        "pressure_vertex": p_vertex,         # (V, T) float（路线 A）
        "interaction_segments": segments,
    })
```

### 9.8 输出格式

- **估力模型训练格式（HOPE 风格）**：`(N, V, T)` 逐顶点压力 + `(N, V, T)` 接触掩码 + HaWoR 手参数 + 物体 GLB/6D。
- **机器人数据集格式（LeRobot 风格）**：复用 Roadmap §0.5 的 `episode.meta.json`，追加 `contact_vertex` / `pressure_vertex` / `force` 字段（`info.json` 的 features 表里加 `observation.tactile.*`，该表已预留扩展点）。

### 9.9 接入计划（具体步骤）

1. **拉取代码**：TouchAnything 仓库（github api / codeload 均可达，本地 github 主站不通则走 api 逐个下载或 codeload tarball），解压到 `ReplicateAnyScene/../tactile_toolkit/` 或 `HaWoR/../TouchAnything/`。
2. **下载数据**：用 `huggingface_hub`（HF_ENDPOINT=hf-mirror.com）下载 EgoTouch 选定 subset（先 Home/ 数个 episode）。
3. **手重建**：对 episode 的 `chest/left/right.mp4` 跑 HaWoR（`demov2.py`），得到世界系 MANO。
4. **映射**：用 `ta_to_mano_mapping_*.json` + `jq_pressure.json` 生成逐顶点压力/接触（§9.5），落盘 `(V,T)` 数组。
5. **场景**：同一视频跑 ReplicateAnyScene 得物体 GLB/6D（路线 B/C 需要）。
6. **验证**：可视化手网格上压力/接触着色，对照 `manual_contact_annotation.json` 检查接触时间段是否一致。
7. **产出**：HOPE 风格数据集目录（含 split.json）或 LeRobot sidecar 追加字段。

### 9.10 已实测可行性结论

- **代码**：TouchAnything / EgoTactile 源码经 github api 确认完整可获取（文件树已实测列出）；HaWoR + ReplicateAnyScene 为本地已有代码。
- **数据**：EgoTouch 在 hf-mirror 实测可下载（含压力 + 接触标注）；OpenTouch（Google Drive）本地不可达。
- **环境**：hawor env（torch 1.13 / pytorch3d 0.7.4 / smplx）可跑 HaWoR；A6000 单卡空闲 14~28G 足够推理与轻量训练。
- **网络限制**：github 主站不通需走 api.github.com / codeload；Google Drive 不通需避开 OpenTouch 数据源。
- **建议起步顺序**：先跑通「EgoTouch 一个 episode → HaWoR 手重建 → ta_to_mano_mapping 逐顶点压力/接触 → 可视化验证」，再决定是否加路线 C 物理仿真力真值。


## 10. 压力数据的实际调用：接入 Ego-Video-to-SIM 夹爪映射

### 10.1 两类产物及各自用途

| 产物 | 类型 | 用途 |
|---|---|---|
| `viz_*_f*.png` 压力着色图 | 调试可视化 | 肉眼确认"何处受力、映射是否正确"，**非最终产物** |
| `vertex_pressure.npz`（(T,778) 逐顶点压力 + 顶点网格) | 数据 | **真正供下游调用的数据**，喂给夹爪决策 |

压力着色图的价值 = 让"手哪里受力"肉眼可见，是校验映射正确性的手段；实际消费的是 npz。

### 10.2 在 Ego-Video-to-SIM 链路上的接入点（已核实代码）

Ego-Video-to-SIM 用 dex_retargeting 把 MANO 手引导到 R1 夹爪，约束点（`04_physics_simulation.py` L889-920）:
```
target_link_human_indices = [4, 8, 0]   # 食指尖, 拇指尖, 手腕
```
其物理仿真的 `_fetch_contacts`（L685-715）用 `scene.get_contacts()` 读夹爪-物体冲量。

**接入方式（顶点压力 → 指尖力 → 夹爪决策）**:
1. MANO 顶点按手指分组（MANO 指尖 joint=4/8/12/16/20 对应拇指/食指/中指/无名指/小指）
2. 从 `vertex_pressure.npz` 聚合每帧每手指区域受压力 → `finger_force[5]` + 接触掩码
3. 在 `_physics_step` 前，用 `finger_force` 加权夹爪 gripper 的 drive target / 拧紧刚度，作为"人手真实受力"的**先验**
4. 已有的 `_fetch_contacts` 物理冲量作为**校验/兜底**，形成双源

### 10.3 数据格式衔接

- 我们输出 `vertex_pressure.npz`（numpy `.npz`，与 Ego-Video-to-SIM 用的 `hawor_results_*.npz` 同属 numpy, 无额外依赖）
- 帧索引对齐: 压力帧(ts 对齐) ↔ MANO 帧(start_idx..end_idx) ↔ 仿真控制帧(CONTROL_FREQ=30)
- 若采用 EgoTouch 官方数据，则其原始为 HDF5（.h5）; 我们已统一转换为轻量 .npz 便于 hawor 环境直接读

### 10.4 HDF5 说明

EgoTouch 官方每 episode 存为 `.h5`（层级二进制容器）:
```
file.h5
├── images/chest_color          (T,480,640,3)
├── images/left_color/right_color (T,480,640,3)
├── pressure/left_pressure_grid   (T,21,21)   ← 压力(已归一化[0,1])
├── pressure/right_pressure_grid  (T,21,21)
├── hands/wilor_{left,right}_joint_xyz (T,21,3)
├── hands/wilor_{left,right}_valid      (T,)
├── poses/chest_pose / left_pose / right_pose (T,7)
├── masks/glove_masks  (T,N,480,640)
├── metadata/attrs  ...
└── timestamps (T,)
```
特点: 单文件装多模态、支持切片读取、跨平台。Python 用 `h5py` (pip install h5py)。我们项目生成的暂为轻量 npz 以复用 hawor 环境。



## 11. 实测产出与工具固化（已跑通，落盘 HaWoR/output/）

### 11.1 工具
- `tools/generate_vertex_pressure.py` : 传感器压力 -> MANO 778 顶点压力
- `tools/generate_finger_force.py`   : 压力 -> 顶点 -> 5 指受力聚合(曲线+npz+着色图)

调用模块: **纯数据后处理**(numpy+matplotlib+json), 只用 TouchAnything 官方映射表
(assets/tactile/), 非大模型, 无 GPU 推理。HaWoR 才是大模型(重建 MANO 手)。

### 11.2 实测结果(演示: output/7 网格 + EgoTouch 压力)
输出目录: `output/tactile_sensor_demo/`
- finger_force_curve_right.png : 5 指受力曲线
- viz_right_f{0,49,77}.png    : 顶点压力着色
- finger_force.npz            : (T,5) 五指受力 + (T,778) 顶点压力 + 顶点
- 右手: 拇指峰值 454.0; 总受力 thumb=18938 index=1546 middle=807 ring=0 pinky=0
  显示拇指主导/食中指辅助/小指无名指未接触(抓握特征)

### 11.3 力的定义与局限(重要)
- 数值 = EgoTouch 传感器【相对压力强度】(0~454), **非牛顿**
- 未做物理标定, 仅时序/手指分布; 不等于夹爪真实受力
- 转牛顿需: 力 = 压力 × 受力面积, 并统一到夹爪 URDF link 坐标系
- 演示用 output/7(自采视频无同步压力), 真数据需 EgoTouch 本人视频跑 HaWoR


## 12. 方案B调研结论：TouchAnything 与 HaWoR 不兼容（已挖源码证实）

调研目标是「找能直接吃 HaWoR 778顶点 MANO 并输出顶点力的开源代码」。结论: **没有**。

### 12.1 TouchAnything 真实代码逻辑（证据）
- 核心输出 = 21x21 网格压力图 (`src/utils/vis_pressure.py::pred_to_21`), 套固定 _NAN_L/_NAN_R 传感器掩码;
  或 16x16 bend-sensor 网格 (`src/utils/pressure_map.py`, 用关节弯曲度伪生成)。**两者均非 MANO 顶点力**。
- `ta_to_mano_mapping_{l,r}_visual.json` 仅用于「MANO 3D 可视化」分支
  (README: "only used by optional MANO 3D visualization branches"),
  其 `mano_right_neutral_subdiv.obj` 是 subdiv(~12k)顶点, 非 HaWoR 的 778 顶点。
- 实测: 该表引用顶点编号 44..13314, 落在 0..777 仅 1.5%; 我们的 `if vid<778` 把 98.5% 静默丢弃。
  → 之前「6个generate跑通、拇指峰值454」是误用不兼容表的自欺结果, 不可信。

### 12.2 结论
- TouchAnything 数据管线(gripper网格压力/伪压力) 与 HaWoR(778顶点MANO+手物交互) 坐标系+格式不兼容, 弃用。
- HOPE(逐778顶点+接触, 最兼容) 与 EgoPHI 均未开源 → 无现成开源代码可直接消费 HaWoR 输出。

### 12.3 建议路线(方案A 正确版)
用【传感器真实 3D 位置】做几何最近邻投到 778 顶点, 而非顶点编号查表:
  HaWoR 778顶点MANO + EgoTouch真实压力(用 EgoTouch本人视频跑 HaWoR, 保证手形/坐标对齐)
  → 3D最近邻 → (778,)逐顶点压力 + 5指力
前提: EgoTouch 需提供传感器 3D 位置; 若 sensor 无坐标则需手套-手标定。

## 13. 全开源代码扫描：从视频估力是否有开箱可用者（2026-08 实测）

核心结论: **把 RGB 视频映射成「力(牛顿/压力)的起伏」目前没有任何官方放好预训练权重的开箱即用开源代码。**
能做此事的最接近真实者均需【自己训练】。以下按"能否真跑到出力"分级。

### 13.1 真实存在且相关的开源仓库（api 实测）
| 仓库 | 内容 | 能否开箱出"力" |
|---|---|---|
| **eth-siplab/EgoPressure** (ETH, CVPR2025) | 自视视频 + MANO 手 + 逐接触压力(带力值/UV). 代码齐全: egopressure/{dataset,train?,eval?,parquet,senselpad,mano,viewer,cli}, 自带 mano_models/ | 仅数据+工具, 需自训/推理, 无现成权重 |
| **apple/ml-egodex** (Apple, 379star) | EgoDex 灵巧操作数据可视化+compute_metrics.py, 结构极简(14文件) | 否(评估/可视化, 非估力网络) |
| **Sid2697/HOPformer** (ECCV2026) | Ego 双手 MANO + 物体姿态, EPIC-Contact 接触标注 | 仅到接触状态, 无力值 |
| **Jianyi2004/TouchAnything** | 见第12章, 输出 21x21/16x16 网格压力, 与 HaWoR 不兼容 | 否(格式不兼容+无权重) |
| HOI-DETR / EgoDex(数据集) / EPIC-Contact | 手物交互检测/接触标注 | 仅接触, 非力 |

### 13.2 为什么"视频→力(牛顿)"无开箱者
- 人手-物体逐顶点真实力标注极少(需力感手套), 数据稀缺 → 相关模型仍处论文+未开源阶段: **HOPE**、**EgoPHI**(均 ECCV2026, 无代码)
- 最接近真事的 EgoPressure 只有数据+训练工具, 无预训练权重 → 若要"视频估力"必须自己训(需 GPU+数据+时间)

### 13.3 务实路径（按可靠性排序, 供决策）
1. **Ego-Video-to-SIM 已有物理仿真闭环**: 04 脚本 SAPIEN `_fetch_contacts` 的 impulse = 唯一开箱即出「真实牛顿力」的方式(虽是仿真力非视频估力)。零外部依赖, 最稳。
2. **EgoPressure 作自训数据源**: 训一个 自视视频→手压力 模型, 工程量较大(GPU/时间)。
3. TouchAnything/HaWoR 顶点查表: 已确认死路(第12章)。

### 13.4 决策
用户选择: 先写调研文档(本章)。后续可在 13.3 的 1/2 之间选择。
