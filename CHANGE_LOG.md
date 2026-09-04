# Change Log

项目变更记录。每次 Agent 任务完成后自动追加。

---

## [2026-06-02 14:10] - demov2.py: 确保所有数据来自视频真实估计，使用SLAM焦距和主点

### Changed
- `demov2.py`: 从SLAM输出中读取 `img_focal` 和 `img_center`，替代之前使用 `hawor_motion_estimation` 返回的焦距和图像中心计算的主点
- `demov2.py`: `generate_vis_verify` 函数新增 `cx`, `cy` 参数，使用SLAM主点而非硬编码的图像中心
- `demov2.py`: 添加详细日志输出，明确标注每个数据来源（SLAM、MANO等）
- `demov2.py`: 结果npz文件新增保存 `img_center` 和 `slam_scale` 字段

### Verification
- 往返投影验证通过：cam_space直接投影 vs world_space→R_x→w2c投影，3D差异=0，像素差异=0
- cam_space直接投影验证：frame 50/100/150 均成功渲染1538个面片和21个关节点
- SLAM数据确认：focal=600.0, center=[640, 360], scale=0.4403

### ⚠️ Docs to Review
- `HaWoR/HaWoR_Pipeline_Summary.md`: 可能需要更新关于焦距和主点来源的说明

---

## [2026-06-03 12:30] - SLAM焦距自动搜索 + demov2.py bug修复 + hoi4d视频测试

### Changed
- `hawor_slam.py`: 新增SLAM焦距自动搜索功能——当 `est_focal.txt` 包含默认值600但图像尺寸暗示焦距应更大时，自动调用 `search_focal_length` 搜索最佳焦距（基于SLAM重投影误差）
- `hawor_slam.py`: 搜索范围根据 `est_calib` 估算值动态调整（0.5x~1.5x），步长自适应
- `demov2.py`: 修复 `generate_vis_verify` 中 torch tensor `.transpose(0,2,1)` 错误，改为先转numpy再transpose
- `demov2.py`: 从SLAM输出中读取焦距和主点（上一轮修改）

### Verification
- hoi4d.mp4 (1920x1080, 300帧) 测试通过
- SLAM焦距搜索结果：600→1056（更接近真实值）
- SLAM scale：0.7244→1.023（更接近1，说明焦距更准确）
- 所有3种可视化输出完整生成：vis_cam(600帧)、vis_world(121帧+mp4)、vis_verify(双手各121帧+mp4)

### ⚠️ Docs to Review
- `HaWoR/HaWoR_Pipeline_Summary.md`: 需要更新关于焦距自动搜索的说明

---

## [2026-06-03 18:00] - demov2.py: Q&A总结 + 输出目录重构 + Bug修复

### Changed
- `demov2.py`: 添加文件头部Q&A总结，涵盖5个关键问题（数据来源、相机轨迹、scale、坐标系变换、焦距确定）
- `demov2.py`: 输出目录从视频所在目录改为 `HaWoR/output/<video_name>/`，所有可视化结果统一存放
- `demov2.py`: 修复 `draw_camera_trajectory` 中 `.numpy()` 缺少 `hasattr` guard 的潜在bug
- `demov2.py`: 修复 `draw_camera_trajectory` 未截取 `total_frames` 导致轨迹图可能显示多余帧的问题
- `demov2.py`: 移除 `draw_camera_trajectory` 未使用的 `traj` 参数
- `demov2.py`: 修复 `generate_vis_verify` 中图片路径忽略 `start_idx` 的潜在bug（改为 `f'{start_idx + idx:04d}.jpg'`）
- `demov2.py`: `generate_vis_verify` 新增 `output_base` 参数，输出到统一目录

### Verification
- hoi4d.mp4 完整测试通过，输出正确生成到 `output/hoi4d/` 目录
- vis_cam: 600帧, vis_world: 600帧+mp4, vis_verify: 双手各120帧+mp4, reconstruction: npz

### ⚠️ Docs to Review
- `HaWoR/HaWoR_Pipeline_Summary.md`: 需要更新输出目录说明

---

## [2026-06-05 18:45] - demov2.py: 文档独立 + 输出包含完整数据文件

### Added
- `demov2_doc.md`: 独立文档，包含Pipeline流程图、函数调用说明、输出目录结构、Q&A

### Changed
- `demov2.py`: 头部Q&A移至 `demov2_doc.md`，代码头部仅保留文档引用
- `demov2.py`: 输出目录现在包含完整数据文件（SLAM、cam_space、tracks、world_space_res.pth、est_focal.txt），不再只有可视化图片
- `demov2.py`: 使用 `shutil.copytree/copy2` 将关键数据从 seq_folder 复制到 output 目录
- `demov2.py`: 最终输出信息区分"Data files"和"Visualization"两类

### Verification
- hoi4d.mp4 测试通过，输出目录包含：
  - SLAM/hawor_slam_w_scale_0_600.npz
  - cam_space/0/, cam_space/1/
  - tracks_0_600/ (4个npy文件)
  - world_space_res.pth (295KB)
  - est_focal.txt (1056)
  - reconstruction/hawor_results_0_600.npz
  - vis_cam_0_600/ (600帧), vis_world_0_600/, vis_verify/

---

## [2026-06-07 20:00] - 新增 demo-vggt-omega: VGGT-Omega 替换 SLAM 相机估计

### Added
- `demo-vggt-omega/demo_vggt_omega.py`: 完整 Pipeline 脚本，用 VGGT-Omega 替换 DROID-SLAM 相机估计
  - VGGT-Omega 模型加载与推理 (w2c → c2w 转换)
  - 帧数插值 (SLERP 旋转 + 线性平移)，支持 max_frames 限制
  - 焦距从 VGGT-Omega FoV 推算 (分辨率自适应转换)
  - 可选 Room Alignment (SAM3 + floor/wall，需 ReplicateAnyScene)
  - 修改版 Infiller (直接使用 VGGT-Omega 相机位姿，无需 SLAM NPZ 文件)
  - 输出目录标识: `output/<video_name>_vggt-omega/`
- `demo-vggt-omega/README.md`: 使用说明文档

### Key Design
- 坐标系: VGGT-Omega 输出 OpenCV convention (Y-down, Z-forward), 可视化时 R_x=diag(1,-1,-1) 转 Y-up
- 尺度: slam_scale=1.0 (VGGT-Omega metric depth, 无需 Metric3D)
- 焦距: 从 VGGT-Omega FoV 推算原始分辨率焦距，而非 SLAM 估计
- 与 demov2.py 输出格式完全兼容 (npz 字段一致, 新增 camera_source='vggt_omega')

### ⚠️ Docs to Review
- `HaWoR/HaWoR_Pipeline_Summary.md`: 需要新增 VGGT-Omega 替换 SLAM 的说明

---

## [2026-06-07 21:00] - 修复 OOM + 分块注意力 + 函数注释 + 运行测试

### Changed
- `demo-vggt-omega/demo_vggt_omega.py`:
  - SDPA fallback 改为分块计算 (CHUNK=2048), 避免 O(N²) 内存分配
  - 加载模型前清空 GPU 缓存 (torch.cuda.empty_cache + gc.collect)
  - 默认 max_frames 从 160 改为 20 (47GB GPU 实测安全值)
  - 添加 extracted_images 到输出复制列表 (与 demov2 对齐)
  - 所有函数添加中文 docstring 注释
- `demov2.py`: 所有函数添加中文 docstring 注释 (不修改原有注释)
- `demo.py`: 添加中文 docstring 注释 (不修改原有注释)
- `demo-vggt-omega/README.md`: 更新 max_frames 默认值和 GPU 内存说明

### Verified
- hoi4d.mp4 (600帧) 完整 Pipeline 运行成功, 输出目录 `output/hoi4d_vggt-omega/`
- 输出结构: cam_space/, tracks_0_600/, extracted_images/, vggt_omega_cam/, reconstruction/, vis_cam_0_600/, vis_world_0_600/, vis_verify/, world_space_res.pth, est_focal.txt
- VGGT-Omega 20帧推理 → SLERP 插值到 600帧 → cam2world + infiller → 可视化

---

## [2026-06-07 22:30] - 新增 demo_comparison.md: 四个 Pipeline 对比总结

### Added
- `demo_comparison.md`: 对比 demo.py / demov2.py / demo-vggt-omega / mainv2.py 四个 Pipeline 的区别
  - 总览表格 (所属项目、核心任务、相机来源、可视化方式、Room Alignment、输出格式)
  - 流水线对比 (各 Pipeline 的完整流程)
  - 相机位姿生成方式对比 (DROID-SLAM vs VGGT-Omega vs VGGT)
  - 坐标系变换对比 (R_x 变换、Room Alignment)
  - 相机轨迹特征对比 (尺度、平滑度、物理含义)
  - 焦距对比
  - 适用场景
  - 数据兼容性

### Key Findings
- demo.py 和 demov2.py 使用 DROID-SLAM，轨迹平滑但尺度偏小
- demo-vggt-omega 使用 VGGT-Omega + 插值，度量尺度但插值段可能不平滑
- mainv2.py 使用 VGGT + Room Alignment，房间坐标系有物理含义但不涉及手部
- 三者轨迹形状相似，但尺度、坐标系和平滑度有显著差异

---

## [2026-06-07 23:00] - GPU 内存优化: VGGT-Omega 先运行→释放→再运行 HaWoR

### Changed
- `demo-vggt-omega/demo_vggt_omega.py`:
  - VGGT-Omega 推理后立即释放模型+结果, 仅保留 R_c2w/t_c2w/intrinsic/extrinsics (numpy)
  - 添加 GPU 内存监控: 推理后打印 `GPU memory after VGGT-Omega cleanup`
  - 修复 vggt_results 引用: 提前复制 intrinsic/extrinsics, 避免释放后访问

### GPU 内存实测 (RTX A6000 47.4GB)

| 阶段 | 峰值 GPU | 模型权重 |
|------|----------|----------|
| VGGT-Omega (20帧推理) | ~13.4 GB | 4.26 GB |
| VGGT-Omega 清理后 | 0.00 GB | - |
| HAWOR 推理 | ~3 GB | 2.58 GB |
| Infiller 推理 | ~0.5 GB | 0.13 GB |

关键: VGGT-Omega 清理后 GPU 占用 0.00GB, HaWoR 可完全独立运行, 无内存冲突。

### Verified
- `CUDA_VISIBLE_DEVICES=0` 单卡测试通过, hoi4d.mp4 600帧完整 Pipeline 成功

---

## [2026-08-21 14:00] - demov2.py: 修复 generate_ras_depth_vis 帧号错位 (depth_with_mano 无效根因)

### Bug
- `demov2.py::generate_ras_depth_vis` 把 RAS 深度图帧号 `fi` 直接当作 HaWoR verts/joints/valid 的索引，两者语义完全不同：
  - 原视频 15fps 300 帧
  - HaWoR ffmpeg 以 fps=30 抽帧 → 600 帧 (每帧复制)
  - RAS color/depth 每 3 个视频帧取 1 帧 → 100 帧
  - 真实对应: RAS ri ≈ HaWoR 6·ri
- 后果: depth_with_mano 越往后越错，第 99 帧叠加了视频帧 25 的手，命中率仅 0.379

### Changed
- `demov2.py::generate_ras_depth_vis`: 新增 **RAS→HaWoR 帧号内容反查** 逻辑（`ri_to_hw` dict）
  - 在函数开头自动扫描 `output/<video_name>/extracted_images/` 与 RAS `color/`，把每帧缩成 64×36 缩略图，逐帧 MSE 匹配得到真实对应
  - 新增 `ras_to_hw(ri)` 辅助函数，缺映射时兜底 `6*ri`
  - 当 RAS color dir 或 HaWoR 图片缺失时自动 fallback 到线性 `6*ri`
  - 循环体中 `fi` 保留为 RAS 帧号（深度图读取/输出文件命名），新增 `hwfi = ras_to_hw(fi)` 用于 `hand_verts/hand_joints/pred_valid` 索引
- 无参数/接口/调用方改动；函数签名、调用位置、输出路径全部不变

### Verification (121_C5_CellPhone_161deg)
- 内容反查命中 100/100 帧, `hw - 6*ri`: mean=1.02 std=1.04 (绝大部分 diff=0)
- 命中指标 (以 HaWoR mask 真值，21 关节投影是否落在手区域内):

| 索引方式 | 命中率 mean / median |
|---------|----------------------|
| 旧代码 `fi = ri` | 0.379 / 0.452 (错位) |
| 新代码 `hwfi = ras_to_hw(ri)` | **0.9993 / 1.0000** (73/73 帧 ≥ 0.9) |

- `python3 -m py_compile demov2.py` 通过，语法正确

### ⚠️ Docs to Review
- `HaWoR/HaWoR_Pipeline_Summary.md`: 如涉及 RAS depth_with_mano 说明，需补充"跨流水线帧号需内容反查"的注意点

---

## [2026-08-21 16:40] - demov2.py: 修复 generate_ras_depth_vis MANO mesh 漂移 (第二处根因)

### Bug
- 修复帧号错位后，生成图 mesh 仍与 depth 画面中手错位。实测发现:
  - `run_mano(pred参数)` 实时前向的 verts/joints 与保存的 `tracks_*/model_verts.npy` **不一致**
  - 两者差异随帧累积漂移: hw=60 时 maxdiff=0.025, hw=240 时 maxdiff=0.216
  - 保存的 model_verts/joints 与 model_masks 自洽(投影命中 1.0), 而 run_mano 前向结果与 mask 错位(overlap 0.0)
- 后果: 早期帧 (hw≤60) 对齐尚可(0.81), 中后期 mesh 完全画错位置(0.00)

### Changed
- `demov2.py::generate_ras_depth_vis`:
  - **优先加载已保存的相机空间 `model_verts.npy` / `model_joints.npy`** (自动扫描 `output/<video_name>/tracks_*/`)
  - 找不到时才 fallback 到 `run_mano` 实时前向 (并打印警告)
  - 无参数/接口/调用方改动

### Verification (121_C5_CellPhone_161deg, RAS depth_with_mano 新文件)
- MANO mesh 绿色像素落在手 mask 内比例:

| 版本 | 命中率 mean / median / min |
|------|---------------------------|
| 帧号错位 (fi=ri) | 0.379 / 0.452 / - |
| 帧号修复但 run_mano 前向 | 0.19 / 0.02 / 0.00 (mesh漂移) |
| 帧号修复 + 加载 saved model_verts | **0.9838 / 0.9856 / 0.9375** (73/73 帧 ≥ 0.9) |

- 27/100 帧无 MANO 叠加属正常: HaWoR 仅恢复出 438/600 帧手数据 (前56帧、中段296-310连续空缺等), pred_valid=False 自动跳过

### ⚠️ Docs to Review
- `HaWoR/HaWoR_Pipeline_Summary.md`: 需补充 "depth_with_mano 必须用 tracks_*/model_verts.npy 而非 run_mano 前向" 的注意点
- 旧的四位编号残留 `depth_with_mano/0000.png-0199.png` 为修复前错误版本，可手动清理

---

## [2026-08-21 17:25] - mano_ras_3d_from_vis.py 通用化 + 121/7 实测

### Changed
- `mano_ras_3d_from_vis.py`: 改造成任意视频通用脚本 (argparse 接口)，不再写死 121
  - `--video-name` 必填: 视频 basename (如 121_C5_CellPhone_161deg / 7)
  - `--ras-dir` / `--ras-suffix`(默认 _vggt_omega): RAS 输出目录
  - `--hawor-out`: HaWoR 输出目录 (默认 ./output/<video-name>)
  - `--hand` (0=left/蓝, 1=right/绿): 可选，默认从 cam_space 推断
  - `--mesh-color` (BGR): 可选，覆盖颜色提取
  - `--no-video`: 跳过 mp4 合成
  - 内参 fx/fy/cx/cy 从 `<ras_dir>/intrinsic.txt` 自动读取（121: 391.44/390.72, 7: 284.56/286.61）
  - 帧数从 `tracks_*/model_joints.npy` 形状自动推断
  - mesh 3D: 优先用 `tracks_*/model_verts.npy` 反投影（与 joints 自洽），缺则从图片颜色主导通道提取（右手 G 主导 / 左手 B 主导）

### Added
- `ReplicateAnyScene/output_v2/121_C5_CellPhone_161deg_vggt_omega/mano_ras_3d.npz`: 121 MANO 3D npz
  - hand=1(right), mesh 100/100 帧 (778 verts/frame), joints 73/100 帧有效, 插值到 600 帧
  - interp vs ref: median=0.0057m mean=0.0131m
- `ReplicateAnyScene/output_v2/7_vggt_omega/mano_ras_3d.npz`: 7 MANO 3D npz
  - hand=0(left), mesh 113/113 帧 (778 verts/frame), joints 113/113 帧有效
  - interp vs ref: median=0.1838m mean=0.3014m (因模型漂移, 参考对比偏大)
- `ReplicateAnyScene/output_v2/7_vggt_omega/depth_with_mano/depth_with_mano.mp4`: 7 视频 (113*3 @15fps)
- `ReplicateAnyScene/output_v2/121_C5_CellPhone_161deg_vggt_omega/depth_with_mano/depth_with_mano.mp4`: 121 视频重生成

### Verification
- 语法检查 ast.parse 通过
- 121: mesh non-empty=100/100, joints valid=73/100, dist median=5.7mm
- 7: mesh non-empty=113/113, joints valid=113/113, dist median=184mm (参考漂移)
- 两个 npz 字段一致: mesh_3d/object, joints_3d_ras(原始), joints_3d_ras_interp(插值+平滑), joints_3d_ref(参考), dist_interp, ri_to_hw, intrin, hand, hawor_frames

### Usage
  cd HaWoR && python3 mano_ras_3d_from_vis.py --video-name <name>
  或显式指定: --video-name 7 --hawor-out ../ReplicateAnyScene/assets/basic_pick_place/7

---
## [2026-08-22 15:50] - mano_ras_3d_from_vis.py: 去除硬编码, 新增 --frame-ratio/--no-smooth

### Changed
- `mano_ras_3d_from_vis.py`: 去除 3 处硬编码——(1) fallback 帧映射 `6*ri` 改为自动推断 `(T-1)/max_ri` 或 `--frame-ratio` 显式指定; (2) 插值时间轴 `3.0*ri`/`arange(T)/2.0` 归一化为 `[0,1]`, 不再依赖 fps/重复次数假设; (3) Savitzky-Golay 平滑改为可用 `--no-smooth` 关闭
- `mano_ras_3d_from_vis.py`: docstring 补充帧映射/时间轴/平滑说明与用法示例

### Verification
- py_compile 通过
- 7 (--no-smooth): fallback ratio 自动=1.0; interp vs ref median=0.0000m (修复前过度平滑为 0.17m)
- 121 (默认平滑): fallback ratio 自动=6.0505; interp vs ref median=4.8mm (优于原 7.2mm); joints valid=73/100
- raw vs ref 两视频均 0.0000m, 反投影自洽

### ⚠️ Docs to Review
- 无需更新 (工具脚本参数变化, 管线文档未涉及该脚本内部实现)

---
## [2026-08-22 16:00] - mano_ras_3d_from_vis.py: 默认不平滑 (与 demo.py 一致)

### Changed
- `mano_ras_3d_from_vis.py`: 平滑默认关闭, 与 demo.py 输出原始每帧值的策略一致; `--no-smooth` 反转为 `--smooth` 显式开启

### Verification
- 7 (默认): interp vs ref median=0.0000m, 无平滑
- 121 (默认): interp vs ref median=4.5mm (比原平滑的 4.8mm 略优), raw=0.0000m

---

## [2026-08-28 23:55] - 新增「手物视频→力估计→物理仿真」方向调研文档

### Added
- `docs/survey_force_simulation_2026.md`: 2026 年手物力/压力估计与视频物理仿真方向调研报告。覆盖 HOPE / EgoPHI / EgoTactile / TouchAnything / LaGSplat / MonoPhysics / OVOW 等模型的方法要点、源码可用性（本地实测）与环境评估。

### 关键结论
- HOPE 官方架构即「视频 → HaWoR 重建 → 逐顶点力估计」，验证 HaWoR 作为估力上游的方向成立
- HOPE / LaGSplat / MonoPhysics 未开源，EgoPHI 名义开源但仓库空；已开源的可落地替代为 EgoTactile / TouchAnything
- 本地 github.com 主站直连不通（无代理），但 pypi / hf-mirror / raw.githubusercontent 可达

### Docs to Review
- 无需同步：本次为新增独立调研文档，不影响 `HaWoR_Pipeline_Summary.md` / `tools/usage_guide.md` 中既有管线说明

---

## [2026-08-29 01:20] - 调研文档新增「数据生成方案」章节并明确唯一推荐（TouchAnything/EgoTouch）

### Changed
- `docs/survey_force_simulation_2026.md`: 新增第 9 章「与 HaWoR + ReplicateAnyScene 的结合：数据生成方案」：
  - 9.1 项目目标回顾（Ego-Video-to-SIM）
  - 9.2 两项目代码级结合点表（demov2.py / mainv2.py / grasp_controller.py / run_simulation.py）
  - 9.3 真值来源本地实测表；9.3.1 明确唯一推荐 TouchAnything/EgoTouch（不再二选一），EgoTactile 仅作估力方法参考
  - 9.4 EgoTouch 单 episode 数据格式；9.4.1 TouchAnything 代码关键文件清单（github api 实测完整文件树）
  - 9.5 核心改造：162 传感器 → MANO 顶点映射（复用 TouchAnything 自带 ta_to_mano_mapping）
  - 9.6 三条接触/力真值路线（A 真实压力 / B 几何接触 / C 物理仿真生成力）
  - 9.7 端到端 pipeline 代码骨架；9.8 输出格式；9.9 接入计划（7 步）；9.10 已实测可行性

### 关键结论
- 唯一推荐：TouchAnything（数据 EgoTouch 三视角+162压力+接触标注，代码 MIT 开源且自带传感器→MANO 映射）
- EgoTactile（EgoPressureDiff）排除为数据源：受控场景、代码混杂人脸识别框架、依赖 SVD 权重
- 已实测：hf-mirror 数据可下、github api 源码可拉、OpenTouch(Google Drive)不可达

### Docs to Review
- 无需同步：仅新增调研文档章节，不涉及 HaWoR 代码/管线变更

---

## [2026-08-30 23:10] - 固化「EgoTouch 真实压力 → HaWoR MANO 顶点」数据生成工具（实测跑通）

### Added
- `tools/generate_vertex_pressure.py`: 新工具。输入 EgoTouch `jq_pressure.json`（逐帧 256 传感器压力）+ HaWoR MANO npz（`verts_ras_world_left/right`），输出 `(T,778)` 逐顶点压力 + 可选压力着色 png。纯 numpy/matplotlib，hawor 环境可直接运行。
- `assets/tactile/ta_to_mano_mapping_{left,right}_visual.json`: TouchAnything 官方「传感器网格 → MANO 顶点索引」映射（固化自官方仓库）。
- `assets/tactile/pressure_position_mapping_{left,right}.json`: TouchAnything 官方「传感器网格 → 传感器索引」映射。

### Changed
- `tools/generate_vertex_pressure.py`: 修复 numpy 数组 `or` 真值判断歧义 bug（`len(mano_left or [])` → 用 None 判断）。

### 实测验证（hawor 环境）
- 输入: EgoTouch arrange_pillow episode jq_pressure.json(1652帧) + HaWoR output/7 mano_ras_reconstruction(113帧)
- 帧数对齐: min() 截断 → 113 帧
- 输出: `vertex_pressure.npz`（right (113,778)，2378 非零元素，峰值 32）+ `viz_right_f0.png`
- 结论: 真实压力→MANO 顶点数据生成链路在 hawor 环境实测通过

### ⚠️ Docs to Review
- `tools/usage_guide.md`: 新增工具建议补充使用说明（后续可加）
- `docs/survey_force_simulation_2026.md`: 第 9 章可行性结论可标注「已实测固化」状态

---

## [2026-08-30 23:30] - 调研文档补第10章：压力数据接入 Ego-Video-to-SIM 夹爪映射 + HDF5 说明

### Added
- `docs/survey_force_simulation_2026.md#10`:
  - 10.1 两类产物区分（着色图=调试可视化, npz=真正消费数据）
  - 10.2 Ego-Video-to-SIM 接入点（已核实 04_physics_simulation.py 约束点 [4,8,0] + _fetch_contacts）：顶点压力→指尖力→夹爪 drive target 先验，与物理冲量形成双源
  - 10.3 数据格式衔接（npz / 帧对齐 ts↔MANO↔仿真控制帧30Hz）
  - 10.4 HDF5(.h5) 格式说明（EgoTouch 官方层级结构 + h5py）

### Docs to Review
- 已直接更新 `docs/survey_force_simulation_2026.md`（新增第10章）

---

## [2026-08-31 15:50] - 固化 generate_finger_force.py 并实测落盘 output/

### Added
- `tools/generate_finger_force.py`: 传感器压力->MANO顶点->5指受力聚合, 输出曲线图+npz+着色图到 output/tactile_sensor_demo/
- `docs/survey_force_simulation_2026.md#11`: 实测结果与力的定义(相对压力强度, 非牛顿)

### Changed
- (工具验证) generate_vertex_pressure.py 与 generate_finger_force.py 均在 hawor 环境跑通

### 实测
- 输出 output/tactile_sensor_demo/: finger_force_curve_right.png, viz_right_f{0,49,77}.png, finger_force.npz
- 右手 拇指峰值454/总受力 thumb18938 index1546 middle807 ring0 pinky0

---

## [2026-08-31 16:50] - 方案B调研: TouchAnything 与 HaWoR 不兼容, 推翻此前的查表方案

### Changed(调研结论)
- 挖源码证实 TouchAnything 输出为 21x21/16x16 网格压力(非 MANO 顶点力);
  ta_to_mano_mapping 仅用于 3D 可视化且为 subdiv(~12k) 顶点, 与 HaWoR 778 顶点不兼容
- 此前 generate_vertex_pressure/finger_force 用该不兼容表, 实际只对上 1.5%(66/778), 结果不可信
- 结论: 无现成开源代码可直接消费 HaWoR 输出 → 改走方案A正确版(传感器3D位置几何最近邻)

### Docs
- docs/survey_force_simulation_2026.md#12: 方案B调研结论 + 建议路线

---

## [2026-08-31 17:30] - 全开源代码扫描：从视频估力是否有开箱可用者

### 调研结论(写入 docs/survey_force_simulation_2026.md#13)
- 从 RGB 视频映射为"力/压力起伏"目前无官方放好预训练权重的开箱即用开源代码
- 真实存在的相关仓库: EgoPressure(ETH,CVPR25)需自训, EgoDex/HOPformer仅到接触非力; HOPE/EgoPHI未开源
- 务实路径: 最稳=Ego-Video-to-SIM已有物理仿真闭环(SAPIEN impulse), 其次=EgoPressure自训

### Docs
- docs/survey_force_simulation_2026.md: 新增 #13 全开源代码扫描结论

---
