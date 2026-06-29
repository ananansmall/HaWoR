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
