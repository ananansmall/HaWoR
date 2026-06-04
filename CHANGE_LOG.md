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
