# HaWoR: 从MP4视频重建世界坐标系手部位姿 — 完整管线与函数实现详解

> 本文档详细介绍了HaWoR项目中从MP4视频重建世界坐标系手部位姿的完整管线，
> 涵盖每个函数的**整体实现思路、逐步执行过程、数学原理、边界处理、调用链**。

## 一、整体管线全景图

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    HaWoR 完整管线                                            │
│                                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐           │
│  │  MP4 视频 │───▶│ ①帧提取+检测  │───▶│ ②HaWoR位姿估计    │───▶│ ③Masked SLAM     │           │
│  │ (input)  │    │   +跟踪       │    │  (相机空间)       │    │  (相机位姿+尺度)  │           │
│  └──────────┘    └──────────────┘    └──────────────────┘    └──────────────────┘           │
│                        │                      │                       │                      │
│                        │                      │              ┌────────┴────────┐             │
│                        │                      │              ▼                 ▼             │
│                        │                      │      DROID-SLAM          Metric3D           │
│                        │                      │      (相机轨迹)          (度量深度)          │
│                        │                      │              │                 │             │
│                        │                      │              └────────┬────────┘             │
│                        │                      │                       ▼                      │
│                        │                      │               尺度恢复(scale)                │
│                        │                      │                       │                      │
│                        │                      ▼                       ▼                      │
│                        │              ┌────────────────────────────────┐                     │
│                        │              │ ④ Infilling 填补缺失帧         │                     │
│                        │              │   cam→world + Transformer      │                     │
│                        │              └────────────────────────────────┘                     │
│                        │                              │                                      │
│                        │                              ▼                                      │
│                        │                    世界空间手部参数                                  │
│                        │                   (pred_trans/rot/pose/betas)                       │
│                        │                              │                                      │
│                        │                              ▼                                      │
│                        │              ┌────────────────────────────────┐                     │
│                        │              │ ⑤ MANO正演 + 可视化渲染        │                     │
│                        │              │   → 手部Mesh → MP4输出         │                     │
│                        │              └────────────────────────────────┘                     │
│                        │                              │                                      │
│                        ▼                              ▼                                      │
│                  中间文件结构:        ┌────────────────────────┐                              │
│                  {video_name}/       │  输出: video_0.mp4     │                              │
│                  ├── extracted_images/│  (世界/相机视角渲染)    │                              │
│                  ├── tracks_0_N/     └────────────────────────┘                              │
│                  ├── cam_space/                                                               │
│                  ├── SLAM/                                                                   │
│                  └── vis_0_N/                                                                │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 函数调用链总览

```
detect_track_video(args)
├── extract_frames(video_path, output_folder)          [ffmpeg抽帧]
└── detect_track(imgfiles, thresh=0.2)                 [YOLO检测跟踪]
    └── YOLO.track()                                   [每帧推理]

hawor_motion_estimation(args, start_idx, end_idx, seq_folder)    （估计位姿）
├── load_hawor(checkpoint_path)                        [加载模型]
│   └── HAWOR.load_from_checkpoint()
├── interpolate_bboxes(bboxes)                         [bbox插值（能够框柱物体的最小框）]
├── parse_chunks(frame, boxes, min_len=1)              [分段]
├── model.inference(img_ck, boxes_ck, ...)             [模型推理]
│   ├── TrackDatasetEval(imgfiles, boxes, ...)         [数据集构建]
│   │   └── __getitem__(index)                         [单帧预处理]
│   │       └── crop() + normalize()                   [裁剪+归一化]
│   └── HAWOR.forward(batch)                           [前向传播]
│       └── forward_step(batch)
│           ├── bbox_est(center, scale, img_focal, img_center)  [CLIFF编码（裁剪原来的最小框）]
│           ├── backbone(image[:,:,:32:-32])            [ViT特征提取  全局感受野]
│           ├── st_module(feature)                      [时空注意力]
│           │   └── temporal_attention.forward(x)
│           ├── mano_head(feature)                      [MANO参数回归]
│           │   └── MANOTransformerDecoderHead.forward(x)
│           ├── motion_module(pred_pose)                [运动模块]
│           ├── mano.query(out)                         [MANO前向]
│           ├── project(j3d, ...)                       [2D投影]
│           └── get_trans(pred_cam, ...)                [弱透视→全平移]
├── run_mano / run_mano_left(...)                       [MANO正演获取顶点]
└── Renderer.render_multiple(...)                       [渲染手部mask]

hawor_slam(args, start_idx, end_idx)
├── est_calib(imgfiles)                                [估计内参]
├── run_slam(imgfiles, masks, calib)                   [Masked SLAM]
│   ├── preprocess_masks(img_folder, masks)             [mask预处理]
│   ├── image_stream(imagedir, calib, stride)           [图像流生成器]
│   └── Droid.track(t, image, ...)                     [SLAM跟踪]
├── Metric3D(imgfiles[t], calib)                       [度量深度估计]
└── est_scale_hybrid(slam_depth, pred_depth, ...)      [尺度恢复]
    └── gmof(x, sigma)                                 [Geman-McClure损失 梯度始终连续且最终趋于零，同时是光滑可微的]

hawor_infiller(args, start_idx, end_idx, frame_chunks_all)
├── load_slam_cam(fpath)                               [加载SLAM相机]
├── cam2world_convert(R_c2w, t_c2w, data_out, hand)    [cam→world]
│   ├── run_mano / run_mano_left(...)                  [MANO正演]
│   └── einsum旋转变换 + 平移变换
├── parse_chunks_hand_frame(frame)                     [缺失帧分段]
├── filling_preprocess(filling_seq)                    [Infiller预处理]
│   ├── world2canonical_convert(R, t, data, hand)      [world→正则空间]
│   │   └── (同cam2world_convert结构)
│   ├── linear_interpolation_nd(trans, valid)          [线性插值]
│   ├── slerp_interpolation_aa(rot, valid)             [球面插值]
│   └── rotmat_to_rot6d()                              [旋转→6D表示]
├── TransformerModel.forward(src, mask, data_mask, atten_mask) [Infiller推理]
│   ├── MultiHeadedAttention(hidden, mask=atten_mask)   [Masked注意力]
│   ├── FeedForward(x)                                 [前馈网络]
│   └── TransformerEncoder(src)                        [标准Transformer]
└── filling_postprocess(output, transform_w_canon)     [Infiller后处理]
    ├── custom_rot6d_to_rotmat(rot6d)                  [6D→旋转矩阵]
    └── world2canonical_convert(R_canon2w, t_canon2w, ...) [正则→世界]

run_vis2_on_video(res_dict, res_dict2, ...)
├── checkerboard_geometry(length, c1, c2, up)          [地面网格]
├── camera_marker_geometry(radius, height)             [相机标记]
├── lookat_matrix(source, target, up)                  [观察矩阵]
└── ARCTICViewer.render_seq(batch, ...)                [aitviewer渲染]
```

### 三大核心阶段概览

HaWoR 管线的精髓在于三个核心阶段，它们各自解决一个独立的关键问题，最终合力完成"从2D视频重建3D世界空间手部位姿"的目标：

```
                        三大核心阶段
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ① hawor_motion_estimation          ② hawor_slam                    │
│  ┌──────────────────────┐          ┌──────────────────────┐         │
│  │ 问题: 我看到了什么？   │          │ 问题: 我在哪？       （两个mask，一个置信度低，一个为0） │         │
│  │                      │          │                      │         │
│  │ 输入: 视频帧 + 检测框  │          │ 输入: 视频帧 + 手部mask│         │
│  │ 模型: HaWoR (ViT+时空)│          │ 模型: DROID-SLAM      │         │
│  │ 输出: 相机空间手部位姿 │          │      + Metric3D      │         │
│  │  (per-frame, cam空间) │          │ 输出: 相机c2w位姿     │         │
│  │                      │          │      + 尺度因子scale   │         │
│  └──────────┬───────────┘          └──────────┬───────────┘         │
│             │                                 │                      │
│             └─────────────┬───────────────────┘                      │
│                           ▼                                          │
│              ③ hawor_infiller                                        │
│              ┌──────────────────────────────────────┐                │
│              │ 问题: 整合到统一世界坐标系              │                │
│              │                                      │                │
│              │ 输入: cam空间手部位姿 + SLAM c2w位姿   │                │
│              │ 步骤: cam→world坐标变换               │                │
│              │       + Transformer填补缺失帧         │                │
│              │       + 双线性插值+Slerp初始化         │                │
│              │ 输出: 世界空间手部位姿                 │                │
│              │  (pred_trans/rot/pose/betas, 全帧)    │                │
│              └──────────────────────────────────────┘                │
│                           │                                          │
│                           ▼                                          │
│                  ⑤ MANO正演 → 3D Mesh → 渲染输出                     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

#### 阶段①: `hawor_motion_estimation` — 手部位姿估计（核心问题：**我看到了什么？**）

| 维度 | 说明 |
|------|------|
| **输入** | 视频帧 (`extracted_images/`) + 手部检测框 (`model_boxes.npy`, `model_tracks.npy`) |
| **模型架构** | ViT backbone (全局感受野) → 时空注意力 (ST-Module) → MANO Transformer Decoder Head |
| **输出** | 每帧的相机空间手部参数: `init_trans` (手腕平移), `init_root_orient` (手腕旋转), `init_hand_pose` (15关节旋转), `init_betas` (手型参数) |
| **输出位置** | `cam_space/{hand_id}/{start}_{end}.json` |
| **坐标系** | 相机空间 (x-right, y-down, z-forward), 原点=相机光心 |
| **关键特点** | 弱透视模型估计深度 `tz = 2*focal/bbox_size`；CLIFF编码将bbox信息注入特征；支持多帧时序推理 |

> 详细实现见 [三、阶段②：手部位姿估计](#三阶段手部位姿估计hawor-模型推理)

---

#### 阶段②: `hawor_slam` — 相机位姿估计（核心问题：**我在哪？**）

| 维度 | 说明 |
|------|------|
| **输入** | 视频帧 (`extracted_images/`) + 手部mask (`model_masks.npy`) |
| **模型架构** | DROID-SLAM (光流+BA优化) + Metric3D (度量深度估计) + 混合尺度恢复 |
| **输出** | 每帧的相机c2w位姿 (`traj`, 7维: tx,ty,tz,qx,qy,qz,qw) + 尺度因子 (`scale`) + 关键帧逆深度图 (`disps`) |
| **输出位置** | `SLAM/hawor_slam_w_scale_{start}_{end}.npz` |
| **坐标系** | SLAM世界空间 (y-down, z-forward), 原点=第0帧相机位置 |
| **关键特点** | Masked SLAM: 手部区域像素置零，避免动态物体干扰特征匹配；Metric3D提供米制尺度；Geman-McClure鲁棒优化估计scale因子；DROID-SLAM输出c2w (camera-to-world) 变换 |

> 详细实现见 [四、阶段③：Masked SLAM](#四阶段masked-slam)

---

#### 阶段③: `hawor_infiller` — 缺失帧填补与坐标变换（核心问题：**整合到统一世界坐标系**）

| 维度 | 说明 |
|------|------|
| **输入** | `cam_space/` 手部参数 + `SLAM/` 相机位姿 + `frame_chunks_all` |
| **模型架构** | Transformer Encoder (Masked Multi-Head Attention) + 坐标变换算子 |
| **输出** | `world_space_res.pth` 包含 `pred_trans` (2,T,3), `pred_rot` (2,T,3), `pred_hand_pose` (2,T,45), `pred_betas` (2,T,10), `pred_valid` (2,T) |
| **输出位置** | `world_space_res.pth` + `reconstruction/hawor_results_{start}_{end}.npz` |
| **坐标系** | 世界空间 → 经 `R_x = diag(1,-1,-1)` 变换后为渲染空间 (y-up, z-backward) |
| **关键特点** | ① cam→world 坐标变换: 将相机空间手部位姿用 SLAM c2w 矩阵变换到世界空间；② Transformer Infiller: 对缺失帧进行填补，用线性插值+Slerp作为初始值，Transformer Refine；③ 双线性插值 + Slerp球面插值为缺失帧生成先验；④ 输出覆盖全部帧 (包括原始无手帧) |

> 详细实现见 [五、阶段④：Infilling](#五阶段infilling缺失帧填补--坐标变换)

---

### 三阶段关系总结

```
时间维度上的关系:
                         hawor_motion_estimation
帧 0 ─────────────────────────────────────────────────────► 帧 N
     │                              │
     │  检测到手 (有效帧)             │  未检测到手 (缺失帧)
     │  输出cam_space参数            │  无输出
     │                              │
     ▼                              ▼
     └──────────────┬───────────────┘
                    │
        haw or_slam  │  (全帧运行，mask遮挡手部)
     ┌──────────────┼──────────────┐
     │ DROID-SLAM   │  Metric3D    │
     │ 相机轨迹     │  度量深度     │
     └──────────────┴──────────────┘
                    │
                    ▼  c2w + cam_space参数
            hawor_infiller
     ┌──────────────────────────────┐
     │ cam→world 变换 (有效帧)       │
     │ Transformer 填补 (缺失帧)     │
     └──────────────────────────────┘
                    │
                    ▼
            world_space_res.pth (全帧覆盖)
```

> **注意**: "Infiller" 的 "fill" 指填补时序缺失的 3D 手部位姿帧，**不是图像修复 (image inpainting)**。
> 它不会生成"把手从图像中去掉再填入背景"的图片。

---

## 二、阶段①：帧提取 + 手部检测与跟踪

### 函数1: `extract_frames(video_path, output_folder)`

**文件**: `scripts/scripts_test_video/detect_track_video.py:13-25`

**功能**: 将MP4视频按30fps抽取为JPG图片序列

**参数**:

- `video_path` (str): 输入MP4视频的文件路径
- `output_folder` (str): 输出图片的保存目录

**返回值**: 无（直接写入磁盘）

**整体实现思路**: 调用系统级工具ffmpeg进行视频解码和帧抽取。ffmpeg是C语言编写的高性能多媒体处理库，比Python层面的视频读取（如OpenCV的VideoCapture）更稳定、更快，且能精确控制输出帧率。通过`subprocess.run`以子进程方式调用，避免Python GIL锁的限制。

**完整实现过程**:

```python
def extract_frames(video_path, output_folder):
    # Step 1: 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Step 2: 构建ffmpeg命令
    command = [
        'ffmpeg',               # 调用系统安装的ffmpeg
        '-i', video_path,       # 指定输入MP4文件
        '-vf', 'fps=30',        # 视频滤镜：固定30fps采样率
        '-start_number', '0',   # 输出文件名从0开始编号
        os.path.join(output_folder, '%04d.jpg')  # 输出格式: 0000.jpg, 0001.jpg, ...
    ]

    # Step 3: 执行命令，check=True确保ffmpeg失败时抛出CalledProcessError
    subprocess.run(command, check=True)
```

**关键细节**:

- 使用 `subprocess.run` 调用系统 ffmpeg，不是 Python 库，因此需要系统预装 ffmpeg
- `check=True` 确保ffmpeg失败时抛出 `CalledProcessError` 异常
- 输出文件名格式 `%04d.jpg` → 4位零填充编号（0000.jpg, 0001.jpg, ...）
- 如果 `output_folder` 已存在则不重复创建（但已有文件不会被覆盖）
- `-vf fps=30` 会将任意帧率的视频统一重采样到30fps，确保后续处理的一致性
- **边界情况**: 如果视频本身帧率低于30fps，ffmpeg会重复帧来达到30fps；如果高于30fps，会跳帧

---

### 函数2: `detect_track_video(args)`

**文件**: `scripts/scripts_test_video/detect_track_video.py:28-62`

**功能**: 整合帧提取和检测跟踪，返回视频信息

**参数**:

- `args`: EasyDict对象，需包含 `video_path` (str) 和可选的 `img_focal` (float)

**返回值**:

- `start_idx` (int): 起始帧索引，始终为0
- `end_idx` (int): 结束帧索引，等于总帧数
- `seq_folder` (str): 视频对应的输出目录路径
- `imgfiles` (list): 所有帧图片的路径列表（自然排序）

**整体实现思路**: 这是阶段①的入口函数，负责：1）解析视频路径并构建输出目录结构；2）利用缓存机制避免重复计算（帧提取和跟踪结果分别缓存）；3）调用YOLO检测器进行手部检测和跨帧跟踪；4）将结果保存为numpy文件供后续阶段使用。

**完整实现过程**:

```python
def detect_track_video(args):
    # Step 1: 解析视频路径
    file = args.video_path                              # 如: './example/video_0.mp4'
    root = os.path.dirname(file)                        # 如: './example'
    seq = os.path.basename(file).split('.')[0]          # 如: 'video_0'（去掉.mp4后缀）

    # Step 2: 构建输出目录结构
    seq_folder = f'{root}/{seq}'                        # 如: './example/video_0'
    img_folder = f'{seq_folder}/extracted_images'       # 如: './example/video_0/extracted_images'
    os.makedirs(seq_folder, exist_ok=True)              # 创建视频目录
    os.makedirs(img_folder, exist_ok=True)              # 创建图片目录

    # Step 3: 帧提取（带缓存机制）
    imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))   # 检查已有图片
    if len(imgfiles) > 0:
        print("Skip extracting frames")                 # 已有图片则跳过
    else:
        _ = extract_frames(file, img_folder)            # 否则调用ffmpeg抽帧
    imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))   # 重新获取图片列表

    # Step 4: 检测+跟踪（带缓存机制）
    start_idx = 0
    end_idx = len(imgfiles)

    # 如果跟踪结果已存在，直接返回
    if os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy'):
        print(f"skip track for {start_idx}_{end_idx}")
        return start_idx, end_idx, seq_folder, imgfiles

    # 否则运行检测跟踪
    os.makedirs(f"{seq_folder}/tracks_{start_idx}_{end_idx}", exist_ok=True)
    boxes_, tracks_ = detect_track(imgfiles, thresh=0.2)  # YOLO检测+跟踪
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy', boxes_)
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy', tracks_)

    return start_idx, end_idx, seq_folder, imgfiles
```

**关键细节**:

- **二级缓存机制**: 第一级检查图片是否已存在，第二级检查跟踪结果是否已存在
- `natsorted` 自然排序确保 `0009.jpg` 在 `0010.jpg` 之前（普通排序会出错）
- `start_idx=0, end_idx=len(imgfiles)` 表示处理全部帧
- 检测阈值 `thresh=0.2` 较低，宁可多检不可漏检
- **缓存文件命名**: `tracks_{start_idx}_{end_idx}` 允许对视频的不同片段分别处理

---

### 函数3: `detect_track(imgfiles, thresh=0.5)`

**文件**: `lib/pipeline/tools.py:25-79`

**功能**: 使用YOLO对每帧进行手部检测和跨帧跟踪

**参数**:

- `imgfiles` (list): 图片路径列表
- `thresh` (float): 检测置信度阈值，默认0.5（实际调用时传入0.2）

**返回值**:

- `boxes_` (np.ndarray, dtype=object): 每帧的检测框（当前未使用，为空列表）
- `tracks` (np.ndarray, dtype=object): 字典形式，key=track_id, value=[{frame, det, det_box, det_handedness}, ...]

**整体实现思路**: 逐帧将图像送入YOLOv8手部检测器，同时启用跟踪模式（`persist=True`）。YOLO的跟踪模式内部维护一个基于ByteTrack的状态机，为同一只手在不同帧分配相同的track_id。每帧只保留第一个检测到的左手和右手（去重），避免同一只手被多次检测。最终将所有检测记录按track_id组织成字典。

**完整实现过程**:

```python
def detect_track(imgfiles, thresh=0.5):
    # Step 1: 加载YOLO手部检测模型
    hand_det_model = YOLO('./weights/external/detector.pt')  # YOLOv8手部检测器

    boxes_ = []
    tracks = {}

    # Step 2: 逐帧处理
    for t, imgpath in enumerate(tqdm(imgfiles)):
        img_cv2 = cv2.imread(imgpath)

        # Step 3: YOLO检测+跟踪（混合精度推理）
        with torch.no_grad():
            with autocast():  # FP16混合精度加速
                results = hand_det_model.track(
                    img_cv2,
                    conf=thresh,        # 置信度阈值
                    persist=True,       # 启用跨帧跟踪模式，维护跟踪状态
                    verbose=False       # 不打印检测信息
                )

                # Step 4: 解析检测结果
                boxes = results[0].boxes.xyxy.cpu().numpy()         # (N, 4) [x1,y1,x2,y2]
                confs = results[0].boxes.conf.cpu().numpy()         # (N,) 置信度
                handedness = results[0].boxes.cls.cpu().numpy()     # (N,) 0=左手, 1=右手
                if not results[0].boxes.id is None:
                    track_id = results[0].boxes.id.cpu().numpy()    # (N,) 跟踪ID
                else:
                    track_id = [-1] * len(boxes)                    # 无跟踪ID时全部设为-1

                # Step 5: 拼接bbox和置信度
                boxes = np.hstack([boxes, confs[:, None]])  # (N, 5) [x1,y1,x2,y2,conf]

                # Step 6: 每帧只保留第一个检测到的左手和右手
                find_right = False
                find_left = False
                for idx, box in enumerate(boxes):
                    # 处理无track_id的情况
                    if track_id[idx] == -1:
                        if handedness[[idx]] > 0:
                            id = int(10000)   # 右手无ID → 临时ID 10000
                        else:
                            id = int(5000)    # 左手无ID → 临时ID 5000
                    else:
                        id = track_id[idx]    # 使用YOLO分配的跟踪ID

                    # 构建当前检测的记录
                    subj = dict()
                    subj['frame'] = t
                    subj['det'] = True
                    subj['det_box'] = boxes[[idx]]           # (1, 5) [x1,y1,x2,y2,conf]
                    subj['det_handedness'] = handedness[[idx]]  # 0.0 or 1.0

                    # 去重：每帧最多一个左手+一个右手
                    if (not find_right and handedness[[idx]] > 0) or \
                       (not find_left and handedness[[idx]] == 0):
                        if id in tracks:
                            tracks[id].append(subj)
                        else:
                            tracks[id] = [subj]

                        if handedness[[idx]] > 0:
                            find_right = True
                        elif handedness[[idx]] == 0:
                            find_left = True

    # Step 7: 转为numpy数组
    tracks = np.array(tracks, dtype=object)
    boxes_ = np.array(boxes_, dtype=object)

    return boxes_, tracks
```

**关键细节**:

- `persist=True` 启用YOLO的跨帧跟踪模式，内部维护一个跟踪状态机，使同一只手在不同帧获得相同ID
- `find_right/find_left` 标志确保每帧每种手只保留一个检测，避免重复
- 无track_id的检测分配临时ID（左手5000，右手10000），这些临时ID在后续左右手分类时会被合并
- `autocast()` 使用FP16混合精度加速推理，在GPU上可提速约2倍
- `handedness` 由YOLO分类头输出：0=左手，1=右手
- **边界情况**: 如果某帧没有检测到手，`find_right`和`find_left`都为False，该帧不会有任何记录添加到tracks中

---

### 函数4: `parse_chunks(frame, boxes, min_len=16)`

**文件**: `lib/pipeline/tools.py:82-109`

**功能**: 将连续帧按断点分段（手消失时断开）

**参数**:

- `frame` (np.ndarray): 帧号数组，如 [0,1,2,5,6,7]
- `boxes` (np.ndarray): 对应的bbox数组
- `min_len` (int): 最小分段长度，短于此长度的分段被丢弃

**返回值**:

- `frame_chunks` (list): 分段后的帧号列表，如 [[0,1,2], [5,6,7]]
- `boxes_chunks` (list): 分段后的bbox列表

**完整实现过程**:

```python
def parse_chunks(frame, boxes, min_len=16):
    frame_chunks = []
    boxes_chunks = []

    # Step 1: 计算相邻帧号之差
    step = frame[1:] - frame[:-1]       # 如 [0,1,2,5,6,7] → [1,1,3,1,1]
    step = np.concatenate([[0], step])   # 前面补0 → [0,1,1,3,1,1]

    # Step 2: 找到断点位置（差值不为1的位置）
    breaks = np.where(step != 1)[0]     # 如 [0, 3]（0是起始，3是断点）

    # Step 3: 按断点分段
    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]       # 当前段
        b_chunk = boxes[start:bk]
        start = bk                       # 更新起始位置

        # 过短的分段丢弃（HaWoR模型需要至少16帧的序列）
        if len(f_chunk) >= min_len:
            frame_chunks.append(f_chunk)
            boxes_chunks.append(b_chunk)

        # 处理最后一段
        if bk == breaks[-1]:
            f_chunk = frame[bk:]
            b_chunk = boxes[bk:]
            if len(f_chunk) >= min_len:
                frame_chunks.append(f_chunk)
                boxes_chunks.append(b_chunk)

    return frame_chunks, boxes_chunks
```

**关键细节**:

- 断点检测的核心：`frame[1:] - frame[:-1]` 不为1的位置就是手消失/重现的断点
- `min_len=16`：HaWoR模型以16帧为窗口推理，短于16帧的分段无法处理
- 在 `hawor_motion_estimation` 中调用时 `min_len=1`（不丢弃短段），因为后续有bbox插值

---

### 函数5: `parse_chunks_hand_frame(frame)`

**文件**: `lib/pipeline/tools.py:145-167`

**功能**: 仅按帧号分段（用于Infilling阶段找出缺失帧的连续段）

**参数**:

- `frame` (np.ndarray): 缺失帧的帧号数组

**返回值**:

- `frame_chunks` (list): 分段后的帧号列表

**完整实现过程**:

```python
def parse_chunks_hand_frame(frame):
    frame_chunks = []
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]
        start = bk
        if len(f_chunk) > 0:            # 只要非空就保留（无min_len限制）
            frame_chunks.append(f_chunk)

        if bk == breaks[-1]:
            f_chunk = frame[bk:]
            if len(f_chunk) > 0:
                frame_chunks.append(f_chunk)

    return frame_chunks
```

**与parse_chunks的区别**: 无min_len限制，无boxes返回，专用于Infilling阶段

---

## 三、阶段②：手部位姿估计（HaWoR 模型推理）

### 函数6: `load_hawor(checkpoint_path)`

**文件**: `scripts/scripts_test_video/hawor_video.py:23-37`

**功能**: 加载HaWoR模型和配置

**参数**:

- `checkpoint_path` (str): 模型权重文件路径，如 `weights/hawor/checkpoints/hawor.ckpt`

**返回值**:

- `model` (HAWOR): 加载好权重的HaWoR模型实例
- `model_cfg` (CfgNode): 模型配置对象

**完整实现过程**:

```python
def load_hawor(checkpoint_path):
    from pathlib import Path
    from hawor.configs import get_config

    # Step 1: 推导配置文件路径
    # checkpoint_path: weights/hawor/checkpoints/hawor.ckpt
    # model_cfg路径:   weights/hawor/model_config.yaml (上两级目录)
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')

    # Step 2: 加载配置，update_cachedir=True将MANO模型路径更新为项目内_DATA/目录
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Step 3: ViT backbone需要特殊的bbox裁剪参数
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()  # 解冻配置以允许修改
        assert model_cfg.MODEL.IMAGE_SIZE == 256  # ViT输入必须是256×256
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]   # 高192, 宽256
        model_cfg.freeze()   # 重新冻结配置

    # Step 4: 加载模型权重
    # strict=False 允许权重文件中有多余的key（如训练时的优化器状态）
    model = HAWOR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg
```

**关键细节**:

- `BBOX_SHAPE = [192, 256]`：ViT输入256×256，但实际手部区域192×256，左右各32像素是padding。在forward_step中会裁掉：`image[:, :, :, 32:-32]`
- `update_cachedir=True` 将MANO模型路径从默认的 `~/.cache/` 更新为项目内的 `_DATA/` 目录
- `strict=False` 是因为checkpoint可能包含训练时的额外状态（如优化器参数），这些在推理时不需要

---

### 函数7: `hawor_motion_estimation(args, start_idx, end_idx, seq_folder)`

**文件**: `scripts/scripts_test_video/hawor_video.py:41-220`

**功能**: 完整的手部位姿估计流程（含mask生成）

**参数**:

- `args`: EasyDict对象，需包含 `checkpoint`, `video_path`, `img_focal`
- `start_idx` (int): 起始帧索引
- `end_idx` (int): 结束帧索引
- `seq_folder` (str): 视频输出目录

**返回值**:

- `frame_chunks_all` (dict): {0: 左手分段列表, 1: 右手分段列表}
- `img_focal` (float): 使用的焦距值

**整体实现思路**: 这是阶段②的核心函数，完成以下任务：1）加载HaWoR模型；2）对YOLO跟踪结果进行左右手分类（基于handedness投票）；3）对缺失帧的bbox进行线性插值；4）按连续帧分段，每段送入HaWoR模型推理；5）对左手的输出进行翻转修正；6）保存相机空间结果为JSON；7）运行MANO正演获取3D顶点，用PyTorch3D渲染手部mask（供SLAM使用）。

**完整实现过程**（逐步详解）:

```python
def hawor_motion_estimation(args, start_idx, end_idx, seq_folder):
    # ===== 1. 加载模型 =====
    model, model_cfg = load_hawor(args.checkpoint)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # ===== 2. 加载帧列表和跟踪结果 =====
    file = args.video_path
    video_root = os.path.dirname(file)
    video = os.path.basename(file).split('.')[0]
    img_folder = f"{video_root}/{video}/extracted_images"
    imgfiles = np.array(natsorted(glob(f'{img_folder}/*.jpg')))

    tracks = np.load(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy',
                     allow_pickle=True).item()

    # ===== 3. 焦距获取（三级优先级）=====
    img_focal = args.img_focal                    # 优先级1: 用户指定
    if img_focal is None:
        try:
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'r') as file:
                img_focal = float(file.read())     # 优先级2: 读缓存
        except:
            img_focal = 600                        # 优先级3: 默认600像素
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'w') as file:
                file.write(str(img_focal))         # 缓存默认值

    # ===== 4. 缓存检查 =====
    if os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy'):
        frame_chunks_all = joblib.load(...)
        return frame_chunks_all, img_focal

    # ===== 5. 左右手分类（handedness投票）=====
    tid = np.array([tr for tr in tracks])  # 所有track_id
    left_trk, right_trk = [], []

    for k, idx in enumerate(tid):
        trk = tracks[idx]
        # 获取该track中所有有效检测的handedness
        valid = np.array([t['det'] for t in trk])
        is_right = np.concatenate([t['det_handedness'] for t in trk])[valid]

        # 投票：多数帧为右手→右手轨迹，否则→左手轨迹
        if is_right.sum() / len(is_right) < 0.5:
            left_trk.extend(trk)
        else:
            right_trk.extend(trk)

    # 按帧号排序
    left_trk = sorted(left_trk, key=lambda x: x['frame'])
    right_trk = sorted(right_trk, key=lambda x: x['frame'])
    final_tracks = {0: left_trk, 1: right_trk}
    tid = [0, 1]  # 统一ID：0=左手, 1=右手

    # ===== 6. 初始化渲染器和mask =====
    img = cv2.imread(imgfiles[0])
    img_center = [img.shape[1] / 2, img.shape[0] / 2]  # [W/2, H/2]
    H, W = img.shape[:2]
    model_masks = np.zeros((len(imgfiles), H, W))  # (T, H, W) 全零mask

    # 创建PyTorch3D渲染器（用于生成手部mask）
    renderer = Renderer(W, H, img_focal, 'cuda', bin_size=128, max_faces_per_bin=20000)

    # MANO面片 + 14个指尖补充面片
    faces = get_mano_faces()  # (1538, 3) 标准MANO面片
    faces_new = np.array([[92,38,234], [234,38,239], ...])  # 14个指尖面片
    faces_right = np.concatenate([faces, faces_new], axis=0)  # (1552, 3)
    faces_left = faces_right[:, [0,2,1]]  # 左手：翻转面片绕序（法线方向取反）

    # ===== 7. 对每个tracklet分段推理 =====
    frame_chunks_all = defaultdict(list)
    for idx in tid:  # 0=左手, 1=右手
        trk = final_tracks[idx]
        valid = np.array([t['det'] for t in trk])

        if valid.sum() < 2:
            continue  # 有效检测少于2帧则跳过

        # 7a. 提取bbox并插值缺失帧
        boxes = np.concatenate([t['det_box'] for t in trk])
        non_zero_indices = np.where(np.any(boxes != 0, axis=1))[0]
        first_non_zero = non_zero_indices[0]
        last_non_zero = non_zero_indices[-1]

        # 对首末有效帧之间的缺失bbox进行线性插值
        boxes[first_non_zero:last_non_zero+1] = interpolate_bboxes(
            boxes[first_non_zero:last_non_zero+1])
        valid[first_non_zero:last_non_zero+1] = True  # 插值帧标记为有效

        # 7b. 分段
        boxes = boxes[first_non_zero:last_non_zero+1]
        frame = np.array([t['frame'] for t in trk])[valid]
        frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=1)
        frame_chunks_all[idx] = frame_chunks

        # 7c. 对每段推理
        for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
            do_flip = (idx == 0)  # 左手需要翻转
            img_ck = imgfiles[frame_ck]

            # 调用HaWoR模型推理
            results = model.inference(img_ck, boxes_ck,
                                     img_focal=img_focal, img_center=img_center,
                                     do_flip=do_flip)

            # 7d. 整理输出为标准格式
            data_out = {
                "init_root_orient": results["pred_rotmat"][None, :, 0],   # (1,T,3,3) 根旋转
                "init_hand_pose":   results["pred_rotmat"][None, :, 1:],  # (1,T,15,3,3) 关节旋转
                "init_trans":       results["pred_trans"][None, :, 0],    # (1,T,3) 平移
                "init_betas":       results["pred_shape"][None, :]        # (1,T,10) shape
            }

            # 7e. 左手翻转处理
            # 因为模型输入时对左手图像做了水平翻转，输出的旋转需要对应修正
            init_root = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
            init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
            if do_flip:
                init_root[..., 1] *= -1       # y轴角度取反
                init_root[..., 2] *= -1       # z轴角度取反
                init_hand_pose[..., 1] *= -1
                init_hand_pose[..., 2] *= -1
            data_out["init_root_orient"] = angle_axis_to_rotation_matrix(init_root)
            data_out["init_hand_pose"] = angle_axis_to_rotation_matrix(init_hand_pose)

            # 7f. 保存相机空间结果为JSON
            pred_path = os.path.join(seq_folder, 'cam_space', str(idx),
                                    f"{frame_ck[0]}_{frame_ck[-1]}.json")
            os.makedirs(os.path.dirname(pred_path), exist_ok=True)
            with open(pred_path, "w") as f:
                json.dump({k: v.tolist() for k, v in data_out.items()}, f, indent=1)

            # 7g. 生成手部mask（用于SLAM遮挡）
            # 将旋转转回axis-angle用于MANO前向传播
            data_out["init_root_orient"] = rotation_matrix_to_angle_axis(
                data_out["init_root_orient"])
            data_out["init_hand_pose"] = rotation_matrix_to_angle_axis(
                data_out["init_hand_pose"])

            # 运行MANO获取3D顶点
            if do_flip:
                outputs = run_mano_left(data_out["init_trans"],
                    data_out["init_root_orient"], data_out["init_hand_pose"],
                    betas=data_out["init_betas"])
            else:
                outputs = run_mano(data_out["init_trans"],
                    data_out["init_root_orient"], data_out["init_hand_pose"],
                    betas=data_out["init_betas"])

            vertices = outputs["vertices"][0].cpu()  # (T, 778, 3)

            # 逐帧渲染手部mask
            for img_i, _ in enumerate(img_ck):
                faces_t = torch.from_numpy(faces_left if do_flip else faces_right).cuda()
                cam_R = torch.eye(3).unsqueeze(0).cuda()  # 单位旋转（相机空间）
                cam_T = torch.zeros(1, 3).cuda()           # 零平移
                cameras, lights = renderer.create_camera_from_cv(cam_R, cam_T)
                verts_color = torch.tensor([0, 0, 255, 255]) / 255  # 蓝色
                vertices_i = vertices[[img_i]]
                rend, mask = renderer.render_multiple(
                    vertices_i.unsqueeze(0).cuda(), faces_t,
                    verts_color.unsqueeze(0).cuda(), cameras, lights)
                model_masks[frame_ck[img_i]] += mask  # 累加（可能双手重叠）

    # ===== 8. 二值化并保存 =====
    model_masks = model_masks > 0  # 二值化：任何手部像素都为True
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_masks.npy', model_masks)
    joblib.dump(frame_chunks_all,
        f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy')
    return frame_chunks_all, img_focal
```

**左手翻转的数学原理**:
当图像水平翻转时，3D空间中的变换等价于绕x轴旋转180°（即y和z取反）。因此axis-angle表示中y和z分量取反即可恢复正确的旋转。

**mask生成流程详解**:

1. 将HaWoR输出的旋转参数转为axis-angle → 运行MANO正演获取3D顶点
2. 在相机空间（cam_R=I, cam_T=0）下用PyTorch3D渲染手部mesh
3. 渲染结果的alpha通道即为手部mask
4. 双手mask累加后二值化，得到最终的手部遮挡区域

**为什么需要mask**: SLAM（DROID-SLAM）通过特征点匹配估计相机运动。手部是移动物体，其上的特征点会干扰SLAM的相机运动估计。通过mask将手部区域遮挡，SLAM只利用背景特征点，从而获得更准确的相机轨迹。

---

### 函数8: `HAWOR.inference(imgfiles, boxes, img_focal, img_center, device, do_flip)`

**文件**: `lib/models/hawor.py:379-439`

**功能**: HaWoR模型推理入口，16帧滑动窗口

**参数**:

- `imgfiles` (list): 图片路径列表
- `boxes` (np.ndarray): 检测框 (N, 5) [x1,y1,x2,y2,conf]
- `img_focal` (float): 焦距
- `img_center` (list): 图像中心 [cx, cy]
- `device` (str): 设备，默认'cuda'
- `do_flip` (bool): 是否翻转（左手）

**返回值**:

- `results` (dict): 包含 pred_cam, pred_pose, pred_shape, pred_rotmat, pred_trans, img_focal, img_center

**完整实现过程**:

```python
def inference(self, imgfiles, boxes, img_focal, img_center, device='cuda', do_flip=False):
    # Step 1: 构建评估数据集
    db = TrackDatasetEval(imgfiles, boxes, img_focal=img_focal,
                          img_center=img_center, normalization=True,
                          dilate=1.2, do_flip=do_flip)
    # dilate=1.2: bbox扩大20%，确保手部周围有足够上下文

    # Step 2: 16帧滑动窗口推理
    pred_cam, pred_pose, pred_shape, pred_rotmat, pred_trans = [], [], [], [], []
    items = []

    for i in tqdm(range(len(db))):
        item = db[i]           # 读取单帧：crop+normalize
        items.append(item)

        # 最后一帧不足16帧时padding
        if i == len(db) - 1 and len(db) % 16 != 0:
            pad = 16 - len(db) % 16
            for _ in range(pad):
                items.append(item)  # 重复最后一帧作为padding

        # 积累到16帧时进行推理
        if len(items) < 16:
            continue
        elif len(items) == 16:
            batch = default_collate(items)  # 整理为batch字典
            items = []
        else:
            raise NotImplementedError

        # Step 3: 模型前向传播
        with torch.no_grad():
            batch = {k: v.to(device).unsqueeze(0) for k, v in batch.items()
                     if type(v) == torch.Tensor}
            # unsqueeze(0) 添加batch维度: (16,...) → (1,16,...)
            output = self.forward(batch)
            out = output['out']

        # Step 4: 裁掉padding帧的结果
        if i == len(db) - 1 and len(db) % 16 != 0:
            out = {k: v[:len(db) % 16] for k, v in out.items()}
            # 只保留真实帧的结果，丢弃padding帧

        # Step 5: 收集结果
        pred_cam.append(out['pred_cam'].cpu())
        pred_pose.append(out['pred_pose'].cpu())
        pred_shape.append(out['pred_shape'].cpu())
        pred_rotmat.append(out['pred_rotmat'].cpu())
        pred_trans.append(out['trans_full'].cpu())

    # Step 6: 拼接所有窗口结果
    results = {
        'pred_cam': torch.cat(pred_cam),       # (T, 3) 弱透视参数 [s, tx, ty]
        'pred_pose': torch.cat(pred_pose),     # (T, 96) rot6d (16关节×6)
        'pred_shape': torch.cat(pred_shape),   # (T, 10) MANO betas
        'pred_rotmat': torch.cat(pred_rotmat), # (T, 16, 3, 3) 旋转矩阵
        'pred_trans': torch.cat(pred_trans),   # (T, 1, 3) 全3D平移
        'img_focal': img_focal,
        'img_center': img_center,
    }
    return results
```

**滑动窗口机制详解**:

- 每次积累16帧数据，组成一个batch送入模型
- 16帧是HaWoR的时序窗口大小，模型内部利用时序注意力建模帧间关系
- 最后不足16帧时，用最后一帧重复填充，推理后裁掉填充部分
- 这种非重叠窗口方式意味着相邻窗口之间没有信息交互

---

### 函数9: `HAWOR.forward_step(batch, train=False)`

**文件**: `lib/models/hawor.py:145-227`

**功能**: HaWoR模型前向传播的核心实现——从图像到3D手部参数的完整推理链

**参数**:

- `batch` (Dict): 包含 img, center, scale, img_focal, img_center 等张量
- `train` (bool): 是否训练模式

**返回值**:

- `output` (Dict): 包含 pred_mano_params, pred_keypoints_3d, pred_keypoints_2d, out

**整体实现思路**: 这是HaWoR模型的核心推理函数，实现了从图像到3D手部参数的完整推理链。整个流程分为7步：1）CLIFF bbox编码（让模型感知手在图像中的位置和大小）；2）ViT backbone提取图像特征；3）Space-Time时序注意力（在特征层面建模帧间运动关系）；4）MANO Transformer Decoder Head（从特征预测MANO参数）；5）Motion Module（在pose层面做时序平滑）；6）左手翻转处理；7）计算全3D平移和2D投影。

**完整实现过程**（7步详解）:

```python
def forward_step(self, batch, train=False):
    # ===== 准备输入 =====
    image = batch['img'].flatten(0, 1)       # (B*16, 3, 256, 256)
    center = batch['center'].flatten(0, 1)   # (B*16, 2) 裁剪区域中心
    scale = batch['scale'].flatten(0, 1)     # (B*16,) bbox缩放因子
    img_focal = batch['img_focal'].flatten(0, 1)  # (B*16,)
    img_center = batch['img_center'].flatten(0, 1)  # (B*16, 2) 图像中心
    bn = len(image)  # B*16

    # ===== Step 1: CLIFF bbox特征编码 =====
    # 将bbox在图像中的位置和大小编码为3维特征向量
    bbox_info = self.bbox_est(center, scale, img_focal, img_center)
    # bbox_info: (B*16, 3) = [normalized_cx, normalized_cy, normalized_b]
    # 这让模型知道"手在图像的哪个位置、有多大"，从而更准确估计3D位姿

    # ===== Step 2: ViT Backbone特征提取 =====
    feature = self.backbone(image[:, :, :, 32:-32])  # 裁掉左右32px padding
    # 输入: (B*16, 3, 256, 192) → 输出: (B*16, 1280, 16, 12) 特征图
    # 裁剪原因: BBOX_SHAPE=[192,256]，实际手部区域192宽，左右各32px是padding

    # ===== Step 3: Space-Time Temporal Attention =====
    # 在空间特征图上做时序注意力，建模帧间运动关系
    if self.st_module is not None:
        # 将bbox_info广播到特征图每个空间位置
        bb = einops.repeat(bbox_info, 'b c -> b c h w', h=16, w=12)
        # bb: (B*16, 3, 16, 12)

        # 拼接bbox特征到图像特征
        feature = torch.cat([feature, bb], dim=1)  # (B*16, 1283, 16, 12)

        # 重排为 (B*H*W, T, C) 做时序注意力
        # 每个空间位置独立做时序注意力，16帧之间交换信息
        feature = einops.rearrange(feature, '(b t) c h w -> (b h w) t c', t=16)
        # (B*16*12, 16, 1283) → 每个空间位置看16帧

        feature = self.st_module(feature)  # 6层Transformer时序注意力
        # 输出: (B*16*12, 16, 1280) → 投影回1280维

        # 重排回空间特征图格式
        feature = einops.rearrange(feature, '(b h w) t c -> (b t) c h w', h=16, w=12)
        # (B*16, 1280, 16, 12)

    # ===== Step 4: MANO Transformer Decoder Head =====
    # 从特征图预测MANO参数
    pred_pose, pred_shape, pred_cam = self.mano_head(feature)
    # pred_pose:  (B*16, 96)  16关节 × 6D旋转表示
    # pred_shape: (B*16, 10)  MANO shape参数(betas)
    # pred_cam:   (B*16, 3)   弱透视相机参数 [s, tx, ty]

    # 转为旋转矩阵（用于后续计算）
    pred_rotmat_0 = rot6d_to_rotmat(pred_pose).reshape(-1, 16, 3, 3)

    # ===== Step 5: Motion Module =====
    # 对pose序列做额外的时序平滑
    if self.motion_module is not None:
        bb = einops.rearrange(bbox_info, '(b t) c -> b t c', t=16)
        pred_pose = einops.rearrange(pred_pose, '(b t) c -> b t c', t=16)
        pred_pose = torch.cat([pred_pose, bb], dim=2)  # 拼接bbox信息
        # (B, 16, 96+3=99)

        pred_pose = self.motion_module(pred_pose)       # 6层Transformer时序平滑
        # (B, 16, 96) → 输出维度=pose维度

        pred_pose = einops.rearrange(pred_pose, 'b t c -> (b t) c')
        # (B*16, 96)

    # ===== Step 6: 翻转处理（左手）=====
    if 'do_flip' in batch:
        pred_cam[..., 1] *= -1                          # ty取反（y轴翻转）
        center[..., 0] = img_center[..., 0]*2 - center[..., 0] - 1  # cx翻转

    # ===== Step 7: 计算全平移 + 投影 =====
    out = {}
    out['pred_cam'] = pred_cam
    out['pred_pose'] = pred_pose
    out['pred_shape'] = pred_shape
    out['pred_rotmat'] = rot6d_to_rotmat(pred_pose).reshape(-1, 16, 3, 3)
    out['pred_rotmat_0'] = pred_rotmat_0

    # MANO前向传播获取3D关节
    s_out = self.mano.query(out)
    j3d = s_out.joints  # (B*16, 21, 3)

    # 投影到2D（用于训练时的2D损失）
    j2d = self.project(j3d, pred_cam, center, scale, img_focal, img_center)
    j2d = j2d / self.crop_size - 0.5  # 归一化到 [-0.5, 0.5]

    # 弱透视→全3D平移
    trans_full = self.get_trans(pred_cam, center, scale, img_focal, img_center)
    out['trans_full'] = trans_full

    output = {
        'pred_mano_params': {
            'global_orient': out['pred_rotmat'][:, :1].clone(),   # (B*16, 1, 3, 3)
            'hand_pose': out['pred_rotmat'][:, 1:].clone(),       # (B*16, 15, 3, 3)
            'betas': out['pred_shape'].clone(),                    # (B*16, 10)
        },
        'pred_keypoints_3d': j3d.clone(),    # (B*16, 21, 3)
        'pred_keypoints_2d': j2d.clone(),    # (B*16, 21, 2)
        'out': out,
    }

    return output
```

**模型架构总结**:

```
图像 → ViT Backbone → Space-Time Attention → MANO Head → Motion Module → 输出
         (1280维)       (时序建模)           (参数回归)    (时序平滑)
              ↑               ↑                    ↑
         裁掉padding    拼接bbox_info       rot6d→旋转矩阵
```

**Space-Time Module vs Motion Module的区别**:

- **Space-Time Module**: 在特征层面做时序建模，输入是1280维图像特征+3维bbox信息，输出1280维特征。使用残差连接（`residual=True`），即输出=输入+变换。每个空间位置(16×12=192个)独立做时序注意力。
- **Motion Module**: 在pose层面做时序平滑，输入是96维pose+3维bbox信息，输出96维pose。不使用残差连接（`residual=False`），即输出=变换结果。直接对pose序列做全局时序注意力。

---

### 函数10: `MANOTransformerDecoderHead.forward(x)`

**文件**: `lib/models/modules.py:58-78`

**功能**: 从ViT特征图预测MANO参数（pose, shape, camera）**形状参数**、**姿态参数**和**全局变换参数**

**参数**:

- `x` (Tensor): 特征图 (B, 1280, H, W)

**返回值**:

- `pred_pose` (Tensor): (B, 96) 16关节×6D旋转
- `pred_shape` (Tensor): (B, 10) MANO betas
- `pred_cam` (Tensor): (B, 3) 弱透视相机参数

**完整实现过程**:

```python
def forward(self, x, **kwargs):
    batch_size = x.shape[0]

    # Step 1: 将通道优先的特征图转为token序列
    x = einops.rearrange(x, 'b c h w -> b (h w) c')
    # (B, 1280, 16, 12) → (B, 192, 1280)
    # 每个空间位置变成一个token，共16×12=192个token

    # Step 2: 准备初始参数（从训练集统计的均值）
    init_hand_pose = self.init_hand_pose.expand(batch_size, -1)  # (B, 96)
    init_betas = self.init_betas.expand(batch_size, -1)          # (B, 10)
    init_cam = self.init_cam.expand(batch_size, -1)              # (B, 3)

    # Step 3: Transformer解码
    # 使用可学习的query token通过交叉注意力从特征图中提取信息
    token = torch.zeros(batch_size, 1, 1).to(x.device)  # (B, 1, 1) query token
    token_out = self.transformer(token, context=x)        # 交叉注意力
    token_out = token_out.squeeze(1)                      # (B, 1024)

    # Step 4: 从token输出预测MANO参数（残差方式）
    pred_pose = self.decpose(token_out) + init_hand_pose   # (B, 96) + 残差
    pred_shape = self.decshape(token_out) + init_betas     # (B, 10) + 残差
    pred_cam = self.deccam(token_out) + init_cam           # (B, 3) + 残差

    return pred_pose, pred_shape, pred_cam
```

**设计要点**:

- 使用**交叉注意力**而非自注意力：query token从192个图像token中聚合信息
- **残差预测**：预测的是相对于均值参数的偏移量，加速收敛
- `init_hand_pose/init_betas/init_cam` 从 `mean_params.npy` 加载，是训练集的统计均值

---

### 函数11: `temporal_attention.forward(x)`

**文件**: `lib/models/modules.py:99-113`

**功能**: 时序注意力模块，用于Space-Time Module和Motion Module

**参数**:

- `x` (Tensor): (T, B, C) 时序特征序列

**返回值**:

- `x` (Tensor): (B, T, C_out) 时序建模后的特征

**完整实现过程**:

```python
def forward(self, x):
    # 输入: (T, B, C_in) — 时序优先格式

    # Step 1: 维度转换
    x = x.permute(1,0,2)  # (T,B,C) → (B,T,C) batch优先

    # Step 2: 线性投影到隐藏维度
    h = self.l1(x)         # (B, T, hdim)

    # Step 3: 添加位置编码
    h = self.pos_embedding(h)  # 正弦位置编码，让模型知道时序关系

    # Step 4: Transformer编码器（多层自注意力）
    h = self.trans(h)       # nlayer层TransformerEncoderLayer
    # 每帧可以attend到所有其他帧，建模时序依赖

    # Step 5: 线性投影回输出维度
    h = self.l2(h)          # (B, T, out_dim)

    # Step 6: 残差连接（可选）
    if self.residual:
        x = x[..., :self.out_dim] + h  # 残差：输入+输出
    else:
        x = h                          # 无残差：直接使用输出

    x = x.permute(1,0,2)  # (B,T,C) → (T,B,C) 转回时序优先
    return x
```

**两种使用场景**:

- **Space-Time Module**: `residual=True`, in_dim=1283, out_dim=1280, 在特征层面做时序建模
- **Motion Module**: `residual=False`, in_dim=99(96+3), out_dim=96, 在pose层面做时序平滑

---

### 函数12: `HAWOR.bbox_est(center, scale, img_focal, img_center)`

**文件**: `lib/models/hawor.py:514-524`

**功能**: CLIFF方法的bbox特征编码

**参数**:

- `center` (Tensor): (B, 2) 裁剪区域中心 [cx, cy]
- `scale` (Tensor): (B,) bbox缩放因子
- `img_focal` (Tensor): (B,) 焦距
- `img_center` (Tensor): (B, 2) 图像中心

**返回值**:

- `bbox_info` (Tensor): (B, 3) 归一化的bbox特征

**完整实现过程**:

```python
def bbox_est(self, center, scale, img_focal, img_center):
    # Step 1: 获取图像中心和bbox参数
    img_cx, img_cy = img_center[:,0], img_center[:,1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    # b = scale × 200 是bbox的像素大小（200是参考尺寸）

    # Step 2: 构建原始特征向量
    bbox_info = torch.stack([cx - img_cx, cy - img_cy, b], dim=-1)
    # [bbox中心x偏移, bbox中心y偏移, bbox大小]

    # Step 3: 归一化
    # 位置偏移：除以焦距×2.8（近似归一化到[-1,1]范围）
    bbox_info[:, :2] = bbox_info[:, :2] / img_focal.unsqueeze(-1) * 2.8
    # bbox大小：减去0.24×焦距后除以0.06×焦距（标准化到0附近）
    bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * img_focal) / (0.06 * img_focal)

    return bbox_info  # (B, 3)
```

**CLIFF方法的核心思想**: 将bbox在图像中的位置和大小编码为特征向量，让模型知道"手在图像的哪个位置、有多大"。这是因为同样的3D手部姿态，如果手在图像中心vs边缘，或手大vs小，对应的3D参数是不同的。0.24和0.06是经验值，对应训练数据中bbox大小的均值和标准差（以焦距为单位）。

---

### 函数13: `HAWOR.get_trans(pred_cam, center, scale, img_focal, img_center)`

**文件**: `lib/models/hawor.py:497-512`

**功能**: 将弱透视相机参数转换为全3D平移

**参数**:

- `pred_cam` (Tensor): (B, 3) 弱透视参数 [s, tx, ty]
- `center` (Tensor): (B, 2) 裁剪区域中心
- `scale` (Tensor): (B,) bbox缩放因子
- `img_focal` (Tensor): (B,) 焦距
- `img_center` (Tensor): (B, 2) 图像中心

**返回值**:

- `trans_full` (Tensor): (B, 1, 3) 全3D平移向量

**完整实现过程**:

```python
def get_trans(self, pred_cam, center, scale, img_focal, img_center):
    # Step 1: 计算bbox像素大小
    b = scale * 200                          # bbox在原图中的像素大小

    # Step 2: 解包弱透视参数
    cx, cy = center[:,0], center[:,1]        # 裁剪区域中心
    s, tx, ty = pred_cam.unbind(-1)          # 弱透视参数
    img_cx, img_cy = img_center[:,0], img_center[:,1]  # 图像中心

    # Step 3: 计算缩放后的bbox大小
    bs = b * s                               # s是弱透视缩放因子

    # Step 4: 转换到全图坐标
    tx_full = tx + 2*(cx - img_cx)/bs        # 全图x偏移 = 裁剪区偏移 + 中心偏移
    ty_full = ty + 2*(cy - img_cy)/bs        # 全图y偏移
    tz_full = 2 * img_focal / bs             # 深度 = 2f/bs

    # Step 5: 组装并添加维度
    trans_full = torch.stack([tx_full, ty_full, tz_full], dim=-1)  # (B, 3)
    trans_full = trans_full.unsqueeze(1)     # (B, 1, 3)

    return trans_full
```

**数学原理**:

- 弱透视模型假设物体深度变化远小于物体到相机距离，因此可以用一个缩放因子s近似
- `s` 是缩放因子，`tx, ty` 是裁剪区域内的2D偏移
- 转换到全图坐标需要加上裁剪中心与图像中心的偏移
- `tz = 2f/bs` 来自弱透视投影公式：物体在图像中的大小 ∝ f × 实际大小 / 深度
- 系数2是因为bbox大小b的定义方式（b = scale × 200，200是参考尺寸）

---

### 函数14: `TrackDatasetEval.__getitem__(index)`

**文件**: `lib/datasets/track_dataset.py:42-77`

**功能**: 单帧数据预处理（裁剪+归一化）

**参数**:

- `index` (int): 帧索引

**返回值**:

- `item` (dict): 包含 img, scale, center, img_focal, img_center, do_flip, img_idx

**完整实现过程**:

```python
def __getitem__(self, index):
    item = {}
    imgfile = self.imgfiles[index]
    scale = self.scales[index] * self.box_dilate  # dilate=1.2, 扩大20%
    center = self.centers[index]                   # 从bbox转换来的中心

    img_focal = self.img_focal
    img_center = self.img_center

    # Step 1: 读取图像并处理翻转
    img = cv2.imread(imgfile)[:, :, ::-1]  # BGR → RGB
    if self.do_flip:
        img = img[:, ::-1, :]              # 水平翻转
        center[0] = img.shape[1] - center[0] - 1  # 翻转中心x坐标

    # Step 2: 裁剪到256×256
    img_crop = crop(img, center, scale,
                    [self.crop_size, self.crop_size], rot=0)
    # crop函数通过仿射变换将bbox区域裁剪到指定大小

    # Step 3: 归一化
    if self.normalization:
        img_crop = self.normalize_img(img_crop)  # ImageNet归一化
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

    # Step 4: 组装输出
    item['img'] = img_crop                       # (3, 256, 256)
    if self.do_flip:
        item['do_flip'] = torch.tensor(1).float()
    item['img_idx'] = torch.tensor(index).long()
    item['scale'] = torch.tensor(scale).float()
    item['center'] = torch.tensor(center).float()
    item['img_focal'] = torch.tensor(img_focal).float()
    item['img_center'] = torch.tensor(img_center).float()
    return item
```

**关键细节**:

- `dilate=1.2` 将bbox扩大20%，确保手部周围有足够上下文信息
- `boxes_2_cs` 将 `[x1,y1,x2,y2]` 转为 `[center_x, center_y]` + `scale`
- `crop` 函数通过仿射变换将bbox区域裁剪到256×256，保持手部在中心
- ImageNet归一化使用固定的mean和std，与ViT预训练一致

---

### 函数15: `interpolate_bboxes(bboxes)`

**文件**: `lib/eval_utils/custom_utils.py:141-156`

**功能**: 对缺失帧的bbox进行线性插值

**参数**:

- `bboxes` (np.ndarray): (T, 5) bbox序列 [x1,y1,x2,y2,conf]，缺失帧为全零

**返回值**:

- `interpolated_bboxes` (np.ndarray): (T, 5) 插值后的bbox

**完整实现过程**:

```python
def interpolate_bboxes(bboxes):
    T = bboxes.shape[0]

    # Step 1: 找到缺失帧和有效帧
    zero_indices = np.where(np.all(bboxes == 0, axis=1))[0]      # 全零行=缺失帧
    non_zero_indices = np.where(np.any(bboxes != 0, axis=1))[0]  # 非零行=有效帧

    if len(zero_indices) == 0:
        return bboxes  # 没有缺失帧，直接返回

    # Step 2: 对5个维度分别线性插值
    interpolated_bboxes = bboxes.copy()
    for i in range(5):  # x1, y1, x2, y2, conf
        interp_func = interp1d(
            non_zero_indices,               # 已知x坐标（帧号）
            bboxes[non_zero_indices, i],    # 已知y值（bbox坐标/置信度）
            kind='linear',                  # 线性插值
            fill_value="extrapolate"        # 允许外推（超出范围的用端点值）
        )
        interpolated_bboxes[zero_indices, i] = interp_func(zero_indices)

    return interpolated_bboxes
```

**为什么需要插值**: YOLO检测器在某些帧可能漏检（手被遮挡或模糊），导致bbox缺失。插值可以填补这些空缺，使HaWoR模型能处理连续帧序列。

---

## 四、阶段③：Masked SLAM

### 函数16: `hawor_slam(args, start_idx, end_idx)`

**文件**: `scripts/scripts_test_video/hawor_slam.py:46-139`

**功能**: 完整的SLAM流程（含尺度恢复）

**参数**:

- `args`: EasyDict对象，需包含 `video_path`, `img_focal`
- `start_idx` (int): 起始帧索引
- `end_idx` (int): 结束帧索引

**返回值**: 无（结果保存到文件）

**整体实现思路**: 这是阶段③的核心函数，分为4大步骤：1）**Masked DROID-SLAM**：加载手部mask，将手部区域像素置零后送入DROID-SLAM，估计相机轨迹（关键帧位姿+逆深度图）；2）**Metric3D深度估计**：对每个关键帧预测度量深度（米制单位）；3）**尺度恢复**：SLAM输出是无量纲的，需要通过对比SLAM深度和Metric3D深度来估计尺度因子。使用混合方法（迭代中位数粗估计 + Geman-McClure鲁棒优化精估计）；4）**保存结果**：将相机轨迹、逆深度、尺度因子等保存为npz文件。

**完整实现过程**:

```python
def hawor_slam(args, start_idx, end_idx):
    # ===== 1. 准备路径 =====
    file = args.video_path
    video_root = os.path.dirname(file)
    video = os.path.basename(file).split('.')[0]
    seq_folder = os.path.join(video_root, video)
    img_folder = f'{seq_folder}/extracted_images'
    imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))

    # ===== 2. 加载手部mask =====
    masks = np.load(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_masks.npy',
                    allow_pickle=True)
    masks = torch.from_numpy(masks)  # (T, H, W) bool

    # ===== 3. 相机内参 =====
    focal = args.img_focal
    if focal is None:
        try:
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'r') as file:
                focal = float(file.read())
        except:
            focal = 600
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'w') as file:
                file.write(str(focal))

    # 估计内参 [focal, focal, cx, cy]，然后用指定焦距覆盖
    calib = np.array(est_calib(imgfiles))
    calib[:2] = focal  # 用指定焦距替换估计值

    # ===== 4. Masked DROID-SLAM =====
    droid, traj = run_slam(imgfiles, masks=masks, calib=calib)
    # traj: (N_kf, 7) [tx,ty,tz, qx,qy,qz,qw] 关键帧位姿

    n = droid.video.counter.value  # 关键帧数量
    tstamp = droid.video.tstamp.cpu().int().numpy()[:n]  # 关键帧时间戳
    disps = droid.video.disps_up.cpu().numpy()[:n]        # 逆深度图

    del droid
    torch.cuda.empty_cache()  # 释放SLAM占用的GPU内存

    # ===== 5. Metric3D 深度估计 =====
    block_print()  # 抑制Metric3D初始化输出
    metric = Metric3D('thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth')
    enable_print()

    min_threshold = 0.4   # 近距离阈值（米）
    max_threshold = 0.7   # 远距离阈值（米）

    pred_depths = []
    H, W = get_dimention(imgfiles)  # DROID-SLAM使用的图像尺寸
    for t in tqdm(tstamp):
        pred_depth = metric(imgfiles[t], calib)  # 度量深度（米）
        pred_depth = cv2.resize(pred_depth, (W, H))  # 调整到SLAM分辨率
        pred_depths.append(pred_depth)

    # ===== 6. 尺度恢复 =====
    scales_ = []
    for i in tqdm(range(n)):
        t = tstamp[i]
        disp = disps[i]                    # SLAM逆深度
        pred_depth = pred_depths[i]         # Metric3D度量深度
        slam_depth = 1 / disp               # SLAM深度 = 1/逆深度

        msk = masks[t].numpy().astype(np.uint8)  # 手部mask
        scale = est_scale_hybrid(slam_depth, pred_depth,
                                sigma=0.5, msk=msk,
                                near_thresh=min_threshold,
                                far_thresh=max_threshold)

        # 如果scale为NaN，放宽阈值重试
        while math.isnan(scale):
            min_threshold -= 0.1  # 放宽近阈值
            max_threshold += 0.1  # 放宽远阈值
            scale = est_scale_hybrid(slam_depth, pred_depth,
                                    sigma=0.5, msk=msk,
                                    near_thresh=min_threshold,
                                    far_thresh=max_threshold)
        scales_.append(scale)

    # 取所有关键帧的中位数作为最终尺度
    median_s = np.median(scales_)

    # ===== 7. 保存 =====
    os.makedirs(f"{seq_folder}/SLAM", exist_ok=True)
    save_path = f'{seq_folder}/SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz'
    np.savez(save_path,
            tstamp=tstamp, disps=disps, traj=traj,
            img_focal=focal, img_center=calib[-2:],
            scale=median_s)
```

**NaN处理策略**: 当有效像素太少导致尺度估计失败时，逐步放宽深度阈值（近阈值减小，远阈值增大），增加有效像素数量。

---

### 函数17: `run_slam(imagedir, masks, calib, ...)`

**文件**: `lib/pipeline/masked_droid_slam.py:130-158`

**功能**: Masked DROID-SLAM主循环

**参数**:

- `imagedir` (str/list): 图片目录或路径列表
- `masks` (Tensor): (T, H, W) 手部mask
- `calib` (np.ndarray): [fx, fy, cx, cy] 相机内参
- `stride` (int): 帧步长
- `filter_thresh` (float): 运动阈值

**返回值**:

- `droid` (Droid): DROID-SLAM对象（包含视频信息）
- `traj` (np.ndarray): (N_kf, 7) 关键帧位姿

**整体实现思路**: DROID-SLAM是一种基于深度学习的视觉SLAM系统，通过迭代Dense Optical Flow估计帧间对应关系，进而优化相机位姿和深度。HaWoR对其进行了"Masked"改进：1）在图像层面将手部像素置零，防止SLAM在手部区域提取特征点；2）在特征层面传入confidence mask，降低手部区域的匹配权重。双重mask确保SLAM只利用背景特征点，避免手部运动干扰相机运动估计。

**完整实现过程**:

```python
def run_slam(imagedir, masks, calib=None, depth=None, stride=1, filter_thresh=2.4):
    droid = None
    masks = masks[::stride]  # 按步长采样mask

    # Step 1: 预处理mask尺寸
    img_msks, conf_msks = preprocess_masks(imagedir, masks)
    # img_msks: (T, H, W) 图像级mask
    # conf_msks: (T, H//8, W//8) 特征级mask

    if calib is None:
        calib = est_calib(imagedir)  # 粗略估计内参

    # Step 2: 逐帧送入DROID-SLAM
    for (t, image, intrinsics) in tqdm(image_stream(imagedir, calib, stride)):
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)  # 初始化DROID-SLAM

        # Step 3: 应用mask
        img_msk = img_msks[t]       # 图像级mask (H, W)
        conf_msk = conf_msks[t]     # 置信度mask (H//8, W//8)
        image = image * (img_msk < 0.5)  # 手部区域像素置零（黑色）
        # 这样SLAM就不会在手部区域提取特征点

        # Step 4: 送入SLAM
        droid.track(t, image, intrinsics=intrinsics, depth=depth, mask=conf_msk)
        # conf_msk传入correlation层，降低手部区域的匹配权重

    # Step 5: 全局优化并返回轨迹
    traj = droid.terminate(image_stream(imagedir, calib, stride))

    return droid, traj
```

**双重mask机制**:

1. **图像级mask** (`img_msk`): 直接将手部像素置零，SLAM不会在手部区域提取特征点
2. **特征级mask** (`conf_msk`): 传入DROID-SLAM的correlation层，即使有漏网的特征点也会被降低权重

---

### 函数18: `preprocess_masks(img_folder, masks)`

**文件**: `lib/pipeline/masked_droid_slam.py:265-283`

**功能**: 将mask调整为DROID-SLAM所需的两种尺寸

**参数**:

- `img_folder` (str): 图片目录
- `masks` (Tensor): (T, H, W) 原始mask

**返回值**:

- `img_msks` (Tensor): (T, H, W) 图像级mask
- `conf_msks` (Tensor): (T, H//8, W//8) 特征级mask

**完整实现过程**:

```python
def preprocess_masks(img_folder, masks):
    H, W = get_dimention(img_folder)  # DROID-SLAM使用的图像尺寸
    resize_1 = Resize((H, W), antialias=True)       # 图像级尺寸
    resize_2 = Resize((H//8, W//8), antialias=True) # 特征级尺寸(1/8分辨率)

    # 分批处理避免OOM（每批500帧）
    img_msks = []
    for i in range(0, len(masks), 500):
        m = resize_1(masks[i:i+500])
        img_msks.append(m)
    img_msks = torch.cat(img_msks)

    conf_msks = []
    for i in range(0, len(masks), 500):
        m = resize_2(masks[i:i+500])
        conf_msks.append(m)
    conf_msks = torch.cat(conf_msks)

    return img_msks, conf_msks
```

**两种mask的区别**:

- `img_msks (H, W)`: 直接与图像相乘，手部像素置零
- `conf_msks (H//8, W//8)`: 传入DROID-SLAM的correlation层，1/8分辨率对应特征图尺寸

---

### 函数19: `est_scale_hybrid(slam_depth, pred_depth, sigma, msk, ...)`

**文件**: `lib/pipeline/est_scale.py:74-110`

**功能**: 两阶段尺度恢复（迭代中位数 + Geman-McClure鲁棒优化）

**参数**:

- `slam_depth` (np.ndarray): (H, W) SLAM深度图
- `pred_depth` (np.ndarray): (H, W) Metric3D度量深度图
- `sigma` (float): Geman-McClure损失参数，默认0.5
- `msk` (np.ndarray): (H, W) 手部mask
- `near_thresh` (float): 近距离阈值
- `far_thresh` (float): 远距离阈值

**返回值**:

- `scale` (float): 度量尺度因子

**整体实现思路**: DROID-SLAM输出的深度是无量纲的（只知道相对深度关系），而Metric3D输出的是米制深度。尺度恢复的目标是找到一个scale因子，使得 `slam_depth × scale ≈ pred_depth`。两阶段方法：第一阶段用迭代中位数做粗估计（比均值更鲁棒，不受离群值影响），每次迭代用当前scale过滤SLAM深度后重新估计；第二阶段用Geman-McClure鲁棒损失做精估计（BFGS优化），该损失对小残差敏感、对大残差不敏感，自动抑制手部区域和遮挡区域的离群值。

**完整实现过程**:

```python
def est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=None,
                     near_thresh=0, far_thresh=10):
    # Step 0: 处理mask
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    # ===== Stage 1: 迭代中位数（粗估计）=====
    s = pred_depth / slam_depth  # 逐像素尺度比

    # 构建鲁棒mask：排除手部、无效深度、过近/过远像素
    robust = (msk < 0.5) * (near_thresh < pred_depth) * (pred_depth < far_thresh)
    s_est = s[robust]
    scale = np.median(s_est)  # 初始估计：中位数（比均值更鲁棒）

    # 迭代优化：用当前scale过滤SLAM深度，重新估计
    for _ in range(10):
        slam_depth_0 = slam_depth * scale  # 用当前scale缩放SLAM深度
        robust = (msk < 0.5) * (0 < slam_depth_0) * (slam_depth_0 < far_thresh) * \
                 (near_thresh < pred_depth) * (pred_depth < far_thresh)
        s_est = s[robust]
        scale = np.median(s_est)

    # ===== Stage 2: Geman-McClure鲁棒优化（精估计）=====
    robust = (msk < 0.5) * (0 < slam_depth_0) * (slam_depth_0 < far_thresh) * \
             (near_thresh < pred_depth) * (pred_depth < far_thresh)
    pm = torch.from_numpy(pred_depth[robust])  # Metric3D深度
    sm = torch.from_numpy(slam_depth[robust])   # SLAM深度

    def f(x):
        loss = sm * x - pm                      # 残差：SLAM深度×scale - Metric3D深度
        loss = gmof(loss, sigma=sigma).mean()   # 鲁棒损失
        return loss

    x0 = torch.tensor([scale])                  # 用Stage 1的结果作为初始值
    result = minimize(f, x0, method='bfgs')     # BFGS优化
    scale = result.x.detach().cpu().item()

    return scale
```

**Geman-McClure损失函数**:

```python
def gmof(x, sigma=100):
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)
```

- 当 `|x| << σ` 时，`gmof ≈ x²`（二次损失，对小残差敏感）
- 当 `|x| >> σ` 时，`gmof ≈ σ²`（常数，对大残差不敏感）
- 这使得离群值（如手部区域的错误深度、遮挡区域）不会影响优化
- `sigma=0.5` 意味着深度差超过0.5米的像素被视为离群值

---

### 函数20: `load_slam_cam(fpath)`

**文件**: `lib/eval_utils/custom_utils.py:129-138`

**功能**: 加载SLAM结果并转换为c2w/w2c矩阵

**参数**:

- `fpath` (str): SLAM结果文件路径(.npz)

**返回值**:

- `R_w2c_sla` (Tensor): (N, 3, 3) world→camera旋转
- `t_w2c_sla` (Tensor): (N, 3) world→camera平移
- `R_c2w_sla` (Tensor): (N, 3, 3) camera→world旋转
- `t_c2w_sla` (Tensor): (N, 3) camera→world平移

**整体实现思路**: DROID-SLAM输出的是c2w（camera-to-world）变换，格式为7维向量 [tx,ty,tz, qx,qy,qz,qw]（平移+四元数）。加载后需要：1）将平移乘以尺度因子（SLAM平移是无量纲的，需要转为米制）；2）将四元数转为旋转矩阵（注意四元数顺序：DROID-SLAM用xyzw，PyTorch的quaternion_to_matrix用wxyz，需要重排）；3）从c2w计算w2c（逆变换：R_w2c = R_c2w^T, t_w2c = -R_w2c @ t_c2w）。

**完整实现过程**:

```python
def load_slam_cam(fpath):
    pred_cam = dict(np.load(fpath, allow_pickle=True))
    pred_traj = pred_cam['traj']  # (N, 7) [tx,ty,tz, qx,qy,qz,qw]

    # Step 1: 平移——应用度量尺度
    t_c2w_sla = torch.tensor(pred_traj[:, :3]) * pred_cam['scale']
    # SLAM输出的平移是无量纲的，乘以scale转为米制单位

    # Step 2: 旋转——四元数→旋转矩阵
    pred_camq = torch.tensor(pred_traj[:, 3:])  # (N, 4) [qx,qy,qz,qw]
    # quaternion_to_matrix期望 [w,x,y,z] 顺序
    R_c2w_sla = quaternion_to_matrix(pred_camq[:, [3,0,1,2]])  # 重排为[w,x,y,z]

    # Step 3: 计算w2c（c2w的逆变换）
    R_w2c_sla = R_c2w_sla.transpose(-1, -2)  # R_w2c = R_c2w^T
    t_w2c_sla = -torch.einsum("bij,bj->bi", R_w2c_sla, t_c2w_sla)
    # t_w2c = -R_w2c @ t_c2w

    return R_w2c_sla, t_w2c_sla, R_c2w_sla, t_c2w_sla
```

**坐标系约定**:

- DROID-SLAM输出的是camera-to-world (c2w) 变换
- HaWoR需要world-to-camera (w2c) 用于投影，以及c2w用于坐标变换
- w2c = c2w的逆：R_w2c = R_c2w^T, t_w2c = -R_w2c @ t_c2w

---

## 五、阶段④：Infilling（缺失帧填补 + 坐标变换）

### 函数21: `hawor_infiller(args, start_idx, end_idx, frame_chunks_all)`

**文件**: `scripts/scripts_test_video/hawor_video.py:222-365`

**功能**: 完整的坐标变换 + 缺失帧填补流程

**参数**:

- `args`: EasyDict对象
- `start_idx` (int): 起始帧索引
- `end_idx` (int): 结束帧索引
- `frame_chunks_all` (dict): {0: 左手分段, 1: 右手分段}

**返回值**:

- `pred_trans` (Tensor): (2, T, 3) 世界空间平移
- `pred_rot` (Tensor): (2, T, 3) 世界空间根旋转(axis-angle)
- `pred_hand_pose` (Tensor): (2, T, 45) 世界空间关节旋转(axis-angle)
- `pred_betas` (Tensor): (2, T, 10) shape参数
- `pred_valid` (np.ndarray): (2, T) 有效帧标记

**整体实现思路**: 这是阶段④的核心函数，分为3大步骤：1）**相机空间→世界空间**：加载SLAM相机位姿，对每个有效帧段调用`cam2world_convert`将手部参数从相机空间变换到世界空间；2）**缺失帧填补**：找出缺失帧的连续段，对每段截取120帧窗口，调用`filling_preprocess`转换到正则空间，送入Transformer Infiller推理，再调用`filling_postprocess`转回世界空间；3）**结果合并**：只替换缺失帧的参数，保留有效帧的原始参数，最终保存为world_space_res.pth。

**完整实现过程**:

```python
def hawor_infiller(args, start_idx, end_idx, frame_chunks_all):
    # ===== 1. 加载Infiller模型 =====
    pos_dim, shape_dim, num_joints = 3, 10, 15
    rot_dim = (num_joints + 1) * 6  # 96 (16关节×6D旋转)
    repr_dim = 2 * (pos_dim + shape_dim + rot_dim)  # 218 (双手参数拼接)
    filling_model = TransformerModel(
        seq_len=120, input_dim=repr_dim,
        d_model=384, nhead=8, d_hid=2048,
        nlayers=8, dropout=0.05, out_dim=repr_dim,
        masked_attention_stage=True)
    filling_model.load_state_dict(ckpt['transformer_encoder_state_dict'])
    filling_model.eval()

    # ===== 2. 加载SLAM相机位姿 =====
    fpath = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(fpath)

    # ===== 3. 初始化空的世界空间参数 =====
    T = len(imgfiles)
    pred_trans     = torch.zeros(2, T, 3)      # 双手×帧数×3
    pred_rot       = torch.zeros(2, T, 3)      # axis-angle
    pred_hand_pose = torch.zeros(2, T, 45)     # 15关节×3
    pred_betas     = torch.zeros(2, T, 10)
    pred_valid     = torch.zeros((2, T))        # 0=缺失, 1=有效

    # ===== 4. 相机空间→世界空间 =====
    for idx in [0, 1]:  # 0=左手, 1=右手
        for frame_ck in frame_chunks_all[idx]:
            # 读取cam_space JSON
            data_out = {k: torch.tensor(v) for k, v in pred_dict.items()}
            R_c2w_sla = R_c2w_all[frame_ck]
            t_c2w_sla = t_c2w_all[frame_ck]

            # 坐标变换
            data_world = cam2world_convert(R_c2w_sla, t_c2w_sla, data_out,
                                          'right' if idx > 0 else 'left')
            pred_trans[[idx], frame_ck] = data_world["init_trans"]
            pred_rot[[idx], frame_ck] = data_world["init_root_orient"]
            pred_hand_pose[[idx], frame_ck] = data_world["init_hand_pose"].flatten(-2)
            pred_betas[[idx], frame_ck] = data_world["init_betas"]
            pred_valid[[idx], frame_ck] = 1

    # ===== 5. Infiller填补缺失帧 =====
    pred_valid = (pred_valid > 0).numpy()
    for idx in [1, 0]:  # 先右手后左手
        missing = ~pred_valid[idx]
        frame_chunks = parse_chunks_hand_frame(frame_list[missing])

        for frame_ck in frame_chunks:
            # 5a. 找到缺失段前面的有效帧作为起点
            start_shift = -1
            while frame_ck[0] + start_shift >= 0 and \
                  pred_valid[:, frame_ck[0] + start_shift].sum() != 2:
                start_shift -= 1
            # 需要找到双手都有效的帧作为起点

            # 5b. 截取120帧窗口
            filling_net_start = max(0, frame_ck[0] + start_shift)
            filling_net_end = min(T-1, filling_net_start + 120)
            seq_valid = pred_valid[:, filling_net_start:filling_net_end]

            filling_seq = {
                'trans': pred_trans[:, filling_net_start:filling_net_end].numpy(),
                'rot': pred_rot[:, filling_net_start:filling_net_end].numpy(),
                'hand_pose': pred_hand_pose[:, filling_net_start:filling_net_end].numpy(),
                'betas': pred_betas[:, filling_net_start:filling_net_end].numpy(),
                'valid': seq_valid,
            }

            # 5c. 预处理：世界→正则空间 + 插值 + rot6d
            filling_input, transform_w_canon = filling_preprocess(filling_seq)

            # 5d. 构建mask和padding
            src_mask = torch.zeros((120, 120), device=device).type(torch.bool)
            filling_input = torch.from_numpy(filling_input).unsqueeze(0).to(device)
            filling_input = filling_input.permute(1,0,2)  # (T, B, 218)

            T_original = len(filling_input)
            if T_original < 120:
                # padding到120帧：重复最后一帧
                pad_length = 120 - T_original
                last_time_step = filling_input[-1, :, :]
                padding = last_time_step.unsqueeze(0).repeat(pad_length, 1, 1)
                filling_input = torch.cat([filling_input, padding], dim=0)
                seq_valid_padding = np.concatenate([seq_valid,
                    np.ones((2, 120 - T_original))], axis=1)
            else:
                seq_valid_padding = seq_valid

            # 5e. 构建注意力mask
            T, B, _ = filling_input.shape
            # data_mask: 标记哪些帧有真实值（1=有效，0=缺失）
            valid = torch.from_numpy(seq_valid_padding).unsqueeze(0).all(dim=1).permute(1, 0)
            data_mask = torch.zeros((120, B, 1), device=device, dtype=filling_input.dtype)
            data_mask[valid] = 1

            # atten_mask: 不可见帧不能attend到其他不可见帧
            valid_atten = torch.from_numpy(seq_valid_padding).unsqueeze(0).all(dim=1)
            atten_mask = torch.ones((B, 1, 120), device=device, dtype=torch.bool)
            atten_mask[valid_atten] = False
            atten_mask = atten_mask.unsqueeze(2).repeat(1, 1, T, 1)  # (B,1,T,T)

            # 5f. Transformer推理
            output_ck = filling_model(filling_input, src_mask, data_mask, atten_mask)
            output_ck = output_ck.permute(1,0,2).reshape(T, 2, -1).cpu().detach()
            output_ck = output_ck[:T_original]  # 裁掉padding

            # 5g. 后处理：正则→世界空间
            filling_output = filling_postprocess(output_ck, transform_w_canon)

            # 5h. 只替换缺失帧的参数
            filling_seq['trans'][~seq_valid] = filling_output['trans'][~seq_valid]
            filling_seq['rot'][~seq_valid] = filling_output['rot'][~seq_valid]
            filling_seq['hand_pose'][~seq_valid] = filling_output['hand_pose'][~seq_valid]
            filling_seq['betas'][~seq_valid] = filling_output['betas'][~seq_valid]

            pred_trans[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['trans'])
            pred_valid[:, filling_net_start:filling_net_end] = 1

    # ===== 6. 保存 =====
    joblib.dump([pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid],
                os.path.join(seq_folder, "world_space_res.pth"))
    return pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid
```

---

### 函数22: `cam2world_convert(R_c2w_sla, t_c2w_sla, data_out, handedness)`

**文件**: `lib/eval_utils/custom_utils.py:67-97`

**功能**: 将相机空间手部参数转换到世界空间

**参数**:

- `R_c2w_sla` (Tensor): (T, 3, 3) camera→world旋转矩阵
- `t_c2w_sla` (Tensor): (T, 3) camera→world平移向量
- `data_out` (dict): 相机空间手部参数
- `handedness` (str): 'left' 或 'right'

**返回值**:

- `data_world` (dict): 世界空间手部参数

**整体实现思路**: 坐标变换的核心难点在于平移变换。旋转可以直接用矩阵乘法 `R_world = R_c2w @ R_cam`，但平移不能简单做 `t_world = R_c2w @ t_cam + t_c2w`，因为MANO的 `transl` 参数不是wrist关节的位置，而是模型原点的平移。两者之间有一个固定偏移 `offset = transl - wrist_joint`。正确的做法是：先在相机空间运行MANO获取wrist位置，将wrist变换到世界空间，再加回offset。

**完整实现过程**:

```python
def cam2world_convert(R_c2w_sla, t_c2w_sla, data_out, handedness):
    # ===== Step 1: 旋转变换 =====
    init_rot_mat = copy.deepcopy(data_out["init_root_orient"])  # (B, T, 3, 3)
    # R_world = R_c2w @ R_cam
    init_rot_mat = torch.einsum("tij,btjk->btik", R_c2w_sla, init_rot_mat)
    init_rot = rotation_matrix_to_angle_axis(init_rot_mat)  # → axis-angle (B, T, 3)

    # ===== Step 2: 转换data_out中的旋转为axis-angle（用于MANO）=====
    data_out_init_root_orient = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
    data_out_init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])

    # ===== Step 3: 平移变换（通过MANO root joint）=====
    init_trans = data_out["init_trans"]  # (B, T, 3) 相机空间平移

    # 在相机空间运行MANO获取root joint (wrist) 位置
    if handedness == "right":
        outputs = run_mano(data_out["init_trans"], data_out_init_root_orient,
                          data_out_init_hand_pose, betas=data_out["init_betas"])
    else:
        outputs = run_mano_left(data_out["init_trans"], data_out_init_root_orient,
                               data_out_init_hand_pose, betas=data_out["init_betas"])
    root_loc = outputs["joints"][..., 0, :].cpu()  # (B, T, 3) wrist关节位置

    # offset = transl - root_loc (MANO平移与root joint的固定偏移)
    offset = init_trans - root_loc

    # 世界空间平移 = R_c2w @ root_loc_cam + t_c2w + offset
    init_trans = (
        torch.einsum("tij,btj->bti", R_c2w_sla, root_loc)
        + t_c2w_sla[None, :]
        + offset
    )

    return {
        "init_root_orient": init_rot,              # (B, T, 3) axis-angle
        "init_hand_pose": data_out_init_hand_pose,  # (B, T, 15, 3) axis-angle
        "init_trans": init_trans,                   # (B, T, 3)
        "init_betas": data_out["init_betas"]        # (B, T, 10)
    }
```

**为什么需要offset**: MANO模型的 `transl` 参数不是wrist关节的位置，而是模型原点的平移。两者之间有一个固定偏移 `offset = transl - wrist_joint`。这个偏移在旋转变换下保持不变（因为它是MANO模型内部的偏移，与相机坐标系无关），所以直接加到世界空间的结果上。

**平移变换的推导**:

```
root_world = R_c2w @ root_cam + t_c2w    (wrist关节的世界坐标)
trans_world = root_world + offset         (MANO平移 = wrist + offset)
           = R_c2w @ root_cam + t_c2w + offset
```

---

### 函数22.1: `world2canonical_convert(R_c2w_sla, t_c2w_sla, data_out, handedness)`

**文件**: `lib/eval_utils/filling_utils.py:113-144`

**功能**: 通用的坐标变换函数（世界→正则空间 或 正则→世界空间）

**参数**:

- `R_c2w_sla` (Tensor): 变换旋转矩阵（可以是 R_w2canon 或 R_canon2w）
- `t_c2w_sla` (Tensor): 变换平移向量
- `data_out` (dict): 输入空间的手部参数
- `handedness` (str): 'left' 或 'right'

**返回值**:

- `data_world` (dict): 输出空间的手部参数

**整体实现思路**: 此函数与 `cam2world_convert` 结构完全相同，是一个通用的刚体坐标变换函数。通过传入不同的R和t，可以实现任意两个坐标系之间的变换。在 `filling_preprocess` 中传入 `R_w2canon` 实现世界→正则空间变换；在 `filling_postprocess` 中传入 `R_canon2w` 实现正则→世界空间变换。

**完整实现过程**:

```python
def world2canonical_convert(R_c2w_sla, t_c2w_sla, data_out, handedness):
    # 与cam2world_convert完全相同的结构
    # 1. 旋转变换: R_out = R_transform @ R_in
    init_rot_mat = copy.deepcopy(data_out["init_root_orient"])
    init_rot_mat = torch.einsum("tij,btjk->btik", R_c2w_sla, init_rot_mat)
    init_rot = rotation_matrix_to_angle_axis(init_rot_mat)

    # 2. 转换旋转为axis-angle
    data_out_init_root_orient = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
    data_out_init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])

    # 3. 平移变换: 通过MANO root joint
    init_trans = data_out["init_trans"]
    if handedness == "left":
        outputs = run_mano_left(...)
    elif handedness == "right":
        outputs = run_mano(...)
    root_loc = outputs["joints"][..., 0, :].cpu()
    offset = init_trans - root_loc
    init_trans = (
        torch.einsum("tij,btj->bti", R_c2w_sla, root_loc)
        + t_c2w_sla[None, :]
        + offset
    )

    return {
        "init_root_orient": init_rot,
        "init_hand_pose": data_out_init_hand_pose,
        "init_trans": init_trans,
        "init_betas": data_out["init_betas"]
    }
```

**两种使用场景**:

1. **filling_preprocess中**: 传入 `R_w2canon` 和 `t_w2canon`，将世界空间参数转换到正则空间
2. **filling_postprocess中**: 传入 `R_canon2w` 和 `t_canon2w`，将正则空间参数转换回世界空间

---

### 函数23: `filling_preprocess(item)`

**文件**: `lib/eval_utils/filling_utils.py:146-251`

**功能**: Infiller预处理（世界→正则空间 + 插值 + rot6d编码）

**参数**:

- `item` (dict): 包含 trans(2,T,3), rot(2,T,3), hand_pose(2,T,45), betas(2,T,10), valid(2,T)

**返回值**:

- `global_pose_vec_input` (np.ndarray): (T, 218) 正则空间+rot6d编码的输入向量
- `transform_w_canon` (dict): 世界↔正则空间的变换矩阵（用于后处理逆变换）

**整体实现思路**: Infiller的预处理分为5步：1）计算世界→正则空间的刚体变换（以第一帧的旋转为参考，消除全局旋转，使序列更平滑）；2）将双手参数分别变换到正则空间；3）对缺失帧的平移/shape做线性插值，对旋转做SLERP球面插值；4）将旋转从axis-angle转为rot6d表示（连续且无奇点，更适合神经网络）；5）将所有参数拼接为218维向量，同时保存逆变换矩阵供后处理使用。

**完整实现过程**:

```python
def filling_preprocess(item):
    num_joints = 15
    global_trans = item['trans']       # (2, T, 3)
    global_rot = item['rot']           # (2, T, 3) axis-angle
    hand_pose = item['hand_pose']      # (2, T, 45)
    betas = item['betas']             # (2, T, 10)
    valid = item['valid']             # (2, T) bool
    N, T, _ = global_trans.shape       # N=2(双手)

    # ===== Step 1: 计算世界→正则变换 =====
    # 以第一帧的旋转作为正则坐标系（消除全局旋转，使序列更平滑）
    R_canonical2world_left_aa = torch.from_numpy(global_rot[0, 0])   # 左手第0帧旋转
    R_canonical2world_right_aa = torch.from_numpy(global_rot[1, 0])  # 右手第0帧旋转
    R_world2canonical_left = angle_axis_to_rotation_matrix(R_canonical2world_left_aa).t()
    R_world2canonical_right = angle_axis_to_rotation_matrix(R_canonical2world_right_aa).t()
    # R_world2canonical = R_canonical2world^T（旋转矩阵的逆=转置）

    # ===== Step 2: 左手变换到正则空间 =====
    hand_pose = hand_pose.reshape(N, T, num_joints, 3)
    data_world_left = {
        "init_trans": torch.from_numpy(global_trans[0:1]),
        "init_root_orient": angle_axis_to_rotation_matrix(torch.from_numpy(global_rot[0:1])),
        "init_hand_pose": angle_axis_to_rotation_matrix(torch.from_numpy(hand_pose[0:1])),
        "init_betas": torch.from_numpy(betas[0:1]),
    }

    # 运行MANO获取root joint，计算平移偏移
    data_left_init_root_orient = rotation_matrix_to_angle_axis(data_world_left["init_root_orient"])
    data_left_init_hand_pose = rotation_matrix_to_angle_axis(data_world_left["init_hand_pose"])
    outputs = run_mano_left(data_world_left["init_trans"], data_left_init_root_orient,
                           data_left_init_hand_pose, betas=data_world_left["init_betas"])
    init_trans = data_world_left["init_trans"][0, 0]  # (3,)
    root_loc = outputs["joints"][0, 0, 0, :].cpu()    # (3,) wrist位置
    offset = init_trans - root_loc

    # 计算正则空间的平移：t_w2canon = -R_w2canon @ root - offset
    t_world2canonical_left = -torch.einsum("ij,j->i", R_world2canonical_left, root_loc) - offset

    # 应用变换
    R_world2canonical_left = R_world2canonical_left.repeat(T, 1, 1)
    t_world2canonical_left = t_world2canonical_left.repeat(T, 1)
    data_canonical_left = world2canonical_convert(
        R_world2canonical_left, t_world2canonical_left, data_world_left, "left")

    # ===== Step 3: 右手同理 =====
    data_canonical_right = world2canonical_convert(...)

    # ===== Step 4: 插值缺失帧 =====
    # 对平移和shape做线性插值
    global_trans_lerped = linear_interpolation_nd(global_trans, valid)
    betas_lerped = linear_interpolation_nd(betas, valid)
    # 对旋转做球面线性插值(SLERP)
    global_rot_slerped = slerp_interpolation_aa(global_rot, valid)
    hand_pose_slerped = slerp_interpolation_aa(hand_pose, valid)

    # ===== Step 5: 转为rot6d表示 =====
    # rot6d比axis-angle更适合神经网络处理（连续且无奇点）
    global_rot_slerped_rot6d = rotmat_to_rot6d(
        angle_axis_to_rotation_matrix(global_rot_slerped)).reshape(N, T, -1).numpy()
    hand_pose_slerped_rot6d = rotmat_to_rot6d(
        angle_axis_to_rotation_matrix(hand_pose_slerped)).reshape(N, T, -1).numpy()

    # ===== Step 6: 拼接为输入向量 =====
    global_pose_vec_input = np.concatenate(
        (global_trans_lerped, betas_lerped, global_rot_slerped_rot6d, hand_pose_slerped_rot6d),
        axis=-1).transpose(1, 0, 2).reshape(T, -1)
    # 形状: (T, 218) = (T, 2×(3+10+6+90))
    #       左手: trans(3) + betas(10) + root_rot(6) + hand_pose(90)
    #       右手: trans(3) + betas(10) + root_rot(6) + hand_pose(90)

    # ===== Step 7: 保存逆变换 =====
    R_canon2w_left = R_world2canonical_left.transpose(-1, -2)
    t_canon2w_left = -torch.einsum("tij,tj->ti", R_canon2w_left, t_world2canonical_left)
    R_canon2w_right = R_world2canonical_right.transpose(-1, -2)
    t_canon2w_right = -torch.einsum("tij,tj->ti", R_canon2w_right, t_world2canonical_right)

    transform_w_canon = {
        "R_w2canon_left": R_world2canonical_left, "t_w2canon_left": t_world2canonical_left,
        "R_canon2w_left": R_canon2w_left, "t_canon2w_left": t_canon2w_left,
        "R_w2canon_right": R_world2canonical_right, "t_w2canon_right": t_world2canonical_right,
        "R_canon2w_right": R_canon2w_right, "t_canon2w_right": t_canon2w_right,
    }

    return global_pose_vec_input, transform_w_canon
```

**正则空间的设计动机**: 世界空间中手部可能有大幅度的全局旋转和平移，直接在原始世界空间做填补很困难。转换到正则空间（以第一帧为参考）后，全局旋转被消除，序列更平滑，Transformer更容易学习。

---

### 函数24: `filling_postprocess(output, transform_w_canon)`

**文件**: `lib/eval_utils/filling_utils.py:260-305`

**功能**: Infiller后处理（rot6d解码 + 正则→世界空间）

**参数**:

- `output` (Tensor): (T, 2, 218) Infiller输出
- `transform_w_canon` (dict): 世界↔正则空间变换矩阵

**返回值**:

- `pred_data` (dict): 世界空间手部参数

**整体实现思路**: 后处理是预处理的逆过程，分为4步：1）从218维输出向量中拆分出平移、shape、根旋转rot6d、关节旋转rot6d；2）将rot6d通过Gram-Schmidt正交化转为旋转矩阵；3）使用保存的逆变换矩阵（R_canon2w, t_canon2w）将正则空间参数转换回世界空间；4）将旋转矩阵转为axis-angle格式，汇总为最终的世界空间手部参数。

**完整实现过程**:

```python
def filling_postprocess(output, transform_w_canon):
    output = output.permute(1, 0, 2)  # (2, T, 218)
    N, T, _ = output.shape

    # ===== Step 1: 拆分输出向量 =====
    canon_trans = output[:, :, :3]              # (2, T, 3) 平移
    betas = output[:, :, 3:13]                  # (2, T, 10) shape
    canon_rot_rot6d = output[:, :, 13:19]       # (2, T, 6) 全局旋转rot6d
    hand_pose_rot6d = output[:, :, 19:109].reshape(N, T, 15, 6)  # 关节旋转rot6d

    # ===== Step 2: rot6d → rotation matrix (Gram-Schmidt) =====
    canon_rot_mat = custom_rot6d_to_rotmat(canon_rot_rot6d)     # (2, T, 3, 3)
    hand_pose_mat = custom_rot6d_to_rotmat(hand_pose_rot6d)     # (2, T, 15, 3, 3)

    # ===== Step 3: 正则空间→世界空间 =====
    data_canonical_left = {
        "init_trans": canon_trans[[0]],
        "init_root_orient": canon_rot_mat[[0]],
        "init_hand_pose": hand_pose_mat[[0]],
        "init_betas": betas[[0]],
    }
    data_canonical_right = {
        "init_trans": canon_trans[[1]],
        "init_root_orient": canon_rot_mat[[1]],
        "init_hand_pose": hand_pose_mat[[1]],
        "init_betas": betas[[1]],
    }

    # world2canonical_convert是通用的坐标变换函数
    # 传入 R_canon2w 和 t_canon2w 就是正则→世界
    world_left = world2canonical_convert(
        transform_w_canon['R_canon2w_left'],
        transform_w_canon['t_canon2w_left'],
        data_canonical_left, "left")
    world_right = world2canonical_convert(
        transform_w_canon['R_canon2w_right'],
        transform_w_canon['t_canon2w_right'],
        data_canonical_right, "right")

    # ===== Step 4: 汇总结果 =====
    pred_data = {
        "trans": torch.cat((world_left['init_trans'], world_right['init_trans'])).numpy(),
        "rot": torch.cat((world_left['init_root_orient'], world_right['init_root_orient'])).numpy(),
        "hand_pose": rotation_matrix_to_angle_axis(hand_pose_mat).flatten(-2).numpy(),
        "betas": betas.numpy(),
    }
    return pred_data
```

---

### 函数25: `TransformerModel.forward(src, src_mask, data_mask, atten_mask)`

**文件**: `infiller/lib/model/network.py:251-276`

**功能**: Infiller Transformer前向传播

**参数**:

- `src` (Tensor): (120, B, 218) 输入序列
- `src_mask` (Tensor): (120, 120) 源mask（全False）
- `data_mask` (Tensor): (120, B, 1) 标记哪些帧有真实值
- `atten_mask` (Tensor): (B, 1, 120, 120) 注意力mask

**返回值**:

- `output` (Tensor): (120, B, 218) 填补后的输出序列

**整体实现思路**: Infiller采用两阶段Transformer架构。第一阶段是Masked MultiHeadedAttention（8层），缺失帧只能从可见帧获取信息，不能互相attend，确保填补结果基于真实观测而非"幻觉"。第二阶段是标准TransformerEncoder，对填补后的序列做全局时序平滑，消除填补帧与真实帧之间的不连续性。输入时将data_mask（标记帧是否有效）拼接到218维参数向量后，让模型知道哪些帧是真实值、哪些是插值的。

**完整实现过程**:

```python
def forward(self, src, src_mask, data_mask=None, atten_mask=None):
    # src: (120, B, 218) 输入序列
    # data_mask: (120, B, 1) 标记哪些帧有真实值（1=有效，0=缺失）
    # atten_mask: (B, 1, 120, 120) 注意力mask

    # ===== Step 1: 拼接data_mask到输入 =====
    if data_mask is not None:
        src = torch.cat([src, data_mask.expand(*src.shape[:-1], data_mask.shape[-1])], dim=-1)
        # src: (120, B, 219) = 218(参数) + 1(有效标志)
        # 让模型知道哪些帧是真实值，哪些是插值的

    # ===== Step 2: 线性投影到d_model维度 =====
    src = self.input_layer(src)  # (120, B, 384)

    # ===== Step 3: 位置编码 =====
    src = self.pos_embedding(src)  # 正弦位置编码，让模型知道时序关系

    # ===== Step 4: Masked MultiHeadedAttention (8层) =====
    if self.att_layers:
        src = src.permute(1, 0, 2)  # (B, 120, 384) batch优先
        for i in range(self.nlayers):
            src = self.att_layers[i](src, mask=atten_mask)  # Masked注意力
            src = self.pff_layers[i](src)                    # FeedForward
        src = self.layer_norm(src)
        src = src.permute(1, 0, 2)  # (120, B, 384) 时序优先

    # ===== Step 5: 标准TransformerEncoder =====
    src = self.transformer_encoder(src)  # 全局时序建模

    # ===== Step 6: 解码到输出维度 =====
    src = self.decoder(src)  # (120, B, 218)

    return src
```

**Masked Attention机制详解**:

- `data_mask`: 1=有真实值，0=需要填补。拼接到输入让模型区分
- `atten_mask`: 不可见帧(i)对不可见帧(j)的注意力设为 `-inf`
- 效果：缺失帧只能从可见帧获取信息，不会"互相抄答案"
- 两阶段设计：先Masked Attention（可见→不可见信息传递），再标准Transformer（全局平滑）

---

### 函数26: `slerp_interpolation_aa(pos, valid)`

**文件**: `lib/eval_utils/filling_utils.py:13-48`

**功能**: 对旋转的axis-angle表示进行球面线性插值(SLERP)

**参数**:

- `pos` (np.ndarray): (B, T, N, 3) axis-angle旋转序列
- `valid` (np.ndarray): (B, T) bool 有效帧标记

**返回值**:

- `pos_interp` (np.ndarray): (B, T, N, 3) 插值后的旋转序列

**完整实现过程**:

```python
def slerp_interpolation_aa(pos, valid):
    B, T, N, _ = pos.shape  # B=2, N=关节数
    pos_interp = pos.copy()

    for b in range(B):
        for n in range(N):
            quat_b_n = pos[b, :, n, :]    # (T, 3) axis-angle
            valid_b_n = valid[b, :]        # (T,) bool

            invalid_idxs = np.where(~valid_b_n)[0]
            valid_idxs = np.where(valid_b_n)[0]

            if len(invalid_idxs) == 0 or len(valid_idxs) <= 1:
                continue  # 无缺失帧或只有1个有效帧，无法插值

            # 用scipy的Slerp进行球面插值
            valid_rots = Rotation.from_rotvec(quat_b_n[valid_idxs])
            slerp = Slerp(valid_times=valid_idxs, rotations=valid_rots)

            for idx in invalid_idxs:
                if idx < valid_idxs[0]:
                    # 外推：复制首值（SLERP不支持外推）
                    pos_interp[b, idx, n, :] = quat_b_n[valid_idxs[0]]
                elif idx > valid_idxs[-1]:
                    # 外推：复制末值
                    pos_interp[b, idx, n, :] = quat_b_n[valid_idxs[-1]]
                else:
                    # 内插：SLERP球面线性插值
                    interp_rot = slerp([idx])
                    pos_interp[b, idx, n, :] = interp_rot.as_rotvec()[0]

    return pos_interp
```

**为什么用SLERP而非线性插值**: 旋转空间是流形的，不是欧几里得空间。线性插值axis-angle会导致中间旋转"走捷径"穿过非物理的旋转路径。SLERP在旋转流形上做最短路径插值，保证中间旋转的合理性。

---

### 函数27: `linear_interpolation_nd(pos, valid)`

**文件**: `lib/eval_utils/filling_utils.py:89-111`

**功能**: 对平移和shape参数进行多维线性插值

**参数**:

- `pos` (np.ndarray): (B, T, D) 参数序列，D为任意维度
- `valid` (np.ndarray): (B, T) bool 有效帧标记

**返回值**:

- `pos_interp` (np.ndarray): (B, T, D) 插值后的参数序列

**完整实现过程**:

```python
def linear_interpolation_nd(pos, valid):
    B, T = pos.shape[:2]
    feature_dim = pos.shape[2]
    pos_interp = pos.copy()

    for b in range(B):
        for idx in range(feature_dim):
            pos_b_idx = pos[b, :, idx]  # (T,) 当前维度的时序序列
            valid_b = valid[b, :]

            invalid_idxs = np.where(~valid_b)[0]
            valid_idxs = np.where(valid_b)[0]

            if len(invalid_idxs) == 0:
                continue

            # 对无效部分进行线性插值
            if len(valid_idxs) > 1:
                pos_b_idx[invalid_idxs] = np.interp(
                    invalid_idxs,      # 需要插值的x坐标
                    valid_idxs,        # 已知x坐标
                    pos_b_idx[valid_idxs]  # 已知y值
                )
                pos_interp[b, :, idx] = pos_b_idx

    return pos_interp
```

**与SLERP的区别**: 平移和shape参数在欧几里得空间中，线性插值是合理的。旋转参数在流形上，需要SLERP。

---

## 六、阶段⑤：MANO 正演 + 可视化渲染

### 函数28: `run_mano(trans, root_orient, hand_pose, betas, use_cuda)`

**文件**: `hawor/utils/process.py:29-105`

**功能**: 右手MANO模型前向传播

**参数**:

- `trans` (Tensor): (B, T, 3) 平移
- `root_orient` (Tensor): (B, T, 3) 根旋转axis-angle
- `hand_pose` (Tensor): (B, T, 45) 关节旋转axis-angle (15×3)
- `betas` (Tensor): (B, T, 10) shape参数
- `use_cuda` (bool): 是否使用CUDA

**返回值**:

- `outputs` (dict): {"joints": (B,T,21,3), "vertices": (B,T,778,3), 可选"faces"}

**整体实现思路**: MANO（MANO hand model）是一个参数化手部模型，输入全局旋转(3D)、15个关节旋转(45D)、shape参数(10D)和平移(3D)，输出778个顶点和21个关节的3D坐标。实现步骤：1）将axis-angle旋转转为旋转矩阵（MANO内部使用旋转矩阵表示）；2）将所有参数reshape为(B*T, ...)格式，一次性送入MANO前向传播（`pose2rot=False`跳过内部转换）；3）添加14个指尖补充面片（标准MANO面片只有1538个，指尖区域不够精细）；4）reshape回(B, T, ...)格式输出。

**完整实现过程**:

```python
def run_mano(trans, root_orient, hand_pose, is_right=None, betas=None, use_cuda=True):
    block_print()  # 抑制MANO初始化输出
    mano = MANO(DATA_DIR='_DATA/data/', MODEL_PATH='_DATA/data/mano', ...)
    if use_cuda:
        mano = mano.cuda()

    B, T, _ = root_orient.shape
    NUM_JOINTS = 15

    # ===== Step 1: axis-angle → rotation matrix =====
    mano_params = {
        'global_orient': root_orient.reshape(B*T, -1),     # (B*T, 3)
        'hand_pose': hand_pose.reshape(B*T*NUM_JOINTS, 3), # (B*T*15, 3)
        'betas': betas.reshape(B*T, -1),                   # (B*T, 10)
    }
    rotmat_mano_params = mano_params
    rotmat_mano_params['global_orient'] = aa_to_rotmat(
        mano_params['global_orient']).view(B*T, 1, 3, 3)    # (B*T, 1, 3, 3)
    rotmat_mano_params['hand_pose'] = aa_to_rotmat(
        mano_params['hand_pose']).view(B*T, NUM_JOINTS, 3, 3) # (B*T, 15, 3, 3)
    rotmat_mano_params['transl'] = trans.reshape(B*T, 3)   # (B*T, 3)

    # ===== Step 2: MANO前向传播 =====
    mano_output = mano(**{k: v.float().cuda() for k,v in rotmat_mano_params.items()},
                       pose2rot=False)  # 已经是旋转矩阵，不需要再转换
    # pose2rot=False: 跳过MANO内部的axis-angle→rotation转换（我们已经手动转换了）

    # ===== Step 3: 添加指尖面片 =====
    faces_right = np.concatenate([mano.faces, faces_new], axis=0)  # (1552, 3)
    # 14个指尖面片用于渲染更完整的指尖形状
    faces_left = faces_right[:, [0,2,1]]  # 左手：翻转面片绕序

    # ===== Step 4: 整理输出 =====
    outputs = {
        "joints": mano_output.joints.reshape(B, T, -1, 3),     # (B, T, 21, 3)
        "vertices": mano_output.vertices.reshape(B, T, -1, 3),  # (B, T, 778, 3)
    }

    # 如果提供了is_right标记，根据左右手选择对应的面片
    if not is_right is None:
        is_right = (is_right[:, :, 0].cpu().numpy() > 0)
        faces_result = np.where(is_right[..., np.newaxis, np.newaxis],
                               faces_right_expanded, faces_left_expanded)
        outputs["faces"] = torch.from_numpy(faces_result.astype(np.int32))

    enable_print()
    return outputs
```

---

### 函数29: `run_mano_left(trans, root_orient, hand_pose, betas, ...)`

**文件**: `hawor/utils/process.py:107-188`

**功能**: 左手MANO模型前向传播（修复shapedirs bug）

**与run_mano的关键区别**:

```python
def run_mano_left(trans, root_orient, hand_pose, is_right=None, betas=None,
                  use_cuda=True, fix_shapedirs=True):
    # 使用左手MANO模型
    mano = MANO(DATA_DIR='_DATA/data_left/', MODEL_PATH='_DATA/data_left/mano_left',
                is_rhand=False, ...)

    # 修复MANO左手shapedirs bug
    # 参考: https://github.com/vchoutas/smplx/issues/48
    # MANO左手的shapedirs（shape blend shapes）的x分量符号错误
    if fix_shapedirs:
        mano.shapedirs[:, 0, :] *= -1  # x轴shape方向取反

    # 后续与run_mano完全相同
    ...
```

**shapedirs bug**: MANO左手模型的shape blend shapes在x轴方向上符号错误，导致左手在不同shape下变形方向不正确。修复方法是将x轴方向的shapedirs取反。

---

### 函数30: `Renderer.render_multiple(verts_list, faces, colors_list, cameras, lights)`

**文件**: `lib/vis/renderer.py:281-315`

**功能**: 使用pytorch3d渲染多个mesh（用于生成手部mask或可视化）

**参数**:

- `verts_list` (Tensor): (B, V, 3) 顶点列表
- `faces` (Tensor): (F, 3) 面片
- `colors_list` (Tensor): (B, 4) RGBA颜色
- `cameras`: PyTorch3D相机对象
- `lights`: PyTorch3D光源对象

**返回值**:

- `image` (np.ndarray): (H, W, 3) 渲染图像
- `mask` (np.ndarray): (H, W) bool 渲染mask

**完整实现过程**:

```python
def render_multiple(self, verts_list, faces, colors_list, cameras, lights):
    verts_, faces_, colors_ = [], [], []

    # Step 1: 准备所有mesh的几何数据
    for i, verts in enumerate(verts_list):
        colors = colors_list[[i]]
        # prep_shared_geometry: 将颜色扩展到每个顶点，面片扩展到batch
        verts_i, faces_i, colors_i = prep_shared_geometry(verts, faces, colors)
        # verts_i: (B, V, 3), faces_i: (B, F, 3), colors_i: (B, V, 3)

        if i == 0:
            verts_ = list(torch.unbind(verts_i, dim=0))
            faces_ = list(torch.unbind(faces_i, dim=0))
            colors_ = list(torch.unbind(colors_i, dim=0))
        else:
            # 合并多个mesh（如左右手）
            verts_ += list(torch.unbind(verts_i, dim=0))
            faces_ += list(torch.unbind(faces_i, dim=0))
            colors_ += list(torch.unbind(colors_i, dim=0))

    # Step 2: 合并所有mesh为一个场景
    mesh = create_meshes(verts_, faces_, colors_)  # join_meshes_as_scene

    # Step 3: pytorch3d渲染
    materials = Materials(device=self.device, shininess=0)
    results = self.renderer(mesh, cameras=cameras, lights=lights, materials=materials)

    # Step 4: 提取图像和mask
    image = (results[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)  # RGB图像
    mask = results[0, ..., -1].cpu().numpy() > 0  # alpha通道作为mask
    return image, mask
```

---

### 函数31: `run_vis2_on_video(res_dict, res_dict2, output_pth, ...)`

**文件**: `lib/vis/run_vis2.py:39-143`

**功能**: 世界视角可视化渲染

**参数**:

- `res_dict` (dict): 左手结果 {"vertices": (1,T,778,3), "faces": (1552,3)}
- `res_dict2` (dict): 右手结果
- `output_pth` (str): 输出路径
- `focal_length` (float): 焦距
- `image_names` (list): 图片路径列表
- `R_c2w` (Tensor): camera→world旋转
- `t_c2w` (Tensor): camera→world平移
- `interactive` (bool): 是否交互式渲染

**返回值**:

- `video_path` (str): 输出视频路径

**整体实现思路**: 世界视角可视化使用aitviewer库渲染3D场景。场景包含4类元素：1）左手mesh（紫色）和右手mesh（蓝色）；2）棋盘格地面（z=-2平面）；3）相机轨迹标记（金字塔形，红色=右方向，绿色=上方向）；4）侧视角观察相机（固定位置看向场景中心）。渲染时使用固定的侧视角相机（非视频原始相机），这样可以看到手部在世界坐标系中的3D运动轨迹。

**完整实现过程**:

```python
def run_vis2_on_video(res_dict, res_dict2, output_pth, focal_length,
                      image_names, R_c2w=None, t_c2w=None, interactive=True):
    img0 = cv2.imread(image_names[0])
    height, width, _ = img0.shape

    # ===== 1. 构建可视化字典 =====
    vis_dict = {}

    # 左手mesh（紫色）
    for _id, _verts in enumerate(res_dict['vertices']):
        verts = _verts.cpu().numpy()  # (T, N, 3)
        vis_dict[f"hand_{_id}"] = {
            "v3d": verts,
            "f3d": res_dict['faces'],
            "color": "director-purple",
        }

    # 右手mesh（蓝色）
    for _id, _verts in enumerate(res_dict2['vertices']):
        verts = _verts.cpu().numpy()
        vis_dict[f"hand2_{_id}"] = {
            "v3d": verts,
            "f3d": res_dict2['faces'],
            "color": "director-blue",
        }

    # 棋盘格地面
    v, f, vc, fc = checkerboard_geometry(length=100, c1=0, c2=0, up="z")
    v[:, 2] -= 2  # z平面下移2个单位
    vis_dict["ground"] = {"v3d": v, "f3d": f, "vc": vc, "fc": fc, "color": -1}

    # 相机轨迹标记（金字塔形）
    num_frames = len(res_dict['vertices'][_id])
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = R_c2w[:num_frames]
    Rt[:, :3, 3] = t_c2w[:num_frames]
    verts, faces, face_colors = camera_marker_geometry(0.05, 0.1)
    # 将相机标记变换到每帧的相机位置
    verts = np.einsum("tij,nj->tni", Rt[:, :3, :3], verts) + Rt[:, None, :3, 3]
    vis_dict["camera"] = {"v3d": verts, "f3d": faces, "fc": face_colors, "color": -1}

    # ===== 2. 侧视角观察相机 =====
    side_source = torch.tensor([0.463, -0.478, 2.456])   # 相机位置
    side_target = torch.tensor([0.026, -0.481, -3.184])  # 观察目标
    up = torch.tensor([1.0, 0.0, 0.0])                   # 上方向
    view_camera = lookat_matrix(side_source, side_target, up)
    viewer_Rt = np.tile(view_camera[:3, :4], (num_frames, 1, 1))

    # ===== 3. aitviewer渲染 =====
    meshes = viewer_utils.construct_viewer_meshes(vis_dict, draw_edges=False)
    K = np.array([[1000, 0, width/2], [0, 1000, height/2], [0, 0, 1]])
    data = viewer_utils.ViewerData(viewer_Rt, K, width, height)
    batch = (meshes, data)

    if interactive:
        viewer = viewer_utils.ARCTICViewer(interactive=True, size=(width, height))
        viewer.render_seq(batch, out_folder=os.path.join(output_pth, 'aitviewer'))
    else:
        viewer = viewer_utils.ARCTICViewer(interactive=False, size=(width, height),
                                           render_types=['video'])
        viewer.render_seq(batch, out_folder=os.path.join(output_pth, 'aitviewer'))
        return os.path.join(output_pth, 'aitviewer', "video_0.mp4")
```

---

## 七、辅助函数详解

### 辅助函数1: `est_calib(imagedir)`

**文件**: `lib/pipeline/masked_droid_slam.py:59-71`

**功能**: 粗略估计相机内参

**参数**:

- `imagedir` (str/list): 图片目录或路径列表

**返回值**:

- `calib` (list): [fx, fy, cx, cy] 相机内参

**完整实现过程**:

```python
def est_calib(imagedir):
    if isinstance(imagedir, list):
        imgfiles = imagedir
    else:
        imgfiles = sorted(glob(f'{imagedir}/*.jpg'))
    image = cv2.imread(imgfiles[0])

    h0, w0, _ = image.shape
    focal = np.max([h0, w0])    # 焦距≈max(H,W)，经验估计
    cx, cy = w0/2., h0/2.       # 主点=图像中心
    calib = [focal, focal, cx, cy]
    return calib
```

**原理**: 当不知道相机内参时，一个常见的近似是假设焦距约等于图像较长边的像素数，主点在图像中心。这个估计虽然粗糙，但DROID-SLAM对内参的初始值不敏感，后续会通过优化调整。在HaWoR中，这个估计值会被用户指定的焦距覆盖（`calib[:2] = focal`），只保留主点估计。

---

### 辅助函数2: `get_dimention(imagedir)`

**文件**: `lib/pipeline/masked_droid_slam.py:74-89`

**功能**: 获取DROID-SLAM使用的图像尺寸

**参数**:

- `imagedir` (str/list): 图片目录或路径列表

**返回值**:

- `H, W` (int, int): 调整后的图像高度和宽度

**完整实现过程**:

```python
def get_dimention(imagedir):
    if isinstance(imagedir, list):
        imgfiles = imagedir
    else:
        imgfiles = sorted(glob(f'{imagedir}/*.jpg'))
    image = cv2.imread(imgfiles[0])

    h0, w0, _ = image.shape
    # 等比缩放使面积≈384×512=196608像素
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    # 裁剪到8的倍数（DROID-SLAM的特征提取需要8倍下采样）
    image = cv2.resize(image, (w1, h1))
    image = image[:h1-h1%8, :w1-w1%8]
    H, W, _ = image.shape
    return H, W
```

**为什么需要调整尺寸**: DROID-SLAM内部使用特征提取器（类似ResNet），需要8倍下采样。如果图像尺寸不是8的倍数，下采样后的尺寸会不对。同时，DROID-SLAM对大约384×512的图像效果最好，所以先等比缩放到这个面积附近。

---

### 辅助函数3: `image_stream(imagedir, calib, stride, max_frame=None)`

**文件**: `lib/pipeline/masked_droid_slam.py:92-127`

**功能**: 图像流生成器，逐帧读取图像并预处理为DROID-SLAM所需的格式

**参数**:

- `imagedir` (str/list): 图片目录或路径列表
- `calib` (list): [fx, fy, cx, cy] 相机内参
- `stride` (int): 帧步长
- `max_frame` (int, optional): 最大帧数

**返回值**: 生成器，每次yield (t, image, intrinsics)

**完整实现过程**:

```python
def image_stream(imagedir, calib, stride, max_frame=None):
    fx, fy, cx, cy = calib[:4]

    # 构建内参矩阵K
    K = np.eye(3)
    K[0,0] = fx; K[0,2] = cx
    K[1,1] = fy; K[1,2] = cy

    image_list = sorted(glob(f'{imagedir}/*.jpg'))
    image_list = image_list[::stride]  # 按步长采样
    if max_frame is not None:
        image_list = image_list[:max_frame]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(imfile)
        # 如果有畸变系数，去畸变
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        # 等比缩放到≈384×512
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        # 裁剪到8的倍数
        image = image[:h1-h1%8, :w1-w1%8]
        # HWC → CHW，添加batch维度
        image = torch.as_tensor(image).permute(2, 0, 1)

        # 内参按缩放比例调整
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)  # fx, cx 按宽度缩放
        intrinsics[1::2] *= (h1 / h0)  # fy, cy 按高度缩放

        yield t, image[None], intrinsics  # image: (1, C, H, W)
```

**关键细节**: 内参需要根据图像缩放比例调整。如果图像缩小了一半，焦距和主点也要缩小一半，否则投影就不对了。

---

### 辅助函数4: `MultiHeadedAttention.forward(hidden, memory=None, mask=None)`

**文件**: `infiller/lib/model/network.py:70-151`

**功能**: Infiller中的Masked多头注意力层

**参数**:

- `hidden` (Tensor): (batch, seq, dim) 输入序列
- `memory` (Tensor, optional): 记忆张量（当前未使用）
- `mask` (BoolTensor, optional): (batch, 1, seq, seq) 注意力mask

**返回值**:

- `output` (Tensor): (batch, seq, dim) 注意力输出

**完整实现过程**:

```python
def forward(self, hidden, memory=None, mask=None, extra_atten_score=None):
    combined = hidden

    # Step 1: Pre-LayerNorm（先归一化再计算，训练更稳定）
    if self.pre_lnorm:
        hidden = self.layer_norm(hidden)
        combined = self.layer_norm(combined)

    # Step 2: QKV投影
    q = self.q_linear(hidden)       # (batch, seq, n_head*d_head)
    k = self.k_linear(combined)
    v = self.v_linear(combined)

    # Step 3: 重排为多头格式
    q = q.reshape(q.shape[0], q.shape[1], self.n_head, self.d_head).transpose(1, 2)
    k = k.reshape(k.shape[0], k.shape[1], self.n_head, self.d_head).transpose(1, 2)
    v = v.reshape(v.shape[0], v.shape[1], self.n_head, self.d_head).transpose(1, 2)
    # (batch, n_head, seq, d_head)

    # Step 4: 计算注意力分数
    atten_score = torch.matmul(q, k.transpose(-1, -2))  # (batch, n_head, q_len, k_len)
    atten_score = atten_score * self.atten_scale  # 缩放: 1/sqrt(d_model)

    # Step 5: 应用mask（缺失帧不能attend到其他缺失帧）
    if mask is not None:
        atten_score = atten_score.masked_fill(mask, float("-inf"))

    # Step 6: Softmax + Dropout
    atten_score = atten_score.softmax(dim=-1)
    atten_score = self.atten_dropout_layer(atten_score)

    # Step 7: 加权求和
    atten_vec = torch.matmul(atten_score, v)  # (batch, n_head, q_len, d_head)
    atten_vec = atten_vec.transpose(1, 2).flatten(start_dim=-2)  # (batch, q_len, n_head*d_head)

    # Step 8: 输出投影 + 残差连接
    output = self.droput_layer(self.out_linear(atten_vec))
    if self.pre_lnorm:
        return hidden + output  # 残差连接
    else:
        return self.layer_norm(hidden + output)
```

**Masked Attention的核心机制**: `atten_mask` 的shape为 (B, 1, T, T)，其中缺失帧(i)对缺失帧(j)的注意力位置设为True（即-inf），缺失帧对可见帧的注意力位置设为False（正常计算）。效果是：缺失帧只能从可见帧获取信息，不会"互相抄答案"。

---

### 辅助函数5: `FeedForward.forward(x)`

**文件**: `infiller/lib/model/network.py:154-189`

**功能**: 位置前馈网络（Position-wise FeedForward）

**参数**:

- `x` (Tensor): (batch, seq, d_model) 输入

**返回值**:

- `x` (Tensor): (batch, seq, d_model) 输出

**完整实现过程**:

```python
def forward(self, x):
    if self.pre_lnorm:
        return x + self.network(self.layer_norm(x))  # Pre-LN: x + FFN(LN(x))
    else:
        return self.layer_norm(x + self.network(x))  # Post-LN: LN(x + FFN(x))
```

**网络结构**: `Linear(d_model, d_inner) → ReLU → Dropout → Linear(d_inner, d_model) → Dropout`

---

### 辅助函数6: `PositionalEncoding.forward(x)`

**文件**: `lib/models/modules.py:116-133`

**功能**: 正弦位置编码，为序列添加位置信息

**参数**:

- `x` (Tensor): (T, B, C) 输入序列

**返回值**:

- `x` (Tensor): (T, B, C) 添加位置编码后的序列

**完整实现过程**:

```python
def forward(self, x):
    x = x + self.pe[:x.shape[0], :]  # 加上预计算的位置编码
    return self.dropout(x)
```

**位置编码公式**:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

其中pos是位置索引，i是维度索引。这种编码方式让模型能够感知序列中元素的相对位置关系。

---

### 辅助函数7: `lookat_matrix(source_pos, target_pos, up)`

**文件**: `lib/vis/run_vis2.py:216-230`

**功能**: 构建观察矩阵（从source看向target）

**参数**:

- `source_pos` (Tensor): (*, 3) 相机位置
- `target_pos` (Tensor): (*, 3) 观察目标
- `up` (Tensor): (3,) 上方向向量

**返回值**:

- `pose` (Tensor): (*, 4, 4) 4×4观察矩阵

**完整实现过程**:

```python
def lookat_matrix(source_pos, target_pos, up):
    # 归一化up向量
    up = up / torch.linalg.norm(up, dim=-1, keepdim=True)

    # 计算三个正交基向量
    back = normalize(target_pos - source_pos)    # 前方（从source指向target）
    right = normalize(torch.linalg.cross(up, back))  # 右方 = up × back
    up = normalize(torch.linalg.cross(back, right))  # 重新计算up确保正交

    # 组装旋转矩阵 [right, up, back]
    R = torch.stack([right, up, back], dim=-1)  # (*, 3, 3)

    # 组装4×4变换矩阵
    return make_4x4_pose(R, source_pos)
```

**坐标系约定**: 使用Right-Up-Back (XYZ)约定，即x=右，y=上，z=后（从相机看向场景的方向）。这与OpenCV的相机坐标系（x=右，y=下，z=前）不同，需要注意转换。

---

### 辅助函数8: `HAWOR.project(points, pred_cam, center, scale, img_focal, img_center)`

**文件**: `lib/models/hawor.py:478-498`

**功能**: 将3D关节点投影到2D图像坐标

**参数**:

- `points` (Tensor): (B, N, 3) 3D关节点
- `pred_cam` (Tensor): (B, 3) 弱透视相机参数
- `center` (Tensor): (B, 2) 裁剪区域中心
- `scale` (Tensor): (B,) bbox缩放因子
- `img_focal` (Tensor): (B,) 焦距
- `img_center` (Tensor): (B, 2) 图像中心

**返回值**:

- `points2d` (Tensor): (B, N, 2) 裁剪图像中的2D坐标

**完整实现过程**:

```python
def project(self, points, pred_cam, center, scale, img_focal, img_center, return_full=False):
    # Step 1: 计算全3D平移
    trans_full = self.get_trans(pred_cam, center, scale, img_focal, img_center)

    # Step 2: 3D点 + 平移 → 透视投影到全图2D
    points = points + trans_full
    points2d_full = perspective_projection(points, rotation=None, translation=None,
                    focal_length=img_focal, camera_center=img_center)
    # points2d_full: (B, N, 2) 全图坐标

    # Step 3: 从全图坐标转换到裁剪图像坐标
    b = scale * 200
    points2d = points2d_full - (center - b[:,None]/2)[:,None,:]  # 减去裁剪区域左上角
    points2d = points2d * (self.crop_size / b)[:,None,None]       # 缩放到256×256

    if return_full:
        return points2d_full, points2d
    else:
        return points2d
```

**两阶段投影的原因**: HaWoR模型在裁剪后的256×256图像上预测手部参数，但2D监督信号可能来自全图坐标。因此需要先投影到全图坐标，再转换到裁剪图像坐标。

---

### 辅助函数9: `Renderer.__init__(width, height, focal_length, device, ...)`

**文件**: `lib/vis/renderer.py:103-115`

**功能**: 初始化PyTorch3D渲染器

**参数**:

- `width` (int): 图像宽度
- `height` (int): 图像高度
- `focal_length` (float): 焦距
- `device` (str): 设备
- `bin_size` (int): 光栅化bin大小
- `max_faces_per_bin` (int): 每个bin最大面片数

**完整实现过程**:

```python
class Renderer():
    def __init__(self, width, height, focal_length, device, bin_size=None, max_faces_per_bin=None):
        self.width = width
        self.height = height
        self.focal_length = focal_length
        self.device = device

        self.initialize_camera_params()  # 初始化默认相机参数
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])
        self.create_renderer(bin_size, max_faces_per_bin)

    def initialize_camera_params(self):
        # 外参：单位旋转 + 零平移（相机空间）
        self.R = torch.diag(torch.tensor([1, 1, 1])).float().to(self.device).unsqueeze(0)
        self.T = torch.tensor([0, 0, 0]).unsqueeze(0).float().to(self.device)

        # 内参矩阵
        self.K = torch.tensor([
            [self.focal_length, 0, self.width/2],
            [0, self.focal_length, self.height/2],
            [0, 0, 1]
        ]).unsqueeze(0).float().to(self.device)

        # 默认bbox=全图
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        self.cameras = self.create_camera()

    def create_camera_from_cv(self, R, T, K=None, image_size=None):
        # 从OpenCV格式的R,T,K创建PyTorch3D相机
        if K is None: K = self.K
        if image_size is None: image_size = torch.tensor(self.image_sizes)
        cameras = _cameras_from_opencv_projection(R, T, K, image_size)
        lights = PointLights(device=K.device, location=T)
        return cameras, lights
```

**PyTorch3D vs OpenCV坐标系**: PyTorch3D使用NDC坐标系，而HaWoR使用OpenCV坐标系。`_cameras_from_opencv_projection` 负责将OpenCV格式的相机参数转换为PyTorch3D格式。

---

### 辅助函数10: `custom_rot6d_to_rotmat(rot6d)`

**文件**: `lib/eval_utils/filling_utils.py:253-258`

**功能**: rot6d → 旋转矩阵的封装函数

**参数**:

- `rot6d` (Tensor): (*, 6) 6D旋转表示

**返回值**:

- `mat` (Tensor): (*, 3, 3) 旋转矩阵

**完整实现过程**:

```python
def custom_rot6d_to_rotmat(rot6d):
    original_shape = rot6d.shape[:-1]
    rot6d = rot6d.reshape(-1, 6)
    mat = rot6d_to_rotmat(rot6d)  # 调用底层实现（Gram-Schmidt正交化）
    mat = mat.reshape(*original_shape, 3, 3)
    return mat
```

**rot6d → 旋转矩阵的Gram-Schmidt过程**:

```
输入: a1, a2 (各3维，共6维)
b1 = normalize(a1)                              # 第一列：直接归一化
b2 = normalize(a2 - (b1·a2)b1)                  # 第二列：减去a2在b1上的分量后归一化
b3 = b1 × b2                                    # 第三列：叉积
R = [b1 | b2 | b3]                              # 组装旋转矩阵
```

---

## 八、完整数据流图（含张量形状）

```
MP4 视频
  │
  ▼ ffmpeg -vf fps=30
JPG 图片序列: [0000.jpg, 0001.jpg, ..., {N-1}.jpg]
  │
  ▼ YOLO track (detector.pt)
tracks: {id: [{frame, det, det_box(5,), det_handedness}, ...]}
  │
  ├──────────────────────────────────────────────────────────┐
  ▼                                                          ▼
左右手分类 + BBox插值                                   (暂不使用)
final_tracks = {0: left, 1: right}
  │
  ▼ parse_chunks (按断点分段)
frame_chunks: [[f0,f1,...,f15], [f16,...,f31], ...]
boxes_chunks: [[box0,...,box15], ...]
  │
  ├─────────────────────────────────────────────┐
  ▼                                             ▼
HaWoR推理 (16帧窗口)                       生成手部Mask
  │  输入: img(16,3,256,256) + bbox            │  MANO正演 → pytorch3d渲染
  │                                             │
  ▼                                             ▼
相机空间手部参数:                          model_masks: (T, H, W) bool
  init_root_orient: (1,T,3,3) 旋转矩阵          │
  init_hand_pose:   (1,T,15,3,3)                │
  init_trans:       (1,T,3)                      │
  init_betas:       (1,T,10)                     │
  │                                               │
  │  保存为 cam_space/{idx}/{start}_{end}.json    │
  │                                               │
  │                                               ▼
  │                                    Masked DROID-SLAM (droid.pth)
  │                                    图像×mask → DROID.track()
  │                                               │
  │                                               ▼
  │                                    traj: (N_kf, 7) [tx,ty,tz,qx,qy,qz,qw]
  │                                               │
  │                                               ▼
  │                                    Metric3D (metric_depth_vit_large_800k.pth)
  │                                    pred_depth: (H, W) 米制深度
  │                                               │
  │                                               ▼
  │                                    est_scale_hybrid()
  │                                    scale: float (度量尺度因子)
  │                                               │
  │                                               ▼
  │                                    SLAM结果: hawor_slam_w_scale.npz
  │                                               │
  ▼                                               │
cam2world_convert ◄────────────────────────────────┘
  │  R_world = R_c2w @ R_cam
  │  t_world = R_c2w @ root_cam + t_c2w + offset
  ▼
世界空间手部参数 (部分帧有效):
  pred_trans:     (2, T, 3)     ← 大部分帧为0
  pred_rot:       (2, T, 3)
  pred_hand_pose: (2, T, 45)
  pred_betas:     (2, T, 10)
  pred_valid:     (2, T)        ← 0/1标记
  │
  ▼ filling_preprocess
正则空间 + rot6d表示 + 插值:
  global_pose_vec: (T, 218)     ← 2×(3+10+6+90)
  │
  ▼ Transformer Infiller (infiller.pt)
output: (120, 1, 218)           ← 填补后的完整序列
  │
  ▼ filling_postprocess
世界空间手部参数 (全部帧有效):
  pred_trans:     (2, T, 3)     ← 全部已填补
  pred_rot:       (2, T, 3)     axis-angle
  pred_hand_pose: (2, T, 45)    axis-angle
  pred_betas:     (2, T, 10)
  pred_valid:     (2, T)        ← 全1
  │
  ▼ run_mano / run_mano_left
手部Mesh顶点:
  right_verts: (T, 778, 3)
  left_verts:  (T, 778, 3)
  │
  ▼ 坐标系翻转 R_x = diag(1,-1,-1)
  │
  ▼ aitviewer 渲染
输出: video_0.mp4
```

---

## 九、中间文件结构总览

```
{video_name}/
├── extracted_images/                    # 阶段①: 抽帧结果
│   ├── 0000.jpg
│   └── ...
│
├── tracks_{start}_{end}/                # 阶段①②: 检测跟踪结果
│   ├── model_boxes.npy                  #   检测框
│   ├── model_tracks.npy                 #   跟踪信息
│   ├── model_masks.npy                  #   手部mask (T,H,W) bool
│   └── frame_chunks_all.npy             #   分段信息
│
├── cam_space/                           # 阶段②: 相机空间手部参数
│   ├── 0/ {start}_{end}.json            #   左手
│   └── 1/ {start}_{end}.json            #   右手
│
├── SLAM/                                # 阶段③: SLAM结果
│   └── hawor_slam_w_scale_{start}_{end}.npz
│
├── est_focal.txt                        # 焦距缓存
│
├── world_space_res.pth                  # 阶段④: 世界空间手部参数
│
└── vis_{start}_{end}/                   # 阶段⑤: 可视化输出
    └── aitviewer/
        └── video_0.mp4
```

---

## 十、所需权重文件


| 文件          | 路径                                                          | 用途           | 来源                                                                                   |
| ------------- | ------------------------------------------------------------- | -------------- | -------------------------------------------------------------------------------------- |
| HaWoR 模型    | `weights/hawor/checkpoints/hawor.ckpt`                        | 手部位姿估计   | [HuggingFace](https://huggingface.co/ThunderVVV/HaWoR)                                 |
| Infiller 模型 | `weights/hawor/checkpoints/infiller.pt`                       | 缺失帧填补     | [HuggingFace](https://huggingface.co/ThunderVVV/HaWoR)                                 |
| 模型配置      | `weights/hawor/model_config.yaml`                             | 模型结构配置   | [HuggingFace](https://huggingface.co/ThunderVVV/HaWoR)                                 |
| YOLO 检测器   | `weights/external/detector.pt`                                | 手部检测跟踪   | [HuggingFace](https://huggingface.co/spaces/rolpotamias/WiLoR)                         |
| DROID-SLAM    | `weights/external/droid.pth`                                  | 视觉SLAM       | [Google Drive](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view) |
| Metric3D      | `thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth` | 度量深度估计   | [Google Drive](https://drive.google.com/file/d/1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr/view) |
| MANO 右手     | `_DATA/data/mano/MANO_RIGHT.pkl`                              | 右手参数化模型 | [MANO官网](https://mano.is.tue.mpg.de)                                                 |
| MANO 左手     | `_DATA/data_left/mano_left/MANO_LEFT.pkl`                     | 左手参数化模型 | [MANO官网](https://mano.is.tue.mpg.de)                                                 |

---

## 十一、关键数学公式

### 10.1 CLIFF bbox特征编码

```
bbox_info = [(cx - img_cx) / focal × 2.8,  (cy - img_cy) / focal × 2.8,  (b - 0.24×focal) / (0.06×focal)]
```

### 10.2 弱透视→全平移

```
b = scale × 200,  bs = b × s
tx_full = tx + 2(cx - img_cx)/bs
ty_full = ty + 2(cy - img_cy)/bs
tz_full = 2 × focal / bs
```

### 10.3 相机空间→世界空间

```
R_world = R_c2w @ R_cam
offset  = t_cam - root_cam   (MANO平移与root joint的固定偏移)
t_world = R_c2w @ root_cam + t_c2w + offset
```

### 10.4 尺度恢复

```
scale = argmin Σ gmof(slam_depth × s - pred_depth, σ)
gmof(x, σ) = σ²x² / (σ² + x²)    (Geman-McClure鲁棒损失)
```

### 10.5 rot6d → rotation matrix (Gram-Schmidt)

```
a1, a2 = x[:, :3], x[:, 3:]
b1 = normalize(a1)
b2 = normalize(a2 - (b1·a2)b1)
b3 = b1 × b2
R = [b1, b2, b3]
```

### 10.6 SLERP球面线性插值

```
q(t) = (sin((1-t)Ω) × q₀ + sin(tΩ) × q₁) / sin(Ω)
其中 Ω = arccos(q₀ · q₁)
```

---

## 十二、Python API 调用示例

```python
import os, torch, numpy as np
from easydict import EasyDict
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video

args = EasyDict()
args.video_path = './example/video_0.mp4'
args.input_type = 'file'
args.checkpoint = './weights/hawor/checkpoints/hawor.ckpt'
args.infiller_weight = './weights/hawor/checkpoints/infiller.pt'
args.vis_mode = 'world'
args.img_focal = 600

# Step 1: 帧提取 + 检测跟踪
start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

# Step 2: 手部位姿估计 (相机空间)
frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

# Step 3: Masked SLAM
slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
if not os.path.exists(slam_path):
    hawor_slam(args, start_idx, end_idx)

# Step 4: Infilling (世界空间)
pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
    args, start_idx, end_idx, frame_chunks_all)

# Step 5: 仅获取手部3D顶点（不渲染视频）
right_output = run_mano(pred_trans[1:2], pred_rot[1:2], pred_hand_pose[1:2], betas=pred_betas[1:2])
left_output = run_mano_left(pred_trans[0:1], pred_rot[0:1], pred_hand_pose[0:1], betas=pred_betas[0:1])
right_vertices = right_output['vertices'][0]  # (T, 778, 3)
left_vertices = left_output['vertices'][0]    # (T, 778, 3)
right_joints = right_output['joints'][0]      # (T, 21, 3)
left_joints = left_output['joints'][0]        # (T, 21, 3)
```

---

## 八、评价指标体系详解

> HaWoR 使用5个核心指标评估手部位姿估计质量，所有指标计算函数位于 `lib/eval_utils/eval_utils.py`，
> 完整评估脚本位于 `scripts/scripts_eval/eval_hawor_hot3d.py`，
> 简单测试脚本位于 `assess/` 文件夹。

### 8.1 指标总览


| 指标         | 全称                       | 单位       | 评估内容             | 对齐方式             |
| ------------ | -------------------------- | ---------- | -------------------- | -------------------- |
| **PA-MPJPE** | Procrustes-Aligned MPJPE   | mm         | 关节位置误差         | 逐帧 Procrustes 对齐 |
| **W-MPJPE**  | World-space MPJPE          | mm         | 世界空间关节位置误差 | 前两帧对齐           |
| **WA-MPJPE** | World-Aligned MPJPE        | mm         | 世界空间关节位置误差 | 全局对齐             |
| **RTE**      | Relative Translation Error | cm         | 相对平移误差         | 全局对齐后归一化     |
| **Accel**    | Acceleration Error         | mm/frame² | 运动平滑度误差       | 无对齐               |

### 8.2 PA-MPJPE (Procrustes-Aligned Mean Per Joint Position Error)

**含义**: 逐帧对预测关节和GT关节做 Procrustes 对齐（允许旋转、平移、缩放），然后计算平均关节位置误差。

**数学公式**:

```
对每一帧 t:
  1. 找到最优 s, R, t 使得 s·R·pred_t + t 最接近 gt_t
  2. pred_aligned_t = s·R·pred_t + t
  3. PA-MPJPE_t = mean(||gt_t - pred_aligned_t||₂)

PA-MPJPE = mean(PA-MPJPE_t) × 1000  (m → mm)
```

**核心函数**: `batch_compute_similarity_transform_torch(S1, S2)` + `compute_jpe(S1, S2)`

**实现过程**:

1. 对每帧的预测关节和GT关节分别去中心化
2. 计算互相关矩阵 H = X1ᵀ·X2
3. 对 H 做 SVD 分解: H = U·Σ·Vᵀ
4. 恢复旋转 R = V·Z·Uᵀ (Z 保证 det(R)=1)
5. 恢复缩放 s = trace(R·H) / var(X1)
6. 恢复平移 t = μ₂ - s·R·μ₁
7. 对齐后计算 JPE

**特点**: 消除了全局旋转、平移、缩放的影响，只衡量关节的**相对形状**是否正确。

### 8.3 W-MPJPE (World-space MPJPE)

**含义**: 在世界空间中，仅用前两帧对齐预测和GT，然后计算所有帧的关节位置误差。

**数学公式**:

```
1. 用前两帧的关节做 Umeyama 对齐，得到 s_first, R_first, t_first
2. pred_aligned = s_first · R_first · pred + t_first  (对所有帧)
3. W-MPJPE = mean(||gt - pred_aligned||₂) × 1000  (m → mm)
```

**核心函数**: `first_align_joints(gt_joints, pred_joints)` + `compute_jpe()`

**实现过程**:

1. 取前两帧的关节 (2, J, 3) → reshape 为 (1, 2J, 3)
2. 调用 `align_pcl` 做 Umeyama 对齐
3. 将得到的 s, R, t 应用到所有帧
4. 计算对齐后的 JPE

**特点**: 保留了时间维度上的误差累积，衡量世界空间中的**绝对位置**精度。

### 8.4 WA-MPJPE (World-Aligned MPJPE)

**含义**: 在世界空间中，用所有帧的关节做全局 Umeyama 对齐，然后计算关节位置误差。

**数学公式**:

```
1. 将所有帧关节 reshape 为 (T×J, 3)
2. 做 Umeyama 对齐，得到 s_glob, R_glob, t_glob
3. pred_aligned = s_glob · R_glob · pred + t_glob  (对所有帧)
4. WA-MPJPE = mean(||gt - pred_aligned||₂) × 1000  (m → mm)
```

**核心函数**: `global_align_joints(gt_joints, pred_joints)` + `compute_jpe()`

**实现过程**:

1. 将 (T, J, 3) reshape 为 (-1, 3)
2. 调用 `align_pcl` 做全局 Umeyama 对齐
3. 将 s, R, t 应用到所有帧的所有关节
4. 计算对齐后的 JPE

**特点**: 全局对齐消除了坐标系差异，但保留了帧间运动轨迹的误差。

### 8.5 RTE (Relative Translation Error)

**含义**: 评估手腕根部平移轨迹的相对误差，归一化到GT轨迹总位移。

**数学公式**:

```
1. 对 pred_trans 和 gt_trans 做全局对齐 (固定尺度)
2. pred_trans_hat = R · pred_trans + t
3. 计算GT轨迹总位移: disp = Σ ||gt[t+1] - gt[t]||₂
4. RTE = mean(||gt_trans - pred_trans_hat||₂) / disp
```

**核心函数**: `compute_rte(target_trans, pred_trans)`

**实现过程**:

1. 调用 `align_pcl` 做固定尺度的 Umeyama 对齐
2. 将对齐变换应用到预测轨迹
3. 计算GT轨迹的累积位移
4. 计算逐帧绝对误差并归一化

**特点**: 衡量手腕运动轨迹的准确性，是**归一化比值**，最终结果乘以 100 转为 cm。

### 8.6 Accel (Acceleration Error)

**含义**: 评估预测运动的加速度与GT运动加速度之间的差异，衡量运动的**平滑性和时序一致性**。

**数学公式**:

```
1. accel_gt[t] = joints_gt[t-1] - 2·joints_gt[t] + joints_gt[t+1]
2. accel_pred[t] = joints_pred[t-1] - 2·joints_pred[t] + joints_pred[t+1]
3. Accel = mean(||accel_pred - accel_gt||₂) × fps²
```

**核心函数**: `compute_error_accel(joints_gt, joints_pred)`

**实现过程**:

1. 计算GT和预测的二阶差分（加速度）
2. 计算加速度差的 L2 范数
3. 对所有帧和关节取平均
4. 乘以 fps² 缩放（通常 fps=30）

**特点**: 不需要任何对齐，直接衡量运动的**时间一致性**。值越大说明运动越抖动。

### 8.7 Umeyama 对齐算法 (`align_pcl`)

所有对齐操作的核心算法，求解相似变换 X' = s·R·X + t 使 X' 最接近 Y:

```
1. 去中心化: x₀ = X - μₓ, y₀ = Y - μᵧ
2. 计算互相关: C = y₀ᵀ·x₀ / N
3. SVD 分解: C = U·D·Vᵀ
4. 修正反射: S = I, 若 det(U)·det(Vᵀ)<0 则 S[2,2]=-1
5. 旋转: R = U·S·V
6. 缩放: s = trace(D·S) / var(X)
7. 平移: t = μᵧ - s·R·μₓ
```

### 8.8 评估脚本调用方式

完整评估（需要HOT3D数据集GT）:

```bash
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR
python scripts/scripts_eval/eval_hawor_hot3d.py \
    --checkpoint weights/hawor/checkpoints/hawor.ckpt \
    --video_root datasets/hot3d_valset_export \
    --eval_stage
```

简单测试（assess文件夹）:

```bash
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR

# 方法1: 合成数据测试 - 用随机生成的关节数据测试所有指标
python assess/test_synthetic.py

# 方法2: 合理性检查 - 验证指标在边界条件下的正确性
python assess/test_sanity.py

# 方法3: 使用示例推理结果测试 - 加载example中的npz文件
python assess/test_with_example_data.py
python assess/test_with_example_data.py --hand left
python assess/test_with_example_data.py --npz_path example/7/reconstruction/hawor_results_0_113.npz
```

### 8.9 指标参考值


| 场景                     | PA-MPJPE | W-MPJPE    | WA-MPJPE | RTE   | Accel |
| ------------------------ | -------- | ---------- | -------- | ----- | ----- |
| 完全相同输入             | ≈0 mm   | ≈0 mm     | ≈0 mm   | ≈0   | ≈0   |
| 相似变换(旋转+平移+缩放) | ≈0 mm   | 取决于对齐 | ≈0 mm   | ≈0   | ≈0   |
| 小噪声扰动(σ=0.02m)     | ~30 mm   | ~32 mm     | ~31 mm   | ~0.01 | ~0.02 |
| 完全随机预测             | ~148 mm  | ~165 mm    | ~159 mm  | ~2.16 | ~3.88 |

> 注: 参考值基于合成数据测试 (T=50, J=21, seed=42)，实际值取决于数据分布。

---

## 九、Masked SLAM 深度分析：手部遮罩策略与对相机位姿估计的影响

> 核心问题：SLAM中手部区域是直接涂黑（置零），没有做任何填充/修补(inpainting)，
> 这对相机位姿估计有什么影响？HaWoR是如何设计的？

### 9.1 Masked SLAM 的核心操作

HaWoR的Masked SLAM采用**双层掩码机制**，而非简单的"涂黑"：

```
原始图像 ──────────────────────────────────────────────────────► SLAM输入
    │                                                              │
    │  ┌──────────────────────┐                                    │
    │  │ 1. 图像级掩码 img_msk │──► image = image * (img_msk < 0.5) │
    │  │    (H × W 分辨率)     │    手部像素直接置零(黑色)           │
    │  └──────────────────────┘                                    │
    │                                                              │
    │  ┌──────────────────────────────┐                            │
    │  │ 2. 置信度掩码 conf_msk         │──► droid.track(mask=conf_msk)│
    │  │    (H//8 × W//8 分辨率)       │    BA优化中手部权重=0       │
    │  └──────────────────────────────┘                            │
    │                                                              │
    ▼                                                              ▼
  手部区域像素=0(黑色)                                    手部区域不参与位姿优化
```

**关键代码** ([masked_droid_slam.py](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/lib/pipeline/masked_droid_slam.py) 第149-154行):

```python
img_msk = img_msks[t]          # 图像级掩码 (H×W)
conf_msk = conf_msks[t]        # 置信度掩码 (H//8×W//8)
image = image * (img_msk < 0.5) # 手部区域涂黑（像素置0）
droid.track(t, image, intrinsics=intrinsics, depth=depth, mask=conf_msk)
```

### 9.2 手部掩码的生成方式

掩码**不是**简单的边界框(bounding box)，而是通过MANO模型+PyTorch3D渲染器生成的**精确手部轮廓**：

```
YOLO检测框 → HAWOR姿态估计 → MANO正演生成3D网格 → PyTorch3D渲染 → 精确2D掩码
```

**为什么用精确轮廓而非边界框？**
- 边界框会多遮掉大量背景区域（手指间的空隙、手腕外的区域）
- 精确轮廓只遮住手部实际占据的像素，最大限度保留背景信息
- 这对SLAM至关重要——保留更多背景特征点意味着更鲁棒的位姿估计

### 9.3 "涂黑不填充"对SLAM各阶段的影响分析

#### 9.3.1 特征提取阶段（影响：✅ 无负面影响）

```python
# motion_filter.py 第53-57行
inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
inputs = inputs.sub_(self.MEAN).div_(self.STDV)
gmap = self.__feature_encoder(inputs)  # 特征图
```

**分析**：
- 手部区域像素为0（黑色），经过ImageNet标准化后变为 `(-0.485/0.229, -0.456/0.224, -0.406/0.225)` ≈ `(-2.12, -2.04, -1.80)`
- 这些值在特征提取网络中会产生特征响应，但这些特征**与手部运动无关**
- 由于手部区域在后续BA优化中被权重掩码排除，这些"错误特征"不会影响最终位姿
- **关键**：黑色区域不会产生有意义的特征匹配，因此不会产生错误的光度一致性约束

#### 9.3.2 运动滤波阶段（影响：✅ 无负面影响）

```python
# motion_filter.py 第78-85行
corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)
_, delta, weight = self.update(self.net[None], self.inp[None], corr)
if delta.norm(dim=-1).mean().item() > self.thresh:
    # 添加新关键帧
```

**分析**：
- 光流估计在黑色区域会产生不确定的结果
- 但运动滤波只关注**整体运动幅度**（`delta.norm(dim=-1).mean()`），手部区域的光流误差被大量背景区域的正确光流稀释
- 手部通常只占图像的5-15%，对整体均值影响有限

#### 9.3.3 Dense Bundle Adjustment（影响：✅ 核心保护机制）

```python
# factor_graph.py 第225-226行
msk = self.video.masks[self.ii] > 0   # 获取手部掩码
weight[:,msk] = 0.0                    # 手部区域优化权重设为0
```

**分析**：
- 这是Masked SLAM最核心的保护机制
- 即使黑色区域产生了错误的深度或光流估计，在BA优化中这些区域的权重为0
- BA优化的目标函数为：`min Σ weight × ||target - reproject(pose, depth)||²`
- 手部区域 `weight=0`，完全不参与相机位姿和深度的优化
- **这意味着"涂黑不填充"是安全的**——即使填充了，BA优化也不会使用这些区域的约束

#### 9.3.4 深度估计阶段（影响：⚠️ 局部影响，全局可控）

**分析**：
- DROID-SLAM的深度估计是联合优化的，黑色区域可能产生不准确的深度值
- 但由于BA中权重掩码的存在，这些不准确的深度不会"污染"相机位姿估计
- 在后续尺度估计中，同样使用掩码排除手部区域：

```python
# est_scale.py 第85行
robust = (msk<0.5) * (near_thresh<pred_depth) * (pred_depth<far_thresh)
# 只使用非手部区域的深度来估计尺度
```

### 9.4 "涂黑不填充" vs "填充修补" 的对比

| 方面 | 涂黑不填充（HaWoR方案） | 填充修补(Inpainting) |
|------|------------------------|---------------------|
| **实现复杂度** | 简单，一行代码 | 需要额外的inpainting模型 |
| **计算开销** | 几乎为零 | 显著增加（每帧需inpainting） |
| **特征提取** | 黑色区域无有效特征 | 填充区域可能产生伪特征 |
| **BA优化** | 权重掩码直接排除，安全 | 填充内容可能引入错误约束 |
| **误差风险** | 无（被掩码完全排除） | 有（inpainting不完美时引入错误） |
| **背景信息** | 手部区域背景丢失 | 试图恢复背景（但不保证正确） |

### 9.5 为什么"涂黑不填充"是合理的？

#### 原因1：SLAM依赖的是静态背景特征

DROID-SLAM等视觉SLAM系统的核心假设是**场景是静态的**。手部是动态物体，无论是否填充，SLAM都不应该使用手部区域的信息。涂黑+权重掩码的组合确保了这一点。

#### 原因2：填充可能引入更严重的错误

Inpainting是对被遮挡背景的"猜测"，如果猜测不准确：
- 填充的背景纹理与真实场景不一致 → 产生错误的光度约束
- 填充的边缘与真实图像不连续 → 产生错误的特征匹配
- 这些错误约束**没有**被权重掩码排除（因为inpainting后不再是手部区域）

**关键洞察**：涂黑让手部区域"不存在"（权重=0），而填充让手部区域"看起来像背景但实际不是"（权重≠0但约束错误）。前者更安全。

#### 原因3：手部区域占比小，信息损失有限

在典型的egocentric视频中：
- 手部通常只占图像的5-15%
- 背景区域（桌面、墙壁、地面等）提供了大量稳定的特征点
- 丢失5-15%的特征点对SLAM的位姿估计影响很小
- DROID-SLAM使用dense光流（所有像素），即使去掉手部区域仍有大量约束

#### 原因4：双层掩码的协同保护

```
┌─────────────────────────────────────────────────────────────────┐
│  第一层：图像级掩码 (img_msk)                                     │
│  image = image * (img_msk < 0.5)                                │
│  作用：防止特征提取器在手部区域提取特征                             │
│  效果：手部区域不产生特征匹配 → 不产生错误的光流约束                │
├─────────────────────────────────────────────────────────────────┤
│  第二层：置信度掩码 (conf_msk)                                    │
│  weight[:, msk] = 0.0                                           │
│  作用：在BA优化中完全排除手部区域的约束                             │
│  效果：即使有残余错误约束，也不影响相机位姿和深度的优化              │
└─────────────────────────────────────────────────────────────────┘
```

### 9.6 潜在的局限性与改进方向

#### 局限1：手部遮挡大面积背景时

当手部占据图像较大比例（>30%），或手部遮挡了关键特征区域（如唯一纹理丰富的区域）时：
- 可用特征点大幅减少
- SLAM可能无法找到足够的特征匹配
- 相机位姿估计精度下降

**改进方向**：对手部区域进行背景inpainting，恢复被遮挡的纹理信息

#### 局限2：黑色区域边缘效应

手部轮廓边缘处，黑色像素与背景像素的急剧变化可能导致：
- 特征提取器在边缘处产生异常响应
- 光流估计在边缘处不稳定

**改进方向**：使用渐变掩码（边缘处平滑过渡）替代硬掩码

#### 局限3：深度估计的空洞

手部区域的深度估计完全缺失，可能影响：
- 后续3D重建的完整性
- 深度图上采样时的边界伪影

**改进方向**：对手部区域使用Metric3D单独估计深度，与SLAM深度融合

### 9.7 完整数据流总结

```
视频帧
  │
  ├─[1] YOLO检测+追踪
  │     输出: model_boxes.npy (手部边界框)
  │
  ├─[2] HAWOR姿态估计 + MANO正演 + PyTorch3D渲染
  │     输出: model_masks.npy (精确手部轮廓掩码, bool)
  │           cam_space/*.json (相机空间手部参数)
  │
  ├─[3] Masked SLAM
  │     │
  │     ├─ preprocess_masks():
  │     │    img_msks  (H×W):    用于图像涂黑
  │     │    conf_msks (H//8×W//8): 用于BA权重
  │     │
  │     ├─ 逐帧处理:
  │     │    image = image * (img_msk < 0.5)   ← 手部涂黑
  │     │    droid.track(mask=conf_msk)         ← 传入权重掩码
  │     │
  │     ├─ BA优化:
  │     │    weight[:, msk>0] = 0.0             ← 手部不参与优化
  │     │
  │     └─ 输出: 相机轨迹 traj (不受手部运动干扰)
  │
  ├─[4] 尺度估计
  │     robust = (msk<0.5) * (...)              ← 排除手部深度
  │     输出: hawor_slam_w_scale_*.npz
  │
  └─[5] cam→world转换 + Infilling + 可视化
```

### 9.8 结论

**"涂黑不填充"是HaWoR经过深思熟虑的设计选择，而非简化处理**：

1. **安全性**：双层掩码机制确保手部运动完全不干扰相机位姿估计
2. **简洁性**：无需额外的inpainting模型，降低计算开销和实现复杂度
3. **鲁棒性**：避免了inpainting引入的潜在错误约束
4. **实用性**：在egocentric视频场景中，手部占比有限，背景特征充足

这种设计在大多数egocentric视频场景下是充分且有效的。仅在极端场景（手部遮挡大面积关键背景区域）下可能需要更复杂的inpainting策略。
