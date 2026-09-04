# demov2 管线与 depth_with_mano / MANO 3D 重建 完整总结

> 整理自问答记录 Q1–Q16（详见 `docs/questions.md`），对应代码状态：2026-08-22。
> 每个阶段按【干什么 / 输入 / 方法 / 输出】四要素描述。

---

## 〇、术语表（先读这个）

| 术语 | 含义 |
|---|---|
| **verts / 顶点 (vertices)** | 3D 网格表面上的采样点。一只 MANO 手 = 778 个顶点连成的三角网格蒙皮，即"用 778 个 3D 点表示整只手的表面" |
| **MANO** | 参数化的数字手模型：输入姿态参数(15个指关节角度+手腕朝向)和形状参数(betas, 10个数描述手掌宽窄手指长短)，输出摆好姿势的 778 顶点 3D 手 |
| **前向 (forward)** | 把参数喂给 MANO 得到 3D 手的过程（拧旋钮→摆造型） |
| **joints / 关节** | 手上的关键点位置。MANO 自带 16 个（腕+15指节），HaWoR 再加 5 个指尖凑成 21 个，排列顺序遵循 OpenPose 约定 |
| **伪相机** | 阶段②渲染结果时使用的内部虚拟针孔相机（焦距600、光心(960,540)、1920×1080）。它不是真实拍摄相机的参数——真实焦距是另一份文件 est_focal.txt。所有掩码/顶点都画在这台虚拟相机的像平面上，彼此自洽 |
| **mask_focal.txt** | 记录阶段②当年实际使用的渲染焦距（通常 600）。是"顶点出生坐标系"的身份证 |
| **est_focal.txt** | SLAM 使用的相机焦距文件（如 121 视频=1152）。数值来自历史写入；当前主流程只读取不搜索（自动搜索函数 search_focal_length 存在但未接线） |
| **c2w / w2c** | camera-to-world / world-to-camera：相机坐标系与世界坐标系之间的互转变换矩阵 |
| **pred_valid** | npz 里标记"某手某帧参数有效"的布尔数组。注意：infiller 补过的帧也会被标 True，不代表真的检出过 |
| **缓存命中** | 每个阶段的产物都会落盘；重跑同一视频时发现文件已存在就直接读取，跳过耗时几分钟的 GPU 推理（断点续跑设计） |
| **RAS** | 复现场景项目(ReplicateAnyScene)侧的深度相机数据：688×384 的深度图 + 内外参，与 demov2 的输入 RGB 视频同一视角、已配准 |

---

## 一、管线 A：demov2 整体流程（六阶段）

> 主流程 demov2.py:505-516 只有三行核心调用：
> `detect_track_video` → `hawor_motion_estimation` → `hawor_slam`(缺缓存才跑) → `hawor_infiller`
> 下面按真实代码逐阶段拆解。

### 阶段① 检测与跟踪（detect_track_video）

```
干什么 : 找到每一帧画面里的手在哪里(bbox框)、是左手还是右手
输入   : 视频抽帧图像 extracted_images/*.jpg
方法   : lib/pipeline/tools.py
         ① ultralytics YOLO 逐帧检测手（阈值0.2）:
            输出边界框 + 左右手分类(det_handedness) + 置信度
         ② ByteTrack 跨帧关联: 相邻帧位置重叠高的框判为同一只手,
            串成轨迹(tracklet)
输出   : tracks_*/model_tracks.npy   轨迹字典
           每条轨迹 = [{frame帧号, det_box框坐标, det_handedness左右手, det有效位}, ...]
缓存   : 该 npy 已存在则整阶段跳过
```

### 阶段② 运动估计（hawor_motion_estimation，HAWOR 网络）——核心 AI 步骤

```
干什么 : 从图像恢复每只手每一帧的 3D 姿态（MANO 参数），并实例化成网格
输入   : ① 阶段①的轨迹 model_tracks.npy
         ② 一个焦距 img_focal，来源优先级(hawor_video.py:54-64):
            命令行 --img_focal → est_focal.txt → 都没有则默认 600 并写盘
内部流程(hawor_video.py:77-236):
  1) 轨迹按左右手多数票分成两组
  2) bbox 序列在丢帧处插值补洞(interpolate_bboxes:140), 补后视为有效
  3) 按"连续有效帧段"切成 chunk
  4) 每个 chunk 调 HAWOR 网络(:167):
       model.inference(图像块, bbox块, img_focal, img_center, do_flip)
       - 左手图像先水平翻转按右手处理(do_flip), 结果再翻回(:179-185)
       - 网络内部: 图像特征编码 + 跨帧时序建模 → 回归每帧
         pred_rotmat(腕+15指节旋转矩阵)、pred_trans(空间位移)、pred_shape(10维手型)
       ⚠️ img_focal 是网络的输入条件: 位移是在这个焦距假设下预测的,
          所以产物坐标系与该焦距绑定 —— 这就是"伪相机系"的真正由来
  5) run_mano/run_mano_left 前向成网格(process.py:88-91, MANO顶点+transl):
       vertices(T,778,3) + joints(T,21,3)【相机空间】
     每段参数先存 cam_space/<手>/<起>_<止>.json(:191, ⭐真检出的凭证)
  6) pyrender 用恒等位姿相机逐帧渲染手掩码(:213-224),
     填入全零底版的 model_masks/verts/joints 数组
输出   : tracks_*/model_verts.npy   (2,T,778,3)  未检出帧保持全0
         tracks_*/model_joints.npy  (2,T,21,3)
         tracks_*/model_masks.npy   (T,1080,1920) bool
         tracks_*/frame_chunks_all.npy        每只手的检出区段
         tracks_*/mask_focal.txt    = 本阶段实际用的 img_focal(:234)
                                    ← 顶点坐标系的"身份证", 之后不再变
特性   : chunk 内有时序建模, chunk 间相互独立; 无后处理平滑
缓存   : 四个 npy 齐全则跳过网络推理(:70-73)
```

### 阶段③ 焦距来源（供 SLAM 使用）

```
干什么 : 确定 SLAM 用的相机内参焦距
现实情况(当前代码):
  hawor_slam.py:71-84 读数优先级:
    --img_focal 参数 → est_focal.txt → 默认600
  est_focal.txt 的数值是历史写入的(如 121视频=1152, 7视频=960)
代码库里备有自动搜索函数但当前主流程未调用:
  search_focal_length(masked_droid_slam.py:237)
  方法: 在 [500,1500) 每100一档共10个候选, 每档跑一次迷你SLAM
        (隔10帧采样、最多50帧), 取"SLAM 重投影误差最小"的焦距
        docstring 原话: "Search for a good focal length by SLAM reprojection error"
要点   : 这里的焦距只影响 SLAM/世界系; 不回改阶段②产物
        （阶段②的坐标系已固化在 mask_focal.txt, 两者从此独立）
```

### 阶段④ SLAM 相机轨迹（hawor_slam.py）

```
干什么 : 解算拍摄相机的运动轨迹, 建立场景世界坐标系和绝对尺度
输入   : 全部视频帧 + 手部掩码(遮蔽非手区域) + 阶段③焦距标定
内部流程:
  1) run_slam(imgfiles, masks, calib) = masked DROID-SLAM(:89)
     经典单目视觉里程计+回环: 跨帧特征关联 → 联合优化位姿与深度
     → traj(T,7) 每帧相机位姿 c2w + disps 视差图【只有相对深度】
  2) Metric3D 单目度量深度网络(:96-97, 加载 vit_large 权重)
     对每帧预测"带真实米数"的深度图
  3) 尺度对齐 est_scale: 计算 DROID相对深度 与 Metric3D绝对深度
     的比例 → 全局 scale（单目 SLAM 天生不知道绝对大小, 靠这一步补上）
输出   : SLAM/hawor_slam_w_scale_*.npz {traj(T,7), scale, disps, ...}
意义   : 世界坐标系由此定义; "手在第几帧相机前"从此可以变成
         "手在场景世界的哪里"(经 c2w 变换)
缓存   : npz 已存在则跳过(:511-513)
```

### 阶段⑤ Infiller 参数补全（hawor_infiller）

```
干什么 : 给漏检/遮挡帧补 MANO 参数, 得到全程连续双手参数序列
输入   : 阶段②的稀疏参数(frame_chunks_all 指明哪些帧有真值)
方法   : 双手协同先验 Transformer(120帧窗口):
         训练时学过大量"双手协作"数据, 能按一只手的动作推断另一只手
         流程: 转canonical系 → 缺失帧lerp/slerp初始化 → 推理整个窗口
               → 只把缺失位置替换为预测值
         ⚠️ 副作用: 会"脑补" —— 某只手从头到尾没出现也会编一套参数
            (视频7的右手即如此), 且把这些帧 pred_valid 置 True(虚高)
输出   : reconstruction/hawor_results_*.npz ★下游核心参数包
           pred_trans/pred_rot/pred_hand_pose/pred_betas 【世界系坐标!】
           pred_valid (2,T)、R_c2w/t_c2w 每帧外参、img_focal
         cam_space/<手>/*.json  ⭐唯一如实记录"哪些帧真检出"的地方
```

### 分支：可视化（默认关闭）

```
gen_vis = --with_vis 时才执行
输出到 : visualization/{vis_cam, vis_world, vis_verify, combined_render}
内容   : 纯人眼检查视频, 无数据价值, 占总产出约40%体积
```

### 阶段⑥ RAS 深度叠加（始终执行 → 进入管线 B）

---

## 二、管线 B：depth_with_mano 生成与对齐

> 目的：把手画到 RAS 深度图上，人眼验证"估计的手"与"真实场景"是否吻合。

```
步骤1  选投影焦距 render_focal
        读 output/<视频>/tracks_*/mask_focal.txt —— 顶点当年用什么焦距生成,
        现在就用什么焦距投影（谁生成的就用谁的坐标系, 必然对齐）
        若文件不存在(老版本缓存): 拿 est_focal 的值暂时顶替,
        同时打印 WARNING 提醒"可能与顶点坐标系不符, 请看自检结果";
        且不把这个临时值写进任何文件, 防止日后被误当成真值

步骤2  加载顶点数据
        model_verts.npy / model_joints.npy（阶段②产物, 与掩码同源）

步骤3  帧对齐 (ras_to_hw)
        depth 帧数 ≠ 视频帧数（如 100 vs 600, 抽帧率不同）
        对每个 depth 帧, 在视频帧里找"内容最像"的一帧（缩略图像素差MSE最小）
        找不到可靠匹配时退化为按时间比例取索引

步骤4  逐帧绘制（每帧独立计算, 无平滑无插值）
        绘制前三项前置检查（原名"守卫", 任一不满足 → 该帧留空不画）:
          a. pred_valid 为真      —— infiller 可能把没检出的帧也标有效
          b. 数值不含 NaN          —— 数据本身不能残缺
          c. 顶点不全为零          —— 未检出帧顶点是全0数组,
                                      不挡会把骨架画成一团聚在图像中心
        通过后, 两步投影:
          第一步 3D→RGB像素:  u = f·X/Z + 960 ,  v = f·Y/Z + 540
                               (f=render_focal, 960/540 是1920×1080的中心)
          第二步 RGB→depth像素: ×(688/1920, 384/1080)
                               (depth与RGB同视角同画面, 只差分辨率, 直接等比缩放)
        画法: 三角形网格填色(蓝=左手, 绿=右手) + 黄色骨架连线与关节点

步骤5  输出
        ReplicateAnyScene/output_v2/<同名>_vggt_omega/depth_with_mano/
          <depth帧号>.png  +  合成 .mp4

步骤6  自动自检（每次渲染完自动跑）
        把 21 个关节的投影点与检测掩码(model_masks.npy)比对:
        统计"落在手部区域内"的比率 → ≥70% 打印 OK, 否则 WARNING
        作用: 焦距配错/数据异常等问题会被数字量化暴露, 不靠肉眼猜
```

**为什么能对齐（两条根本原因）**
1. 顶点和检测掩码在同一台伪相机下生成 → 用同一焦距投影必然落在掩码内（实测 99~100%）。
2. depth 图与 RGB 视频同一视角拍摄且已配准 → 像素等比缩放即可，不需要任何重投影。

---

## 三、管线 C：MANO 3D 坐标重建（`mano_ras_reconstruction.py`）

> 目的：与管线 B 方向相反 —— B 把 3D 画到图上；C 从 depth 图上把每个点的**真实距离查回来**，得到带真实深度的 3D 手。

```
步骤1  读参数（旋钮读数）
        hawor_results npz 的 pred_rot/pred_hand_pose/pred_betas/pred_trans
步骤2  MANO 前向（拧旋钮摆出手）
        smplx 加载左右手模型 → LBS 蒙皮 → 778 顶点 3D 手, 加上 pred_trans 定位
步骤3  投影定位（同管线B的两步投影）
        f=600 @1920×1080 → 等比缩放到 688×384
        → 知道 778 个点分别落在 depth 图哪个像素
步骤4  逐点查真实深度 ★核心
        depth 像素灰度值 × 0.001 = 该点真实距离(米)
        查不到(黑/出界) → 该顶点作废(NaN)
步骤5  反算 3D 坐标
        X=(u-cx)·Z/fx,  Y=(v-cy)·Z/fy,  Z=查到的深度
        → RAS 相机坐标系下的 3D 顶点
步骤6  转世界系 + 提取关节
        extrinsics(w2c) 反变换 → RAS 世界系坐标
        J_regressor(16×778加权) 提 16 关节 + 取 5 个固定指尖顶点 = 21 关节
        整帧全 NaN → invalid（绝不输出假的零坐标）

输出 reconstruction/mano_ras_reconstruction_<起>_<止>.npz:
  verts_ras_{L/R}        (T,778,3)  RAS相机系顶点
  joints_ras_{L/R}       (T,21,3)   RAS相机系21关节
  verts/joints_ras_world_{L/R}      RAS世界系（经extrinsics转换）
  valid_{L/R}            (T,)       该帧是否查到了有效深度
  ri_to_hw               depth↔视频帧映射;  fx/fy/cx/cy/depth_scale 内参备份
```

**意义**：npz 里原本手的 Z 只是网络估的；经过管线 C，每个点的 Z 都是 depth 图量出来的物理真值。

---

## 四、两个 npz 的区别

| | hawor_results（阶段⑤产物） | mano_ras_reconstruction（管线C产物） |
|---|---|---|
| 本质 | MANO 参数源 + SLAM 相机轨迹 | 前向并查深后的几何结果 |
| 坐标系 | HaWoR SLAM 世界系（RGB 视频侧） | RAS depth 相机系 + RAS 世界系 |
| 几何数据 | 无，需自己前向 | verts(T,778,3) + joints(T,21,3) 双坐标系 |
| RAS 信息 | 完全没有 | 内参、depth_scale、ri_to_hw 帧映射 |
| valid 语义 | infiller 置位（真检出+脑补都算 True） | depth 反投影逐帧真实成功与否 |

互补不替代：要重新投影参数用前者；直接拿带真实深度的 3D 手用后者。

## 五、产物体积与价值分级

| 分级 | 目录 | 体积(113帧/600帧) |
|---|---|---|
| B 数据 | cam_space/, SLAM/, reconstruction/, est/mask_focal.txt | ~19M / ~43M |
| C 缓存 | tracks_*/, extracted_images/（换重跑省推理） | ~230M / ~1.2G |
| A 可视化(默认关) | visualization/ 四件套 | ~175M / ~1.0G |
| depth_with_mano(RAS侧) | 始终生成 | ~10M / ~20M |

## 六、常用命令

```bash
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR

# 标准跑法（默认只出数据 + depth_with_mano）
/mnt/data/lza/conda_envs/hawor/bin/python demov2.py --video_path /path/to/x.mp4

# 需要四个可视化时加 --with_vis
/mnt/data/lza/conda_envs/hawor/bin/python demov2.py --video_path x.mp4 --with_vis

# 3D 坐标重建（demov2 跑完后）
/mnt/data/lza/conda_envs/hawor/bin/python mano_ras_reconstruction.py --video-name <视频名>
```

注意事项：
- demov2 必须在 `HaWoR/` 目录下运行（checkpoint 权重是相对路径 `./weights/...`）
- depth_with_mano 需要 RAS 目录 `ReplicateAnyScene/output_v2/<同名>_vggt_omega/` 先有 depth/intrinsics/extrinsics
