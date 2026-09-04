"""
mano_ras_pipeline.py
====================
合并改进版: 一次运行同时产出
  (1) depth_with_mano/*.png             —— MANO 网格投影到 RAS depth 图 (人工核对用)
  (2) mano_ras_3d.npz                   —— MANO 投影像素查 RAS depth 反投影的 3D 坐标

================================================================================
整体管线 (Pipeline)
================================================================================
上游 demov2 主流程 (六阶段, 逐帧产出 MANO 与掩码, 全部在"伪相机系"内自洽):
  ① detect_track_video   YOLO+ByteTrack 检测/跟踪 -> tracks_*/model_tracks.npy
  ② hawor_motion_estimation  HAWOR 逐帧拟合 MANO 参数并前向成网格
        - 产物都画在伪相机(1920x1080, 焦距 = mask_focal.txt, 通常 600)
        - 保存"相机空间"网格: tracks_*/model_verts.npy (2,T,778,3) 与
                              tracks_*/model_joints.npy (2,T,21,3)
        - 保存检测掩码:       tracks_*/model_masks.npy   (T,1080,1920) bool
  ③ 焦距搜索 est_focal      只影响 SLAM, 不改阶段②产物
  ④ SLAM 世界系 c2w + scale
  ⑤ infiller 参数补全        -> reconstruction/hawor_results_*.npz
                            ⚠️ 该 npz 的 pred_trans/pred_rot 是"世界系"
                               (已经过 SLAM 的 c2w 与 R_x 变换)
  ⑥ RAS depth_with_mano 叠加 (demov2 内置, 用阶段②的相机空间 saved verts)
                              所以 hand 网格与检测掩码天然对齐 ~99-100%

本脚本 mano_ras_pipeline.py (管线 B, 离线重跑/合并):
  输入:
    - HaWoR 侧: reconstruction/hawor_results_*.npz  (MANO 参数)
    - HaWoR 侧: tracks_*/model_verts.npy + model_joints.npy (相机空间网格,
      与检测掩码对齐, 覆盖率 ~99% 的来源)
    - RAS 侧:   ReplicateAnyScene/output_v2/<video>_vggt_omega/
                intrinsic.txt (fx/fy/cx/cy)  + depth/*.png (uint16, 0.001m)
    - 顶点"出生焦距": mask_focal.txt (通常 600)
  流程:
    1. 帧映射: 内容反查(RAS ri -> HaWoR 帧, 缩略图 MSE) 为主, 回退线性比例
    2. 取 MANO 3D 顶点: 优先保存的相机空间 model_verts/joints(保证对齐掩码),
       缺帧回退用 npz 参数 run_mano 重建
    3. 投影: HaWoR 伪相机 3D @ render_focal -> 1920x1080 -> 等比缩放 to 688x384
    4. 输出1: 逐帧把 mesh 三角(soup) + 21 关节画到 RAS depth 伪彩图上
    5. 输出2: 投影像素查 RAS depth, 反投影回深度相机 3D (X=(u-cx)z/fx...)
       -> 保存 mano_ras_3d.npz (mesh_3d + joints_3d_ras + 插值 joints_T)
  校验(手区域覆盖率): 把相机空间顶点 @render_focal 前投到 1080x1920,
     统计落在检测掩码(model_masks.npy) 内的比例 -> 期望 ~99% 以上。
     若 <80% 说明顶点/焦距/帧映射错位, 请核对 mask_focal.txt 与坐标帧来源。

相对旧脚本的修正(为什么覆盖率从 99% 掉到 70/40%):
  * 根因: 旧版用保存的"相机空间" model_verts/model_joints 投影(天然对齐掩码,
    覆盖率 ~99-100%); 合并时误改成用 reconstruction npz 的"世界系"参数
    run_mano 重建后再投影, 世界系与伪相机系未对齐 -> 覆盖率掉到 ~70/40%。
    本次改回优先用保存的相机空间网格, 保证覆盖率回到 ~99%。
  * 其余修正保留: 统一投影焦距读 mask_focal.txt; RAS 内参从 intrinsic.txt 读;
    帧映射优先内容反查; 输出同时带 mesh 与 joints 并可合成 mp4。

用法:
  cd HaWoR
  python3 mano_ras_pipeline.py --video-name 7
  python3 mano_ras_pipeline.py --video-name 121_C5_CellPhone_161deg
  python3 mano_ras_pipeline.py --video-name 7 --smooth

输出:
  <ras_dir>/depth_with_mano/*.png (+ .mp4)
  <hawor_out>/mano_ras_3d.npz
"""
import argparse
import os
import sys
import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# 让 hawor.utils.process 里的 `from lib.models.mano_wrapper import MANO` 能解析
sys.path.insert(0, os.path.join(HERE, '..'))  # robot_world_ws/src

from hawor.utils.process import run_mano, run_mano_left, get_mano_faces

# round-trip geometry (RAS depth view 688x384)
W, H = 688, 384
DEPTH_SCALE = 0.001
# HaWoR 相机空间 (1920x1080 伪相机)
W0, H0 = 1920, 1080
cx0, cy0 = W0 / 2.0, H0 / 2.0
sx, sy = W / W0, H / H0

MANO_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGER_TIPS = {4, 8, 12, 16, 20}
COLOR_BY_HAND = {1: (0, 200, 0), 0: (200, 100, 0)}  # right=green, left=orange


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def load_intrinsic(ras_dir):
    intr = os.path.join(ras_dir, 'intrinsic.txt')
    if os.path.isfile(intr):
        raw = np.loadtxt(intr)
        return float(raw[0, 0]), float(raw[1, 1]), float(raw[0, 2]), float(raw[1, 2])
    sys.exit('[cfg] no intrinsic.txt at %s' % intr)


def load_mask_focal(hawor_out):
    """顶点出生焦距: 优先 tracks_*/mask_focal.txt, 否则 output/<name>/mask_focal.txt"""
    for root, _, files in os.walk(hawor_out):
        if 'mask_focal.txt' in files:
            return float(open(os.path.join(root, 'mask_focal.txt')).read().strip())
    # 找不到: 用 600 (HaWoR 默认, 与日志 "use default 600" 一致)
    print('[cfg] WARNING: no mask_focal.txt, use default 600')
    return 600.0


def infer_hands(hawor_out):
    cam = os.path.join(hawor_out, 'cam_space')
    hands = []
    if os.path.isdir(cam):
        for h in os.listdir(cam):
            if os.path.isdir(os.path.join(cam, h)):
                try:
                    hands.append(int(h))
                except ValueError:
                    pass
    if not hands:
        hands = [0]  # 默认左手(与 demov2 一致)
    return sorted(hands)


def find_tracks_dir(hawor_out):
    """定位 tracks_*_* 目录 (内含 model_verts.npy 等相机空间产物)"""
    for name in sorted(os.listdir(hawor_out)):
        if name.startswith('tracks_') and os.path.isdir(os.path.join(hawor_out, name)):
            return os.path.join(hawor_out, name)
    return None


def load_saved_meshes(tracks_dir):
    """读阶段②保存的"相机空间"网格/掩码 (与检测掩码对齐, 覆盖率 ~99% 的来源)"""
    if not tracks_dir:
        return None, None
    vp, jp = os.path.join(tracks_dir, 'model_verts.npy'), os.path.join(tracks_dir, 'model_joints.npy')
    if not (os.path.isfile(vp) and os.path.isfile(jp)):
        return None, None
    verts = np.load(vp)    # (2,T,V,3)
    joints = np.load(jp)   # (2,T,21,3)
    return verts, joints


def build_config(args):
    video_name = args.video_name
    hawor_out = args.hawor_out or os.path.join(HERE, 'output', video_name)
    ras_dir = args.ras_dir or os.path.join(
        HERE, '..', 'ReplicateAnyScene', 'output_v2', f'{video_name}_vggt_omega')

    fx, fy, cx, cy = load_intrinsic(ras_dir)
    render_focal = args.focal if args.focal else load_mask_focal(hawor_out)

    # 加载 hawor_results
    reshots = [f for f in os.listdir(os.path.join(hawor_out, 'reconstruction'))
               if f.startswith('hawor_results') and f.endswith('.npz')]
    if not reshots:
        sys.exit('[cfg] no hawor_results_*.npz under %s/reconstruction' % hawor_out)
    r = np.load(os.path.join(hawor_out, 'reconstruction', sorted(reshots)[-1]))
    pred_trans = r['pred_trans']
    pred_rot = r['pred_rot']
    pred_hand = r['pred_hand_pose']
    pred_betas = r['pred_betas']
    pred_valid = r['pred_valid']
    try:
        start_idx, end_idx = int(r['start_idx']), int(r['end_idx'])
    except (KeyError, IndexError):
        start_idx, end_idx = 0, pred_rot.shape[1] - 1
    fname = sorted(reshots)[-1]

    hands = infer_hands(hawor_out)
    tracks_dir = find_tracks_dir(hawor_out)
    sv, sj = load_saved_meshes(tracks_dir)
    mask_path = os.path.join(tracks_dir, 'model_masks.npy') if tracks_dir else None
    saved_masks = np.load(mask_path) if mask_path and os.path.isfile(mask_path) else None

    print('[cfg] video=%s  hands=%s  focal=%g  intrin=(%.2f,%.2f)' % (
        video_name, hands, render_focal, fx, fy))
    print('[cfg] ras_dir=%s  hawor_out=%s' % (ras_dir, hawor_out))
    print('[cfg] hawor_results=%s  frames=%d' % (fname, pred_rot.shape[1]))
    print('[cfg] saved camera-space mesh=%s masks=%s' % (
        (sv.shape if sv is not None else None),
        (saved_masks.shape if saved_masks is not None else None)))

    return dict(video_name=video_name, ras_dir=ras_dir, hawor_out=hawor_out,
                fx=fx, fy=fy, cx=cx, cy=cy, render_focal=render_focal,
                pred_trans=pred_trans, pred_rot=pred_rot, pred_hand=pred_hand,
                pred_betas=pred_betas, pred_valid=pred_valid,
                saved_verts=sv, saved_joints=sj, saved_masks=saved_masks,
                start_idx=start_idx, end_idx=end_idx, hands=hands,
                hawor_frames=pred_rot.shape[1])


# ---------------------------------------------------------------------------
# MANO 前向 (网格, HaWoR 伪相机空间)
# ---------------------------------------------------------------------------
def build_meshes(cfg):
    """对每只手返回 (T,778,3) 顶点、(T,21,3) 关节。

    顶点来源(重要, 决定手区域覆盖率与手是否画得出):
      * 优先: tracks_*/model_verts.npy + model_joints.npy —— 阶段②在伪相机系
        (focal=mask_focal) 直接保存的网格, 与检测掩码天然对齐, 覆盖率 ~99%。
        但若 saved 有效帧太稀疏(如 15号仅4帧), 手几乎画不出 -> 需回退。
      * 回退: 用 reconstruction npz 参数 run_mano 重建 (pred_valid 通常覆盖大量帧,
        如 15号50帧)。npz 是"世界系", 与伪相机系存在位姿差, 覆盖率可能下降,
        但在 saved 缺失/稀疏时是唯一能画出手的来源。
    判定: saved 有效帧占总帧比例 >=30% 时用 saved(对齐好), 否则回退 npz(帧数全)。
    缺帧则置 NaN。
    """
    out = {}
    saved_verts, saved_joints = cfg.get('saved_verts'), cfg.get('saved_joints')
    T = cfg['hawor_frames']
    for hid in cfg['hands']:
        use_saved = False
        if saved_verts is not None and saved_verts.shape[0] > hid and saved_verts.shape[1] == T:
            sv_valid = ~np.all(saved_verts[hid] == 0, axis=(1, 2))
            ratio = sv_valid.mean() if T else 0.0
            use_saved = ratio >= 0.3
        if use_saved:
            # 1) 相机空间 saved mesh: 与掩码对齐, 覆盖率最优
            verts = saved_verts[hid].astype(np.float64).copy()
            joints = saved_joints[hid].astype(np.float64).copy() if saved_joints is not None else None
            # 全零帧(未检出)判为缺帧
            valid = sv_valid
            if joints is not None:
                valid &= ~np.all(joints == 0, axis=(1, 2))
            verts[~valid] = np.nan
            if joints is not None:
                joints[~valid] = np.nan
            if joints is None:
                joints = np.full_like(verts[:, :21], np.nan)
            out[hid] = (verts, joints)
            print('[mesho] hand %d verts %s (saved camera-space, valid=%d/%d, cov-pref)' % (
                hid, verts.shape, valid.sum(), T))
            continue

        # 2) 回退: npz 世界系参数重建 (pred_valid 覆盖大量帧)
        if hid == 1:
            o = run_mano(torch.from_numpy(cfg['pred_trans'][1:2]).float(),
                         torch.from_numpy(cfg['pred_rot'][1:2]).float(),
                         torch.from_numpy(cfg['pred_hand'][1:2]).float(),
                         betas=torch.from_numpy(cfg['pred_betas'][1:2]).float())
        else:
            o = run_mano_left(torch.from_numpy(cfg['pred_trans'][0:1]).float(),
                              torch.from_numpy(cfg['pred_rot'][0:1]).float(),
                              torch.from_numpy(cfg['pred_hand'][0:1]).float(),
                              betas=torch.from_numpy(cfg['pred_betas'][0:1]).float())
        verts = o['vertices'][0].cpu().numpy().astype(np.float64)   # (T,778,3)
        joints = o['joints'][0].cpu().numpy().astype(np.float64)    # (T,21,3)
        valid = cfg['pred_valid'][hid] if cfg['pred_valid'].shape[0] > hid else np.ones(T, bool)
        verts[~valid] = np.nan
        joints[~valid] = np.nan
        out[hid] = (verts, joints)
        print('[mesho] hand %d verts %s (npz-param rebuild, valid=%d/%d)' % (
            hid, verts.shape, int(valid.sum()), T))
    return out


# ---------------------------------------------------------------------------
# 手区域覆盖率校验: 相机空间顶点 @render_focal 前投, 落在检测掩码内的比例
# ---------------------------------------------------------------------------
def report_coverage(cfg, meshes):
    """前投 saved/相机空间 顶点到检测掩码 (model_masks.npy), 期望 >=99%。
    仅当保存了 model_masks.npy 时有效; 否则打印说明跳过。"""
    masks = cfg.get('saved_masks')
    sv, sj = cfg.get('saved_verts'), cfg.get('saved_joints')
    if masks is None or (sv is None and sj is None):
        print('[cov] SKIP: need tracks_*/model_masks.npy + model_verts.npy/model_joints.npy')
        return
    W0_, H0_ = masks.shape[2], masks.shape[1]
    F = cfg['render_focal']
    for hid in cfg['hands']:
        if sv is None:
            continue
        verts = sv[hid]  # (T,778,3) 相机空间顶点
        n_arrive = n_hit = 0
        for i in range(cfg['hawor_frames']):
            p = verts[i]
            if np.isnan(p).any() or not p.any():
                continue
            z = np.maximum(p[:, 2], 1e-6)
            # 用伪相机 cv0(960,540) 等比: 但这里直接在掩码尺寸上投,
            # 掩码与顶点同相机系同尺寸 —— 直接以掩码中心为光心
            u = F * p[:, 0] / z + W0_ / 2.0
            v_ = F * p[:, 1] / z + H0_ / 2.0
            u = u.astype(int); v_ = v_.astype(int)
            ok = (u >= 0) & (u < W0_) & (v_ >= 0) & (v_ < H0_)
            if not ok.any():
                continue
            m = masks[i]
            n_arrive += ok.sum()
            n_hit += int(m[v_[ok], u[ok]].sum())
        if n_arrive:
            rate = n_hit / n_arrive
            print('[cov] hand %d: forward-proj onto detect-mask coverage = %.1f%%  (verts=%d, hit=%d)' % (
                hid, rate * 100, n_arrive, n_hit))
            if rate < 0.8:
                print('[cov] WARNING: coverage<80% → 顶点/焦距/坐标帧可能错位, 请核对 mask_focal.txt 与顶点来源')
        else:
            print('[cov] hand %d: no valid verts to evaluate' % hid)


# ---------------------------------------------------------------------------
# 投影: HaWoR 伪相机 -> 688x384 像素
# ---------------------------------------------------------------------------
def project3d(pts, focal):
    z = np.maximum(pts[:, 2:3], 1e-6)
    u = focal * pts[:, 0:1] / z + cx0
    v = focal * pts[:, 1:2] / z + cy0
    return np.hstack([u * sx, v * sy])   # (N,2) 688x384 坐标


# ---------------------------------------------------------------------------
# RAD depth 反投影 -> RAS depth 相机 3D (米)
# ---------------------------------------------------------------------------
def reproject(depth, uv, fx, fy, cx, cy):
    u = uv[:, 0].astype(int); v = uv[:, 1].astype(int)
    u = np.clip(u, 0, W - 1); v = np.clip(v, 0, H - 1)
    z = depth[v, u].astype(np.float64) * DEPTH_SCALE
    X = (u.astype(np.float64) - cx) * z / fx
    Y = (v.astype(np.float64) - cy) * z / fy
    return np.stack([X, Y, z], -1)


# ---------------------------------------------------------------------------
# 帧映射: 内容反查 (RAS ri -> HaWoR 帧), 回退线性
# ---------------------------------------------------------------------------
def build_ri_to_hw(cfg):
    img_dir = os.path.join(cfg['hawor_out'], 'extracted_images')
    if not os.path.isdir(img_dir):
        return None
    hw_small = {}
    for i in range(cfg['hawor_frames']):
        im = cv2.imread(os.path.join(img_dir, f'{i:04d}.jpg'))
        if im is not None:
            hw_small[i] = cv2.resize(im, (64, 36))
    if not hw_small:
        return None
    rcdir = os.path.join(cfg['ras_dir'], 'color')
    if not os.path.isdir(rcdir):
        return None
    ri_to_hw = {}
    for f in sorted(os.listdir(os.path.join(cfg['ras_dir'], 'depth')),
                    key=lambda x: int(x.split('.')[0])):
        if not f.endswith('.png'):
            continue
        ri = int(f.split('.')[0])
        im = None
        for ext in ('.jpg', '.jpeg'):
            p = os.path.join(rcdir, f.split('.')[0] + ext)
            if os.path.isfile(p):
                im = cv2.imread(p); break
        if im is None:
            continue
        s = cv2.resize(im, (64, 36))
        ri_to_hw[ri] = min(hw_small, key=lambda k: float(
            np.mean((hw_small[k].astype(float) - s.astype(float)) ** 2)))
    print('[map] content-mapped %d RAS frames' % len(ri_to_hw))
    return ri_to_hw


# ---------------------------------------------------------------------------
# 输出1: depth_with_mano
# ---------------------------------------------------------------------------
def render_depth_with_mano(cfg, meshes, ri_to_hw):
    faces = np.asarray(get_mano_faces(), dtype=int)
    out_dir = os.path.join(cfg['ras_dir'], 'depth_with_mano')
    os.makedirs(out_dir, exist_ok=True)

    depth_files = sorted([f for f in os.listdir(os.path.join(cfg['ras_dir'], 'depth'))
                          if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
    T = cfg['hawor_frames']
    max_ri = int(depth_files[-1].split('.')[0]) if depth_files else 0

    fb = (T - 1) / max(1, max_ri)

    def to_visible(img):
        if img.dtype == np.uint16:
            mn, mx = float(img.min()), float(img.max())
            norm = np.clip((img.astype(np.float32) - mn) / (mx - mn + 1e-8), 0, 1)
            return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return img

    n_rendered = 0
    for f in depth_files:
        ri = int(f.split('.')[0])
        hwfi = int(np.clip(round(ri_to_hw.get(ri, ri * fb)), 0, T - 1)) if ri_to_hw \
            else int(np.clip(round(ri * fb), 0, T - 1))
        d = cv2.imread(os.path.join(cfg['ras_dir'], 'depth', f), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        overlay = to_visible(d)

        for hid, color in COLOR_BY_HAND.items():
            if hid not in meshes:
                continue
            verts, joints = meshes[hid]
            if hwfi >= verts.shape[0]:
                continue
            if np.isnan(verts[hwfi]).any():
                continue
            v2 = project3d(verts[hwfi], cfg['render_focal']).astype(int)
            j2 = project3d(joints[hwfi], cfg['render_focal']).astype(int)
            # mesh fill (triangle soup)
            for tri in faces:
                pts = v2[tri]
                if np.all(pts >= 0) and np.all(pts < [W, H]):
                    cv2.fillPoly(overlay, [pts], color)
            # skeleton: 沿用旧版 7 视频风格 (彩色线, 与 demov2 一致)
            for a, b in MANO_SKELETON:
                pa, pb = tuple(j2[a]), tuple(j2[b])
                if all(0 <= p < W for p in (pa[0], pb[0])) and all(0 <= p < H for p in (pa[1], pb[1])):
                    cv2.line(overlay, pa, pb, color, 2)
            for k in range(21):
                pk = tuple(j2[k])
                if 0 <= pk[0] < W and 0 <= pk[1] < H:
                    c = (0, 0, 255) if k in FINGER_TIPS else (0, 255, 255)  # 指尖红, 关节点青(同7号)
                    cv2.circle(overlay, pk, 4, c, -1)
        cv2.imwrite(os.path.join(out_dir, f), overlay)
        n_rendered += 1
    print('[render] %d depth_with_mano -> %s/' % (n_rendered, out_dir))
    return depth_files


def make_video(cfg, depth_names):
    vis_dir = os.path.join(cfg['ras_dir'], 'depth_with_mano')
    out = os.path.join(vis_dir, 'depth_with_mano.mp4')
    writer = None
    cnt = 0
    for f in depth_names:
        img = cv2.imread(os.path.join(vis_dir, f))
        if img is None:
            continue
        if writer is None:
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), 15.0,
                                     (img.shape[1], img.shape[0]))
        for _ in range(3):
            writer.write(img)
        cnt += 1
    if writer:
        writer.release()
        print('[video] -> %s (%d frames)' % (out, cnt))
    else:
        print('[video] SKIP: no images')


# ---------------------------------------------------------------------------
# 输出2: mano_ras_3d 反投影
# ---------------------------------------------------------------------------
def extract_and_save(cfg, meshes, ri_to_hw, depth_names, smooth):
    from scipy.signal import savgol_filter
    T = cfg['hawor_frames']
    max_ri = int(depth_names[-1].split('.')[0]) if depth_names else 0
    fb = (T - 1) / max(1, max_ri)

    # 预载 RAS depth
    depth_map = {int(f.split('.')[0]): cv2.imread(
        os.path.join(cfg['ras_dir'], 'depth', f), cv2.IMREAD_UNCHANGED)
        for f in depth_names}

    mesh_3d_list, joints_3d_list, joint_valid, ri_list = [], [], [], []
    for f in depth_names:
        ri = int(f.split('.')[0])
        hwfi = int(np.clip(round(ri_to_hw.get(ri, ri * fb)), 0, T - 1)) if ri_to_hw \
            else int(np.clip(round(ri * fb), 0, T - 1))
        d = depth_map[ri]
        if d is None:
            continue
        ri_list.append(ri)

        hand = cfg['hands'][0]  # 取首个有效手 (7->0, 121->1); 后续可扩展两手
        verts, joints = meshes[hand]
        # mesh 3D
        if np.isnan(verts[hwfi]).any():
            mesh_3d_list.append(np.zeros((0, 3)))
        else:
            v2 = project3d(verts[hwfi], cfg['render_focal']).astype(int)
            v2 = np.clip(v2, 0, [W - 1, H - 1])
            mesh_3d_list.append(reproject(d, v2, cfg['fx'], cfg['fy'], cfg['cx'], cfg['cy']))
        # joints 3D
        if np.isnan(joints[hwfi]).any():
            joints_3d_list.append(np.full((21, 3), np.nan))
            joint_valid.append(False)
        else:
            j2 = project3d(joints[hwfi], cfg['render_focal']).astype(int)
            j2 = np.clip(j2, 0, [W - 1, H - 1])
            joints_3d_list.append(reproject(d, j2, cfg['fx'], cfg['fy'], cfg['cx'], cfg['cy']))
            joint_valid.append(True)

    ri = np.array(ri_list)
    valid = np.array(joint_valid)
    joints_3d = np.array(joints_3d_list)
    mesh_3d = np.array(mesh_3d_list, dtype=object)
    print('[extract] %d frames, joints valid=%d, mesh non-empty=%d' % (
        len(ri), valid.sum(), sum(m.shape[0] > 0 for m in mesh_3d_list)))

    if len(ri) == 0:
        print('[interp] SKIP: no frames')
        return

    # 插值到 T 帧 (归一化时间轴)
    t_ras = ri / max(float(ri.max()), 1.0)
    t_target = np.arange(T) / max(T - 1, 1)
    joints_T = np.full((T, 21, 3), np.nan)
    for k in range(21):
        for c in range(3):
            y = np.where(valid, joints_3d[:, k, c], np.nan)
            joints_T[:, k, c] = np.interp(t_target, t_ras, y, left=np.nan, right=np.nan)
    if smooth:
        js = joints_T
        for k in range(21):
            v = ~np.isnan(js[:, k, 0])
            if v.sum() < 7:
                continue
            idx = np.where(v)[0]
            wl = min(15, len(idx) if len(idx) % 2 == 1 else len(idx) - 1)
            try:
                js[idx, k] = savgol_filter(js[idx, k], window_length=wl, polyorder=3, axis=0)
            except Exception:
                pass
    print('[interp] %d frames -> %d; NaN frames: %d' % (
        len(ri), T, int(np.isnan(joints_T[:, 0, 0]).sum())))

    out_npz = os.path.join(cfg['hawor_out'], 'mano_ras_3d.npz')
    np.savez(out_npz,
             mesh_3d=mesh_3d,
             joints_3d_ras=joints_3d,
             joints_3d_ras_interp=joints_T,
             ri=ri, valid=valid,
             ri_to_hw=np.array([(k, v) for k, v in ri_to_hw.items()]) if ri_to_hw else None,
             fx=cfg['fx'], fy=cfg['fy'], cx=cfg['cx'], cy=cfg['cy'],
             render_focal=cfg['render_focal'], depth_scale=DEPTH_SCALE,
             hand=cfg['hands'][0], hawor_frames=T,
             note='joints/mesh: MANO forward (focal=%g) -> pixel -> RAS depth reprojection, RAS depth cam (m)' % cfg['render_focal'])
    print('[save] -> %s' % out_npz)


def main():
    p = argparse.ArgumentParser(description='MANO -> depth_with_mano + mano_ras_3d (combined)')
    p.add_argument('--video-name', type=str, required=True)
    p.add_argument('--ras-dir', type=str, default=None)
    p.add_argument('--hawor-out', type=str, default=None)
    p.add_argument('--focal', type=float, default=None,
                   help='投影焦距 (默认读 mask_focal.txt)')
    p.add_argument('--no-video', action='store_true')
    p.add_argument('--smooth', action='store_true')
    args = p.parse_args()

    cfg = build_config(args)
    meshes = build_meshes(cfg)
    report_coverage(cfg, meshes)
    ri_to_hw = build_ri_to_hw(cfg)
    depth_names = render_depth_with_mano(cfg, meshes, ri_to_hw)
    if not args.no_video:
        make_video(cfg, depth_names)
    extract_and_save(cfg, meshes, ri_to_hw, depth_names, args.smooth)


if __name__ == '__main__':
    main()