"""
从修复后的 depth_with_mano 图 + RAS depth 图，生成 MANO 在 RAS depth 相机坐标系下的
3D 坐标 npz，并可合成 depth_with_mano.mp4 视频。

支持任意视频：
  - 内参 fx/fy/cx/cy 从 <ras_dir>/intrinsic.txt 自动读取
  - 手部从 HaWoR cam_space 目录自动推断（0=左手/蓝色, 1=右手/绿色）
  - HaWoR 帧数从 tracks_*/model_joints.npy 形状自动推断
  - 帧映射: 内容反查(缩略图MSE)为主; 失败时按 --frame-ratio 或帧数自动推断
  - 插值时间轴归一化到 [0,1], 不依赖 fps 假设
  - mesh 优先用 tracks_*/model_verts.npy 反投影（与 joints 自洽），缺则从图片颜色提取（可用 --mesh-color 覆盖）
  - 默认不做平滑 (与 demo.py 一致); 可选 --smooth (Savitzky-Golay w=15)

用法:
  cd HaWoR
  python3 mano_ras_3d_from_vis.py --video-name 121_C5_CellPhone_161deg
  python3 mano_ras_3d_from_vis.py --video-name 7
  python3 mano_ras_3d_from_vis.py --video-name 121_C5_CellPhone_161deg --smooth   # 可选平滑
  python3 mano_ras_3d_from_vis.py --video-name <name> --hawor-out <path> --ras-dir <path>

输出:
  - <ras_dir>/depth_with_mano/depth_with_mano.mp4   (视频合成)
  - <ras_dir>/mano_ras_3d.npz                       (3D 坐标)
"""
import argparse
import numpy as np
import cv2
import os

from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 通用几何常量 (depth 视图 688x384, 源自 RAS) ----
W, H = 688, 384
DEPTH_SCALE = 0.001     # uint16 -> 米
# ---- HaWoR 相机空间投影 (focal=600, 1920x1080) -> 688x384 ----
W0, H0 = 1920, 1080
cx0, cy0 = W0 / 2.0, H0 / 2.0
sx, sy = W / W0, H / H0

# demov2 color_map: right(1)=(0,200,0) green, left(0)=(200,100,0) blue
COLOR_BY_HAND = {1: (0, 200, 0), 0: (200, 100, 0)}


# --------------------------------------------------------------------------
# 配置与路径构建
# --------------------------------------------------------------------------
def build_config(args):
    video_name = args.video_name
    if video_name is None:
        raise SystemExit('[cfg] --video-name is required')

    ras_dir = args.ras_dir
    if ras_dir is None:
        ras_dir = os.path.join(HERE, '..', 'ReplicateAnyScene', 'output_v2',
                               f'{video_name}{args.ras_suffix}')

    hawor_out = args.hawor_out
    if hawor_out is None:
        hawor_out = os.path.join(HERE, 'output', video_name)

    fx, fy, cx, cy = load_intrinsic(ras_dir)

    seq = hawor_out
    tracks_dir = find_tracks_dir(seq)
    joints = np.load(os.path.join(tracks_dir, 'model_joints.npy'))   # (2, T, 21, 3)
    hawor_frames = joints.shape[1]

    verts_path = os.path.join(tracks_dir, 'model_verts.npy')
    verts = np.load(verts_path) if os.path.isfile(verts_path) else None   # (2, T, V, 3)
    if verts is not None:
        print(f'[cfg] loaded model_verts.npy shape={verts.shape}')
    else:
        print('[cfg] WARNING: no model_verts.npy; mesh 3D will fall back to color extraction')

    detected_hands = infer_hands(seq, args.hand)
    hand = detected_hands[0]
    if hand not in (0, 1):
        raise SystemExit(f'[cfg] unsupported hand id {hand}')

    # fallback 帧比例 (HaWoR帧/RAS帧): 显式 --frame-ratio > 按总帧数自动推断
    depth_names = list_depth_frames(ras_dir)
    max_ri = int(depth_names[-1].split('.')[0]) if depth_names else 1
    if args.frame_ratio is not None:
        fb_ratio = float(args.frame_ratio)
        print(f'[cfg] fallback frame-ratio = {fb_ratio:.4f} (--frame-ratio)')
    else:
        fb_ratio = (hawor_frames - 1) / max(1, max_ri)
        print(f'[cfg] fallback frame-ratio = {fb_ratio:.4f} (auto: ({hawor_frames}-1)/{max_ri})')

    mesh_color = args.mesh_color or COLOR_BY_HAND[hand]

    npz_out_dir = args.npz_out_dir
    if npz_out_dir is None:
        npz_out_dir = hawor_out   # 默认放 HaWoR/output/<video-name>/

    config = dict(
        video_name=video_name,
        ras_dir=ras_dir,
        hawor_out=hawor_out,
        npz_out_dir=npz_out_dir,
        tracks_dir=tracks_dir,
        joints=joints,
        verts=verts,
        fx=fx, fy=fy, cx=cx, cy=cy,
        hawor_frames=hawor_frames,
        hand=hand,
        detected_hands=detected_hands,
        mesh_color=mesh_color,
        fb_ratio=fb_ratio,
        do_smooth=bool(args.smooth),
    )
    print('[cfg] video=%s  hand=%s(%s)  mesh_color=%s' % (
        video_name, hand, 'left' if hand == 0 else 'right', mesh_color))
    print('[cfg] ras_dir=%s  hawor_out=%s  tracks=%s  hawor_frames=%d  intrin=(%.2f,%.2f,%.1f,%.1f)' % (
        ras_dir, hawor_out, tracks_dir, hawor_frames, fx, fy, cx, cy))
    return config


def load_intrinsic(ras_dir):
    intr = os.path.join(ras_dir, 'intrinsic.txt')
    if not os.path.isfile(intr):
        print('[cfg] WARNING: %s not found, using 121 default intrinsics' % intr)
        return 391.4355, 390.7242, 344.0, 192.0
    raw = np.loadtxt(intr)
    return float(raw[0, 0]), float(raw[1, 1]), float(raw[0, 2]), float(raw[1, 2])


def find_tracks_dir(seq):
    for name in sorted(os.listdir(seq)):
        if name.startswith('tracks_'):
            return os.path.join(seq, name)
    raise SystemExit('[cfg] no tracks_* dir found under %s' % seq)


def infer_hands(seq, hand_arg):
    if hand_arg is not None:
        return (hand_arg,)
    cam = os.path.join(seq, 'cam_space')
    if os.path.isdir(cam):
        hands = []
        for h in os.listdir(cam):
            if os.path.isdir(os.path.join(cam, h)):
                try:
                    hands.append(int(h))
                except ValueError:
                    pass
        if hands:
            return tuple(hands)
    return (1,)   # default right


# --------------------------------------------------------------------------
# 几何工具
# --------------------------------------------------------------------------
def mano_2d(j3, RF=600.0):
    """HaWoR 相机空间 3D -> 688x384 像素坐标"""
    x, y, z = j3[:, 0], j3[:, 1], j3[:, 2]
    zz = np.maximum(z, 1e-6)
    u = RF * x / zz + cx0
    v = RF * y / zz + cy0
    return np.stack([u * sx, v * sy], -1)


def depth_reproject(d, us, vs, fx, fy, cx, cy):
    """像素 + depth(uint16) -> depth 相机 3D (米)"""
    us = us.astype(np.float64); vs = vs.astype(np.float64)
    z = d[vs.astype(int), us.astype(int)].astype(np.float64) * DEPTH_SCALE
    X = (us - cx) * z / fx
    Y = (vs - cy) * z / fy
    return np.stack([X, Y, z], -1)


def smooth_1d(x, window=15, poly=3):
    out = x.copy()
    valid = ~np.isnan(x)
    for c in range(x.shape[-1]):
        col = x[:, c].copy()
        v = valid[:, c]
        if v.sum() < window:
            continue
        idx = np.where(v)[0]
        n = len(idx)
        wl = min(window, n if n % 2 == 1 else n - 1)
        try:
            col[idx] = savgol_filter(col[idx], window_length=wl, polyorder=poly)
        except Exception:
            pass
        out[:, c] = col
    return out


def list_depth_frames(ras_dir):
    return sorted(
        [f for f in os.listdir(os.path.join(ras_dir, 'depth')) if f.endswith('.png')],
        key=lambda x: int(x.split('.')[0]))


def build_ri_to_hw(cfg):
    """内容反查 RAS ri -> HaWoR 帧号 (与 demov2 相同逻辑)"""
    hw_small = {}
    img_dir = os.path.join(cfg['hawor_out'], 'extracted_images')
    T = cfg['hawor_frames']
    if os.path.isdir(img_dir):
        for i in range(T):
            im = cv2.imread(os.path.join(img_dir, f'{i:04d}.jpg'))
            if im is not None:
                hw_small[i] = cv2.resize(im, (64, 36))
    if not hw_small:
        print('[map] WARNING: no HaWoR extracted images; using linear 6*ri fallback')
        return None

    rc = os.path.join(cfg['ras_dir'], 'color')
    ri_to_hw = {}
    if not os.path.isdir(rc):
        print('[map] WARNING: no RAS color dir; using linear 6*ri fallback')
        return None
    for f in list_depth_frames(cfg['ras_dir']):
        try:
            ri = int(f.split('.')[0])
        except ValueError:
            continue
        im = None
        for name in (f.split('.')[0] + '.jpg', f.split('.')[0] + '.jpeg'):
            p = os.path.join(rc, name)
            if os.path.isfile(p):
                im = cv2.imread(p)
                break
        if im is None:
            continue
        s = cv2.resize(im, (64, 36))
        ri_to_hw[ri] = min(hw_small, key=lambda k: float(
            np.mean((hw_small[k].astype(float) - s.astype(float)) ** 2)))
    print(f'[map] content-mapped {len(ri_to_hw)} RAS frames')
    if ri_to_hw:
        k = min(ri_to_hw)
        print(f'[map] sample ri={k} -> hw={ri_to_hw[k]}')
    return ri_to_hw


# --------------------------------------------------------------------------
# 视频合成
# --------------------------------------------------------------------------
def make_video(cfg):
    vis_dir = os.path.join(cfg['ras_dir'], 'depth_with_mano')
    if not os.path.isdir(vis_dir):
        print('[video] SKIP: no depth_with_mano dir at %s' % vis_dir)
        return
    depth_names = list_depth_frames(cfg['ras_dir'])
    out = os.path.join(vis_dir, 'depth_with_mano.mp4')
    fps = 15.0
    writer = None
    n_img = 0
    for f in depth_names:
        img = cv2.imread(os.path.join(vis_dir, f))
        if img is None:
            continue
        n_img += 1
        if writer is None:
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                     (img.shape[1], img.shape[0]))
        for _ in range(3):
            writer.write(img)
    if writer:
        writer.release()
        print(f'[video] -> {out} ({n_img} frames *3 @{fps}fps)')
    else:
        print('[video] SKIP: no valid images')


# --------------------------------------------------------------------------
# 主流程: 生成 mano_ras_3d.npz
# --------------------------------------------------------------------------
def extract_and_save(cfg, ri_to_hw):
    hand = cfg['hand']
    joints = cfg['joints']
    verts = cfg['verts']
    T = cfg['hawor_frames']

    depth_names = list_depth_frames(cfg['ras_dir'])
    vis_dir = os.path.join(cfg['ras_dir'], 'depth_with_mano')

    mesh_3d_list = []
    joints_3d_list = []
    joint_valid = []
    ri_list = []

    for f in depth_names:
        ri = int(f.split('.')[0])
        hwfi = (ri_to_hw.get(ri, ri * cfg['fb_ratio']) if ri_to_hw else ri * cfg['fb_ratio'])
        hwfi = int(np.clip(round(hwfi), 0, T - 1))
        d = cv2.imread(os.path.join(cfg['ras_dir'], 'depth', f), cv2.IMREAD_UNCHANGED)
        vis = cv2.imread(os.path.join(vis_dir, f)) if os.path.isdir(vis_dir) else None
        if d is None or vis is None:
            continue
        ri_list.append(ri)

        # mesh 3D: 优先用 HaWoR model_verts 反投影(与 joints 自洽); 缺则用颜色提取
        v3 = verts[hand, hwfi] if verts is not None else None
        if v3 is not None and v3.any() and not np.isnan(v3).any():
            v2 = mano_2d(v3)
            u = np.clip(np.round(v2[:, 0]).astype(int), 0, W - 1)
            v = np.clip(np.round(v2[:, 1]).astype(int), 0, H - 1)
            mesh_3d_list.append(depth_reproject(d, u, v, cfg['fx'], cfg['fy'], cfg['cx'], cfg['cy']))
        else:
            b = vis[:, :, 0].astype(int)
            g = vis[:, :, 1].astype(int)
            r = vis[:, :, 2].astype(int)
            if hand == 1:   # 右手/绿: G 主导
                mask = (g > np.maximum(b, r) + 30) & (g > 80)
            else:           # 左手/蓝: B 主导
                mask = (b > np.maximum(g, r) + 30) & (b > 80)
            if mask.sum() > 0:
                vs, us = np.where(mask)
                mesh_3d_list.append(depth_reproject(d, us, vs, cfg['fx'], cfg['fy'], cfg['cx'], cfg['cy']))
            else:
                mesh_3d_list.append(np.zeros((0, 3)))

        # joints 3D: HaWoR joint 2D 投影 + depth 反投影
        j3 = joints[hand, hwfi]
        if j3.any() and not np.isnan(j3).any():
            j2 = mano_2d(j3)
            u = np.clip(np.round(j2[:, 0]).astype(int), 0, W - 1)
            v = np.clip(np.round(j2[:, 1]).astype(int), 0, H - 1)
            j3d = depth_reproject(d, u, v, cfg['fx'], cfg['fy'], cfg['cx'], cfg['cy'])
            joints_3d_list.append(j3d)
            joint_valid.append(True)
        else:
            joints_3d_list.append(np.full((21, 3), np.nan))
            joint_valid.append(False)

    ri = np.array(ri_list)
    valid = np.array(joint_valid)
    joints_3d = np.array(joints_3d_list)
    print(f'[extract] {len(ri)} frames, joints valid={valid.sum()}, mesh non-empty={sum(m.shape[0] > 0 for m in mesh_3d_list)}')

    if len(ri) == 0:
        print('[interp] SKIP: no frames extracted')
        return
    # 归一化时间轴 [0,1]: 不依赖任何 fps/重复次数假设
    t_ras = ri / max(float(ri.max()), 1.0)
    t_target = np.arange(T) / max(T - 1, 1)
    joints_T = np.full((T, 21, 3), np.nan)
    for k in range(21):
        for c in range(3):
            y = np.where(valid, joints_3d[:, k, c], np.nan)
            joints_T[:, k, c] = np.interp(t_target, t_ras, y, left=np.nan, right=np.nan)
    if cfg['do_smooth']:
        joints_T_s = np.stack([smooth_1d(joints_T[:, k, :], window=15, poly=3) for k in range(21)], axis=1)
        print('[interp] smoothing enabled (--smooth)')
    else:
        joints_T_s = joints_T
    print(f'[interp] {len(ri)} frames -> {T} frames; NaN frames: {np.isnan(joints_T_s[:, 0, 0]).sum()}')

    # 参考: HaWoR 相机空间 model_joints -> RAS depth 相机 3D
    depth_map = {int(f.split('.')[0]): cv2.imread(os.path.join(cfg['ras_dir'], 'depth', f), cv2.IMREAD_UNCHANGED)
                 for f in depth_names}
    joints_ref_T = np.full((T, 21, 3), np.nan)
    max_ri = int(max(depth_map.keys())) if depth_map else 0
    for i in range(T):
        j3 = joints[hand, i]
        if not (j3.any() and not np.isnan(j3).any()):
            continue
        j2 = mano_2d(j3)
        u = np.clip(np.round(j2[:, 0]).astype(int), 0, W - 1)
        v = np.clip(np.round(j2[:, 1]).astype(int), 0, H - 1)
        ri_near = int(np.clip(round(i * max_ri / (T - 1)), 0, max_ri))
        dd = depth_map.get(ri_near)
        if dd is None:
            continue
        joints_ref_T[i] = depth_reproject(dd, u, v, cfg['fx'], cfg['fy'], cfg['cx'], cfg['cy'])

    d_interp = np.linalg.norm(joints_T_s - joints_ref_T, axis=-1)
    ok = ~np.isnan(d_interp)
    print(f'\n[compare] interp vs ref ({T} frames): valid={ok.any(axis=1).sum()}/{T}'
          f' mean={np.nanmean(d_interp):.4f}m median={np.nanmedian(d_interp):.4f}m')

    d_raw = np.full((len(ri), 21), np.nan)
    for idx, ri_ in enumerate(ri):
        hwfi = int(np.clip(round((ri_to_hw.get(ri_, ri_ * cfg['fb_ratio']) if ri_to_hw else ri_ * cfg['fb_ratio'])), 0, T - 1))
        d_raw[idx] = np.linalg.norm(joints_3d[idx] - joints_ref_T[hwfi], axis=-1)
    print(f'[compare] raw({len(ri)} frames) vs ref: mean={np.nanmean(d_raw):.4f}m median={np.nanmedian(d_raw):.4f}m')

    out_npz = os.path.join(cfg['npz_out_dir'], 'mano_ras_3d.npz')
    os.makedirs(cfg['npz_out_dir'], exist_ok=True)
    np.savez(
        out_npz,
        mesh_3d=np.array(mesh_3d_list, dtype=object),
        joints_3d_ras=joints_3d,
        joints_3d_ras_interp=joints_T_s,
        joints_3d_ref=joints_ref_T,
        ri=ri, valid=valid,
        dist_interp=d_interp,
        ri_to_hw=np.array([(k, v) for k, v in ri_to_hw.items()]) if ri_to_hw else None,
        fx=cfg['fx'], fy=cfg['fy'], cx=cfg['cx'], cy=cfg['cy'],
        depth_scale=DEPTH_SCALE, hand=hand, hawor_frames=T,
        note=('joints_3d_ras: from depth_with_mano + RAS depth reprojection, RAS depth cam (m); '
              'ref=HaWoR model_joints converted to RAS depth cam; '
              'mesh_3d: from HaWoR model_verts or color extraction'),
    )
    print(f'[save] -> {out_npz}')


def main():
    parser = argparse.ArgumentParser(description='Generate MANO 3D npz from depth_with_mano + RAS depth (any video)')
    parser.add_argument('--video-name', type=str, required=True,
                        help='video basename, e.g. 121_C5_CellPhone_161deg or 7')
    parser.add_argument('--ras-dir', type=str, default=None,
                        help='RAS output dir (default: ../ReplicateAnyScene/output_v2/<video-name>_vggt_omega)')
    parser.add_argument('--ras-suffix', type=str, default='_vggt_omega',
                        help='suffix for RAS dir name (default: _vggt_omega)')
    parser.add_argument('--hawor-out', type=str, default=None,
                        help='HaWoR output dir (default: ./output/<video-name>)')
    parser.add_argument('--npz-out-dir', type=str, default=None,
                        help='npz 输出目录 (default: --hawor-out, 即 HaWoR/output/<video-name>/)')
    parser.add_argument('--hand', type=int, default=None,
                        help='hand id: 0=left(blue), 1=right(green); default: inferred from cam_space')
    parser.add_argument('--mesh-color', type=int, nargs=3, default=None,
                        help='BGR color to extract MANO mesh (default: auto from hand)')
    parser.add_argument('--frame-ratio', type=float, default=None,
                        help='fallback HaWoR-frame-per-RAS-frame ratio when content matching fails '
                             '(default: auto from frame counts, e.g. 121: 599/99=6.05)')
    parser.add_argument('--smooth', action='store_true',
                        help='apply Savitzky-Golay smoothing (window=15, poly=3) to interpolated joints')
    parser.add_argument('--no-video', action='store_true',
                        help='skip mp4 synthesis')
    args = parser.parse_args()

    cfg = build_config(args)
    if not args.no_video:
        make_video(cfg)
    else:
        print('[video] SKIP (--no-video)')

    ri_to_hw = build_ri_to_hw(cfg)
    extract_and_save(cfg, ri_to_hw)


if __name__ == '__main__':
    main()