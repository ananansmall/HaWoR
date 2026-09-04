"""
MANO 前向 -> 778 顶点 -> 分支 C 投影 -> 查 RAS depth 反投影 -> RAS depth cam 3D.

与 demov2 的 hawor_results_*.npz 互补:
  hawor_results:  MANO 参数 (HaWoR cam 空间, 纯算法产出, 不含 RAS depth)
  本脚本:         MANO 前向 3D (RAS depth 相机坐标系, 含 RAS depth 信息)

核心思路: 用 778 顶点密度查 depth (而非 21 关节), 避免 121 视频在 688x384
分辨率下关节像素碰撞导致的 depth 退化; 再用 MANO 关节回归矩阵得到 21 关节
的 RAS depth cam 3D.

用法:
  cd HaWoR
  python3 mano_ras_reconstruction.py --video-name 121_C5_CellPhone_161deg
  python3 mano_ras_reconstruction.py --video-name 7

输出:
  HaWoR/output/<video-name>/reconstruction/mano_ras_reconstruction_<start>_<end>.npz
"""
import argparse
import os
import sys

import numpy as np
import cv2
import torch
import smplx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

W0, H0 = 1920, 1080
W, H = 688, 384
sx, sy = W / W0, H / H0
F_HAWOR = 600.0
DEPTH_SCALE = 0.001

# 与 HaWoR lib/models/mano_wrapper.py 完全一致的 21 关节构造:
# 16 个 J_regressor 回归关节 + 5 个指尖(固定顶点索引), 再按 OpenPose 顺序重排
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
TIP_VERTEX_IDX = [744, 320, 443, 554, 671]  # smplx vertex_ids['mano']: thumb/index/middle/ring/pinky

MANO_DIR = os.path.join(HERE, '_DATA', 'data', 'mano')
MANO_FILES = {
    'left':  os.path.join(MANO_DIR, 'MANO_LEFT.pkl'),
    'right': os.path.join(MANO_DIR, 'MANO_RIGHT.pkl'),
}


def build_config(args):
    if args.video_name is None:
        raise SystemExit('[cfg] --video-name is required')
    video_name = args.video_name
    hawor_out = args.hawor_out or os.path.join(HERE, 'output', video_name)
    ras_dir = args.ras_dir or os.path.join(HERE, '..', 'ReplicateAnyScene',
        'output_v2', f'{video_name}{args.ras_suffix}')

    fx, fy, cx, cy = load_intrinsic(ras_dir)

    import glob
    r_files = sorted(glob.glob(os.path.join(hawor_out, 'reconstruction',
        'hawor_results_*.npz')))
    if not r_files:
        raise SystemExit(f'[cfg] no hawor_results under {hawor_out}/reconstruction')
    r = np.load(r_files[-1])
    pred_rot   = r['pred_rot']
    pred_hand  = r['pred_hand_pose']
    pred_betas = r['pred_betas']
    pred_valid = r['pred_valid']
    pred_trans = r['pred_trans']
    T = pred_rot.shape[1]
    start_idx = int(r['start_idx'])
    end_idx   = int(r['end_idx'])

    hands = sorted([h for h in (0, 1) if pred_valid[h].any()])
    if not hands:
        raise SystemExit('[cfg] no valid hand')

    depth_names = sorted([f for f in os.listdir(os.path.join(ras_dir, 'depth'))
                          if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
    max_ri = int(depth_names[-1].split('.')[0]) if depth_names else 0

    fb_ratio = float(args.frame_ratio) if args.frame_ratio is not None \
               else (T - 1) / max(1, max_ri)

    print(f'[cfg] video={video_name}  hands={hands}  T={T}  frame-ratio={fb_ratio:.4f}')
    print(f'[cfg] intrin=({fx:.2f},{fy:.2f},{cx:.1f},{cy:.1f})')
    print(f'[cfg] ras_depth frames=0..{max_ri}  hawor_results={os.path.basename(r_files[-1])}')

    return dict(video_name=video_name, ras_dir=ras_dir, hawor_out=hawor_out,
        pred_rot=pred_rot, pred_hand=pred_hand, pred_betas=pred_betas,
        pred_valid=pred_valid, pred_trans=pred_trans,
        fx=fx, fy=fy, cx=cx, cy=cy, T=T, start_idx=start_idx, end_idx=end_idx,
        hands=hands, fb_ratio=fb_ratio, depth_names=depth_names, max_ri=max_ri)


def load_intrinsic(ras_dir):
    intr = os.path.join(ras_dir, 'intrinsic.txt')
    if not os.path.isfile(intr):
        print('[cfg] WARNING: no intrinsic.txt, using 121 default')
        return 391.4355, 390.7242, 344.0, 192.0
    raw = np.loadtxt(intr)
    return float(raw[0, 0]), float(raw[1, 1]), float(raw[0, 2]), float(raw[1, 2])


def aa_to_rotmat_np(a):
    """(N, 3) -> (N, 3, 3)  or  (B, M, 3) -> (B*M, 3, 3)"""
    a = np.asarray(a, dtype=np.float64)
    N = a.size // 3
    a = a.reshape(N, 3)
    theta = np.linalg.norm(a, axis=1, keepdims=True).reshape(-1, 1, 1)
    safe = np.where(theta < 1e-6, 1.0, theta)
    v = a / safe[:, 0]
    c = np.cos(theta); s = np.sin(theta)
    K = np.zeros((N, 3, 3), dtype=np.float64)
    K[:, 0, 1] = -v[:, 2]; K[:, 0, 2] =  v[:, 1]
    K[:, 1, 0] =  v[:, 2]; K[:, 1, 2] = -v[:, 0]
    K[:, 2, 0] = -v[:, 1]; K[:, 2, 1] =  v[:, 0]
    return c * np.eye(3) + s * K + (1 - c) * np.einsum('ni,nj->nij', v, v)


def build_pred_rotmat(pred_rot, pred_hand):
    """pred_rot (2,T,3), pred_hand (2,T,45) -> (2,T,16,3,3)"""
    hands, T, _ = pred_rot.shape
    ro = aa_to_rotmat_np(pred_rot).reshape(hands, T, 3, 3)
    hp_flat = aa_to_rotmat_np(pred_hand).reshape(hands, T, 15, 3, 3)
    return np.concatenate([ro[:, :, None], hp_flat], axis=2)


def rotmat_to_aa_np(R):
    """(16, 3, 3) -> (16*3,) axis-angle (for older smplx which expects flat pose vector)"""
    R = R.reshape(-1, 3, 3)
    N = R.shape[0]
    cos_t = (np.trace(R, axis1=1, axis2=2) - 1) / 2
    cos_t = np.clip(cos_t, -1.0, 1.0)
    sin_t = np.sqrt(1 - cos_t ** 2)
    theta = np.arctan2(sin_t, cos_t)
    # axis from skew-symmetric part
    x = (R[:, 2, 1] - R[:, 1, 2]) / (2 * np.maximum(sin_t, 1e-8))
    y = (R[:, 0, 2] - R[:, 2, 0]) / (2 * np.maximum(sin_t, 1e-8))
    z = (R[:, 1, 0] - R[:, 0, 1]) / (2 * np.maximum(sin_t, 1e-8))
    aa = theta[:, None] * np.stack([x, y, z], -1)
    return aa.reshape(-1)


def mano_forward(mano, pred_rotmat_h, betas):
    """单帧 MANO 前向: (16,3,3) + (10,) -> (778,3), (21,3)
    老版 smplx 需要 flat axis-angle pose vector (16*3=48).
    """
    pose_aa = rotmat_to_aa_np(pred_rotmat_h)  # (48,)
    ro = torch.tensor(pose_aa[None, :3], dtype=torch.float32)
    hp = torch.tensor(pose_aa[None, 3:], dtype=torch.float32)
    b  = torch.tensor(betas[None], dtype=torch.float32)
    out = mano(global_orient=ro, hand_pose=hp, betas=b, pose2rot=True)
    return out.vertices[0].cpu().detach().numpy(), out.joints[0].cpu().detach().numpy()


def branch_c_project(v_cam):
    x, y, z = v_cam[:, 0], v_cam[:, 1], v_cam[:, 2]
    zz = np.maximum(z, 1e-6)
    u = (F_HAWOR * x / zz + W0 / 2.0) * sx
    v = (F_HAWOR * y / zz + H0 / 2.0) * sy
    u = u.astype(int); v = v.astype(int)
    valid = (z > 0.01) & (0 <= u) & (u < W) & (0 <= v) & (v < H)
    return u, v, valid


def depth_reproject_batch(d_img, u, v, fx, fy, cx, cy):
    H, W = d_img.shape
    u_clipped = np.clip(u, 0, W - 1)
    v_clipped = np.clip(v, 0, H - 1)
    z = d_img[v_clipped, u_clipped].astype(np.float64) * DEPTH_SCALE
    X = (u.astype(np.float64) - cx) * z / fx
    Y = (v.astype(np.float64) - cy) * z / fy
    return np.stack([X, Y, z], -1)


def build_ri_to_hw(cfg):
    T = cfg['T']
    img_dir = os.path.join(cfg['hawor_out'], 'extracted_images')
    if not os.path.isdir(img_dir):
        print('[map] no extracted_images'); return None
    hw_small = {}
    for i in range(T):
        im = cv2.imread(os.path.join(img_dir, f'{i:04d}.jpg'))
        if im is not None:
            hw_small[i] = cv2.resize(im, (64, 36))
    if not hw_small:
        print('[map] no HaWoR images'); return None
    rc = os.path.join(cfg['ras_dir'], 'color')
    ri_to_hw = {}
    if not os.path.isdir(rc):
        print('[map] no RAS color'); return None
    for f in cfg['depth_names']:
        try: ri = int(f.split('.')[0])
        except ValueError: continue
        im = None
        for ext in ('.jpg', '.jpeg'):
            p = os.path.join(rc, f.split('.')[0] + ext)
            if os.path.isfile(p):
                im = cv2.imread(p); break
        if im is None: continue
        s = cv2.resize(im, (64, 36))
        ri_to_hw[ri] = min(hw_small, key=lambda k: float(
            np.mean((hw_small[k].astype(float) - s.astype(float)) ** 2)))
    print(f'[map] content-mapped {len(ri_to_hw)} RAS frames')
    return ri_to_hw


def reconstruct_one_hand(cfg, hand_id, mano_state):
    T = cfg['T']
    pred_rotmat = cfg['pred_rotmat'][hand_id]
    betas = cfg['pred_betas'][hand_id]
    trans = cfg['pred_trans'][hand_id]
    valid = cfg['pred_valid'][hand_id]

    verts_cam_list = []
    for i in range(T):
        if not valid[i]:
            verts_cam_list.append(np.full((778, 3), np.nan))
            continue
        v, _ = mano_forward(mano_state, pred_rotmat[i], betas[i])
        verts_cam_list.append(v + trans[i])
    return np.array(verts_cam_list)


def apply_depth_to_verts(cfg, verts_cam, hand_id, hw_to_ri):
    """
    778 顶点 → 分支 C 投影 → 查 RAS depth → 反投影.
    输出两个坐标系 (逐帧独立, 无平滑/插值):
      verts_ras:   RAS depth 相机系
      verts_world: RAS 世界系, 用 extrinsics/<ri>.txt (w2c 4x4,
                   约定同 ReplicateAnyScene/mainv2.py: p_cam = R @ p_world + t)
    depth 没查到 (NaN/0) 的顶点保留 NaN.
    帧映射: 优先内容反查 hw_to_ri, fallback 线性比例.
    """
    T = cfg['T']
    verts_ras = np.full((T, 778, 3), np.nan)
    verts_world = np.full((T, 778, 3), np.nan)
    valid_frames = np.zeros(T, dtype=bool)

    for i in range(T):
        if np.isnan(verts_cam[i]).all():
            continue
        ri_near = hw_to_ri.get(i)
        if ri_near is None:
            ri_near = int(np.clip(round(i * cfg['max_ri'] / max(1, T - 1)), 0, cfg['max_ri']))
        dd = cv2.imread(os.path.join(cfg['ras_dir'], 'depth', f'{ri_near}.png'),
                        cv2.IMREAD_UNCHANGED)
        if dd is None:
            continue

        u, v, ok = branch_c_project(verts_cam[i])
        if not ok.any():
            continue

        ras_3d = depth_reproject_batch(dd, u[ok], v[ok],
                                       cfg['fx'], cfg['fy'], cfg['cx'], cfg['cy'])
        idx = np.where(ok)[0]
        # 丢弃 depth=0 的点
        z_ok = ras_3d[:, 2] > 0.01
        if not z_ok.any():
            continue
        idx = idx[z_ok]
        pts_cam = ras_3d[z_ok]
        verts_ras[i, idx] = pts_cam

        # cam -> world: p_world = R_w2c^T @ (p_cam - t_w2c)
        ext_path = os.path.join(cfg['ras_dir'], 'extrinsics', f'{ri_near}.txt')
        if os.path.isfile(ext_path):
            E = np.loadtxt(ext_path).reshape(4, 4)
            R_w2c = E[:3, :3]
            t_w2c = E[:3, 3]
            verts_world[i, idx] = (pts_cam - t_w2c) @ R_w2c
        valid_frames[i] = True

    return verts_ras, verts_world, valid_frames


def verts_to_joints(verts_ras, J_regressor):
    """verts (T,V,3) -> 21 关节 (T,21,3), OpenPose 顺序 (与 HaWoR run_mano 一致).
    16 个 J_regressor 回归关节 + 5 个指尖顶点, 再按 MANO_TO_OPENPOSE 重排.
    NaN-safe: 该帧顶点全 NaN 则整帧关节为 NaN.
    """
    jr = J_regressor if J_regressor.shape[1] == verts_ras.shape[1] else J_regressor.T
    valid = ~np.isnan(verts_ras[:, :, 0])
    frame_valid = valid.any(axis=1)  # (T,)
    verts_safe = np.where(valid[..., None], verts_ras, 0.0)
    j16 = np.einsum('tvd,jv->tjd', verts_safe, jr)  # (T,16,3)
    tips = verts_ras[:, TIP_VERTEX_IDX, :]          # (T,5,3) NaN 自然传播
    j21 = np.concatenate([j16, tips], axis=1)       # raw 顺序: 0-15 关节, 16-20 指尖
    j21[~frame_valid] = np.nan
    return j21[:, MANO_TO_OPENPOSE]


def main():
    parser = argparse.ArgumentParser(
        description='MANO forward + RAS depth reprojection -> RAS depth cam 3D')
    parser.add_argument('--video-name', type=str, required=True)
    parser.add_argument('--ras-dir', type=str, default=None)
    parser.add_argument('--ras-suffix', type=str, default='_vggt_omega')
    parser.add_argument('--hawor-out', type=str, default=None)
    parser.add_argument('--frame-ratio', type=float, default=None)
    args = parser.parse_args()

    cfg = build_config(args)
    cfg['pred_rotmat'] = build_pred_rotmat(cfg['pred_rot'], cfg['pred_hand'])

    # 用 smplx 加载 MANO (会自动找 _DATA/data/mano/*.pkl)
    mano_dict = {}
    for hid in cfg['hands']:
        side = 'right' if hid == 1 else 'left'
        mano = smplx.create(
            os.path.join(MANO_DIR, 'MANO_RIGHT.pkl') if hid == 1
            else os.path.join(MANO_DIR, 'MANO_LEFT.pkl'),
            model_type='mano',
            hand_side=side,
            use_pca=False,
            ext='pkl',
        )
        j_shape = mano.J_regressor.numpy().shape
        print(f'[mano] hand={hid} side={side}  J_regressor={j_shape}')
        mano_dict[hid] = mano

    ri_to_hw = build_ri_to_hw(cfg)
    # 内容反查: 视频帧 i -> RAS depth 帧 (反向映射, 同一视频帧取最小 ri)
    hw_to_ri = {}
    if ri_to_hw:
        for ri, hi in sorted(ri_to_hw.items()):
            hw_to_ri.setdefault(int(hi), int(ri))

    out_path = os.path.join(cfg['hawor_out'], 'reconstruction',
        f'mano_ras_reconstruction_{cfg["start_idx"]}_{cfg["end_idx"]}.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    save_dict = {
        'fx': cfg['fx'], 'fy': cfg['fy'], 'cx': cfg['cx'], 'cy': cfg['cy'],
        'depth_scale': DEPTH_SCALE,
        'start_idx': cfg['start_idx'], 'end_idx': cfg['end_idx'],
        'hawor_frames': cfg['T'], 'ras_frames': cfg['max_ri'] + 1,
        'frame_ratio': cfg['fb_ratio'],
        'ri_to_hw': np.array([(k, v) for k, v in ri_to_hw.items()]) if ri_to_hw else None,
    }

    for hid in cfg['hands']:
        side = 'right' if hid == 1 else 'left'
        print(f'\n--- hand {hid} ({side}) ---')
        verts_cam = reconstruct_one_hand(cfg, hid, mano_dict[hid])
        verts_ras, verts_world, valid_frames = apply_depth_to_verts(
            cfg, verts_cam, hid, hw_to_ri)
        joints_ras = verts_to_joints(verts_ras, mano_dict[hid].J_regressor.numpy())
        joints_world = verts_to_joints(verts_world, mano_dict[hid].J_regressor.numpy())
        ok_v = valid_frames
        ok_j = ~np.isnan(joints_ras[:, 0, 0])
        print(f'  verts_ras valid: {ok_v.sum()}/{cfg["T"]}  joints_ras valid: {ok_j.sum()}/{cfg["T"]}')
        print(f'  joints shape: {joints_ras.shape}')
        for jj, jname in [(0, 'wrist')]:
            js = joints_ras[:, jj, :]
            vjs = js[~np.isnan(js[:, 0])]
            if vjs.size:
                print(f'  {jname}({jj}) cam: mean={np.round(np.nanmean(vjs, 0), 3)} '
                      f'range={np.round(np.ptp(vjs, 0), 3)}')
            jw = joints_world[:, jj, :]
            vjw = jw[~np.isnan(jw[:, 0])]
            if vjw.size:
                print(f'  {jname}({jj}) world: mean={np.round(np.nanmean(vjw, 0), 3)} '
                      f'range={np.round(np.ptp(vjw, 0), 3)}')

        save_dict[f'verts_ras_{side}'] = verts_ras
        save_dict[f'verts_ras_world_{side}'] = verts_world
        save_dict[f'joints_ras_{side}'] = joints_ras
        save_dict[f'joints_ras_world_{side}'] = joints_world
        save_dict[f'verts_cam_{side}'] = verts_cam
        save_dict[f'valid_{side}'] = valid_frames

    np.savez(out_path, **save_dict)
    print(f'\n[save] -> {out_path}')


if __name__ == '__main__':
    main()
