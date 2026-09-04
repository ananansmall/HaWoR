#!/usr/bin/env python3
"""
验证 HOI4D 序列中 MANO (verts/joints) 与 model_masks 是否对齐。

原理（与 demov2.combined_render 完全一致）：
  mask 由 renderer.create_camera_from_cv(I, 0) 生成，即直接用
      K = [[focal, 0, W/2],
           [0, focal, H/2]]
  把 run_mano(init_trans) 得到的 *相机空间* 顶点投影到图像。
  model_verts.npy / model_joints.npy 由同一 run_mano 调用保存，也是相机空间，
  因此用同一个 project_points(v, focal, W/2, H/2) 投影后，应与 mask 完全重合。

本脚本只做“投影 + 与 mask 比对”，不依赖视频/SLAM/infiller。

Usage:
    cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR
    python3 verify_hoi4d_mano_mask.py \
        --seq output/hoi4d/tracks_0_600 \
        --est_focal output/hoi4d/est_focal.txt

指标：
  - 顶点命中率   : 投影到图像内的 mesh 顶点中，落在 mask 内的比例
  - 关节命中率   : 投影到图像内且可见的 21 关节中，落在 mask 内的比例
  - 关节偏移     : 不在 mask 内的关节，到 mask 最近边界像素的平均距离
"""
import os
import argparse
import numpy as np
import joblib


MANO_SKELETON = [
    [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
    [1, 2], [2, 3], [3, 4],
    [5, 6], [6, 7], [7, 8],
    [9, 10], [10, 11], [11, 12],
    [13, 14], [14, 15], [15, 16],
    [17, 18], [18, 19], [19, 20],
]
FINGER_TIPS = [4, 8, 12, 16, 20]


def project_points(pts3d, focal, cx, cy):
    if pts3d.ndim == 1:
        pts3d = pts3d.reshape(1, 3)
    Z = pts3d[:, 2]
    valid = Z > 0.01
    u = np.where(valid, focal * pts3d[:, 0] / np.maximum(Z, 1e-6) + cx, -1)
    v = np.where(valid, focal * pts3d[:, 1] / np.maximum(Z, 1e-6) + cy, -1)
    return np.stack([u, v], axis=-1), valid


def mask_boundary_dt(mask):
    """到 mask 边界的有符号距离（mask 内为负，外为正），单位像素。"""
    from scipy.ndimage import distance_transform_edt
    inv = (~mask).astype(np.uint8)
    # 边界 = mask 外到最近 mask 像素的距离
    d_out = distance_transform_edt(inv)
    # mask 内到边界的距离（为正）
    d_in = distance_transform_edt(mask.astype(np.uint8)) if mask.any() else np.zeros_like(mask, dtype=float)
    dt = np.where(mask, -d_in, d_out)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='output/hoi4d/tracks_0_600')
    ap.add_argument('--est_focal', default=None, help='est_focal.txt 路径；缺省时取 <seq>/../est_focal.txt')
    args = ap.parse_args()

    seq = args.seq
    if args.est_focal is None:
        est_focal = os.path.join(os.path.dirname(seq), 'est_focal.txt')
    else:
        est_focal = args.est_focal
    focal = float(open(est_focal).read().strip())
    print(f'[focal] {focal}  (from {est_focal})')

    mask_path = os.path.join(seq, 'model_masks.npy')
    verts_path = os.path.join(seq, 'model_verts.npy')
    joints_path = os.path.join(seq, 'model_joints.npy')
    fc_path = os.path.join(seq, 'frame_chunks_all.npy')
    for p in (mask_path, verts_path, joints_path):
        assert os.path.exists(p), f'缺少 {p}，请先运行 gen_hoi4d_verts.py 生成'

    masks = np.load(mask_path, mmap_mode='r')          # (T,H,W) bool
    verts = np.load(verts_path)                          # (2,T,V,3) camera space
    joints = np.load(joints_path)                        # (2,T,21,3) camera space
    frame_chunks = joblib.load(fc_path) if os.path.exists(fc_path) else {}

    T, H, W = masks.shape
    cx, cy = W / 2.0, H / 2.0
    print(f'[shape] masks={masks.shape}, verts={verts.shape}, joints={joints.shape}')

    v_hit_list, j_hit_list, j_off_list = [], [], []
    v_hit_frames, j_hit_frames = [], []

    for i in range(T):
        mask = np.asarray(masks[i])
        if not mask.any():
            continue

        frame_v_hit = []
        for hand in (0, 1):
            v = verts[hand, i]
            if not v.any() or np.isnan(v).any():
                continue
            p2d, valid = project_points(v, focal, cx, cy)
            x = np.round(p2d[:, 0]).astype(int)
            y = np.round(p2d[:, 1]).astype(int)
            inside = (x >= 0) & (x < W) & (y >= 0) & (y < H) & valid
            hit = np.zeros_like(inside)
            if inside.sum() > 0:
                hit[inside] = mask[y[inside], x[inside]]
                frame_v_hit.append(hit.sum() / inside.sum())

            # joints
            j = joints[hand, i]
            if not j.any() or np.isnan(j).any():
                continue
            pj, vj = project_points(j, focal, cx, cy)
            xj = np.round(pj[:, 0]).astype(int)
            yj = np.round(pj[:, 1]).astype(int)
            inj = (xj >= 0) & (xj < W) & (yj >= 0) & (yj < H) & vj
            if inj.sum() > 0:
                hit_j = np.zeros_like(inj)
                hit_j[inj] = mask[yj[inj], xj[inj]]
                j_hit_list.append(hit_j.sum() / inj.sum())
                # 不在 mask 内的关节，到边界距离
                miss = inj & (~hit_j)  # hit_j 已与 inj 同形
                if miss.sum() > 0:
                    dt = mask_boundary_dt(mask)
                    j_off_list.append(float(np.mean(dt[yj[miss], xj[miss]])))
                j_hit_frames.append(i)

        if frame_v_hit:
            v_hit_frames.extend([i] * len(frame_v_hit))
            v_hit_list.extend(frame_v_hit)

    v_hit = np.mean(v_hit_list) if v_hit_list else float('nan')
    j_hit = np.mean(j_hit_list) if j_hit_list else float('nan')
    j_off = np.mean(j_off_list) if j_off_list else 0.0

    print('=' * 60)
    print(f'  评估帧数 (有 mask): verts 帧={len(v_hit_frames)}, joints 帧={len(j_hit_frames)}')
    print(f'  顶点命中率 (投影顶点落在 mask 内比例): {v_hit*100:.2f}%')
    print(f'  关节命中率 (21 关节落在 mask 内比例): {j_hit*100:.2f}%')
    print(f'  未命中关节到 mask 边界平均距离:       {j_off:.2f} px')
    print('=' * 60)

    # 判定
    if v_hit > 0.95 and j_hit > 0.95:
        print('  [PASS] MANO 与 mask 对齐良好 (命中率>95%)')
    elif v_hit > 0.8 and j_hit > 0.8:
        print('  [WARN] 基本对齐，但有偏差 (命中率 80-95%)')
    else:
        print('  [FAIL] 明显不对齐 (命中率<80%)，请检查 focal / 坐标帧')


if __name__ == '__main__':
    main()
