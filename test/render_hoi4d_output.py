#!/usr/bin/env python3
"""
用已有的 HaWoR 中间产物，生成 HOI4D 的叠加图片帧 + combined_render.mp4。

输入（均已在 output/hoi4d/ 中由 demov2 流程产出）：
  - extracted_images/0000.jpg ... 0599.jpg   原图
  - tracks_0_600/model_masks.npy             (T,H,W) mask
  - tracks_0_600/model_verts.npy             (2,T,778,3) 相机空间顶点
  - tracks_0_600/model_joints.npy            (2,T,21,3)  相机空间关节
  - est_focal.txt                            focal

投影方式与 mask 渲染完全一致：K=[[focal,0,W/2],[0,focal,H/2]]，
即 project_points(v, focal, W/2, H/2)。

输出：
  - tracks_0_600/vis_combined/0000.png ...   每帧叠加图（原图+mask+MANO网格+关节）
  - tracks_0_600/combined_render.mp4         视频

Usage:
    cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR
    python3 render_hoi4d_output.py --seq output/hoi4d/tracks_0_600
"""
import os
import sys
import argparse

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hawor.utils.process import get_mano_faces

MANO_SKELETON = [
    [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
    [1, 2], [2, 3], [3, 4],
    [5, 6], [6, 7], [7, 8],
    [9, 10], [10, 11], [11, 12],
    [13, 14], [14, 15], [15, 16],
    [17, 18], [18, 19], [19, 20],
]


def project_points(pts3d, focal, cx, cy):
    Z = pts3d[:, 2]
    valid = Z > 0.01
    u = np.where(valid, focal * pts3d[:, 0] / np.maximum(Z, 1e-6) + cx, -1)
    v = np.where(valid, focal * pts3d[:, 1] / np.maximum(Z, 1e-6) + cy, -1)
    return np.stack([u, v], axis=-1), valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='output/hoi4d/tracks_0_600')
    ap.add_argument('--est_focal', default=None)
    ap.add_argument('--img_dir', default=None, help='原图目录，缺省取 <seq>/../extracted_images')
    args = ap.parse_args()

    seq = args.seq
    root = os.path.dirname(seq)
    est_focal = args.est_focal or os.path.join(root, 'est_focal.txt')
    focal = float(open(est_focal).read().strip())
    img_dir = args.img_dir or os.path.join(root, 'extracted_images')
    print(f'[focal] {focal}  [img_dir] {img_dir}')

    masks = np.load(os.path.join(seq, 'model_masks.npy'), mmap_mode='r')   # (T,H,W)
    verts = np.load(os.path.join(seq, 'model_verts.npy'))                   # (2,T,778,3)
    joints = np.load(os.path.join(seq, 'model_joints.npy'))                 # (2,T,21,3)
    faces = get_mano_faces()                                                # (1538,3)

    T, H, W = masks.shape
    cx, cy = W / 2.0, H / 2.0

    out_dir = os.path.join(seq, 'vis_combined')
    os.makedirs(out_dir, exist_ok=True)
    out_mp4 = os.path.join(seq, 'combined_render.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_mp4, fourcc, 10.0, (W, H))

    colors = [(0, 200, 255), (0, 255, 120)]  # 右手, 左手 (BGR)
    for i in range(T):
        img_path = os.path.join(img_dir, f'{i:04d}.jpg')
        if not os.path.exists(img_path):
            print(f'[warn] 缺原图 {img_path}, 跳过帧 {i}')
            continue
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        if frame.shape[0] != H or frame.shape[1] != W:
            frame = cv2.resize(frame, (W, H))

        mask = np.asarray(masks[i]).astype(bool)
        if mask.any():
            ov = frame.copy()
            ov[mask] = (ov[mask] * 0.5 + np.array([30, 30, 200], dtype=np.uint8) * 0.5).astype(np.uint8)
            frame = cv2.addWeighted(ov, 0.7, frame, 0.3, 0)

        for hand in (0, 1):
            v = verts[hand, i]
            if not v.any() or np.isnan(v).any():
                continue
            col = colors[hand]
            p2d, valid = project_points(v, focal, cx, cy)
            p2d = np.round(p2d).astype(int)
            for f in faces:
                a, b, c = p2d[f[0]], p2d[f[1]], p2d[f[2]]
                inv = valid[f]
                inb = np.all((p2d[f] >= 0) & (p2d[f] < [W, H]), axis=1)
                if inv.all() and inb.all():
                    cv2.fillPoly(frame, [np.array([a, b, c])], col, lineType=cv2.LINE_AA)
            # joints + skeleton
            j = joints[hand, i]
            pj, vj = project_points(j, focal, cx, cy)
            pj = np.round(pj).astype(int)
            for (s, e) in MANO_SKELETON:
                if vj[s] and vj[e] and 0 <= pj[s, 0] < W and 0 <= pj[s, 1] < H \
                        and 0 <= pj[e, 0] < W and 0 <= pj[e, 1] < H:
                    cv2.line(frame, tuple(pj[s]), tuple(pj[e]), col, 2, cv2.LINE_AA)
            for k in range(21):
                if vj[k] and 0 <= pj[k, 0] < W and 0 <= pj[k, 1] < H:
                    cv2.circle(frame, tuple(pj[k]), 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, tuple(pj[k]), 2, col, -1, cv2.LINE_AA)

        cv2.putText(frame, f'frame {i}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f'{i:04d}.png'), frame)
        vw.write(frame)

    vw.release()
    print(f'[done] 图片帧 -> {out_dir}/')
    print(f'[done] 视频   -> {out_mp4}')


if __name__ == '__main__':
    main()
