#!/usr/bin/env python3
"""
从已有的 cam_space JSON 重建 model_verts.npy / model_joints.npy（相机空间）。

无需视频 / SLAM / infiller。cam_space/{0,1}/*.json 是 hawor_motion_estimation
在生成 mask 时保存的 *相机空间* MANO 参数（init_trans/root_orient/hand_pose/betas），
其 init_root_orient / init_hand_pose 以 rotmat 形式存储。这里按与 hawor_video.py
完全一致的方式（rotmat -> angle-axis -> run_mano/run_mano_left）前向一遍，
得到与 mask 同源同帧的相机空间顶点(778)与 21 关节，用于 verify_hoi4d_mano_mask.py。

Usage:
    cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR
    python3 gen_hoi4d_verts.py --seq output/hoi4d/tracks_0_600
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hawor.utils.process import run_mano, run_mano_left, get_mano_faces
from hawor.utils.rotation import rotation_matrix_to_angle_axis


def load_chunks(hand_dir):
    """返回 [{start, end, data}] 列表，data 已是 numpy。"""
    chunks = []
    for fn in sorted(os.listdir(hand_dir)):
        if not fn.endswith('.json'):
            continue
        start, end = fn[:-5].split('_')
        start, end = int(start), int(end)
        d = json.load(open(os.path.join(hand_dir, fn)))
        chunks.append({
            'start': start, 'end': end,
            'init_root_orient': np.array(d['init_root_orient'], dtype=np.float32),
            'init_hand_pose': np.array(d['init_hand_pose'], dtype=np.float32),
            'init_trans': np.array(d['init_trans'], dtype=np.float32),
            'init_betas': np.array(d['init_betas'], dtype=np.float32),
        })
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='output/hoi4d/tracks_0_600')
    args = ap.parse_args()
    seq = args.seq

    mask_path = os.path.join(seq, 'model_masks.npy')
    assert os.path.exists(mask_path), f'缺少 {mask_path}'
    T, H, W = np.load(mask_path, mmap_mode='r').shape
    print(f'[shape] masks (T,H,W) = ({T},{H},{W})')

    cam_space = os.path.join(os.path.dirname(seq), 'cam_space')
    assert os.path.exists(cam_space), f'缺少 {cam_space}'

    model_verts = np.zeros((2, T, 778, 3), dtype=np.float32)
    model_joints = np.zeros((2, T, 21, 3), dtype=np.float32)

    for hand in (0, 1):
        hand_dir = os.path.join(cam_space, str(hand))
        if not os.path.isdir(hand_dir):
            continue
        chunks = load_chunks(hand_dir)
        if hand == 0:
            mano_fn, label = run_mano_left, 'left'
        else:
            mano_fn, label = run_mano, 'right'
        for c in chunks:
            start, end = c['start'], c['end']
            # rotmat (1,T,3,3) -> angle-axis (1,T,3)
            root_aa = rotation_matrix_to_angle_axis(
                torch.from_numpy(c['init_root_orient'])).float()
            hand_aa = rotation_matrix_to_angle_axis(
                torch.from_numpy(c['init_hand_pose'])).float()
            trans = torch.from_numpy(c['init_trans']).float()
            betas = torch.from_numpy(c['init_betas']).float()

            out = mano_fn(trans, root_aa, hand_aa, betas=betas)
            verts = out['vertices'][0].cpu().numpy()   # (T,778,3) camera space
            joints = out['joints'][0].cpu().numpy()     # (T,21,3) camera space
            n = verts.shape[0]
            model_verts[hand, start:start + n] = verts
            model_joints[hand, start:start + n] = joints
            print(f'  hand {hand} ({label}) frames [{start},{start+n}) done')

    np.save(os.path.join(seq, 'model_verts.npy'), model_verts)
    np.save(os.path.join(seq, 'model_joints.npy'), model_joints)
    print(f'[saved] model_verts.npy {model_verts.shape}, model_joints.npy {model_joints.shape}')


if __name__ == '__main__':
    main()
