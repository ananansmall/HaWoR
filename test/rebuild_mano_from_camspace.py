"""
Rebuild model_verts.npy / model_joints.npy (camera space) for combined_render
from the per-chunk cam_space/*.json files produced by hawor_motion_estimation.

Why this exists:
  The committed run of hawor_motion_estimation saved model_masks.npy but NOT
  model_verts.npy / model_joints.npy (history artifact). combined_render needs
  the camera-space MANO mesh/joints to overlay on the mask-aligned frames.
  The cam_space/*.json files contain the very same run_mano(init_trans) inputs
  (rotmat + trans + shape) used to render the mask, so rebuilding from them
  reproduces the exact same MANO geometry the mask came from.

Hands:
  Only the hands present under cam_space/ are rebuilt. Missing hands (e.g. the
  left hand when the video only shows the right hand) are left as zeros and
  naturally skipped by combined_render.

Usage:
  python3 rebuild_mano_from_camspace.py \
      --seq_folder /mnt/.../121_C5_CellPhone_161deg \
      --start_idx 0 --end_idx 600
"""
import os
import sys
import json
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawor.utils.rotation import rotation_matrix_to_angle_axis
from hawor.utils.process import run_mano, run_mano_left


def rebuild(seq_folder, start_idx, end_idx):
    tracks_dir = os.path.join(seq_folder, f'tracks_{start_idx}_{end_idx}')
    cam_dir = os.path.join(seq_folder, 'cam_space')
    T = end_idx - start_idx
    model_verts = np.zeros((2, T, 778, 3), dtype=np.float32)   # (hand, T, V, 3)
    model_joints = np.zeros((2, T, 21, 3), dtype=np.float32)   # (hand, T, 21, 3)

    if not os.path.isdir(cam_dir):
        print(f'[rebuild] no cam_space dir at {cam_dir}; nothing to rebuild')
        return model_verts, model_joints

    for hand_id_str in sorted(os.listdir(cam_dir)):
        hand_dir = os.path.join(cam_dir, hand_id_str)
        if not os.path.isdir(hand_dir):
            continue
        hand_id = int(hand_id_str)          # 0=left, 1=right (matches demov2)
        do_flip = (hand_id == 0)
        print(f'[rebuild] hand {hand_id} ({"left" if do_flip else "right"}) ...')

        for json_path in sorted(glob.glob(os.path.join(hand_dir, '*.json'))):
            with open(json_path) as f:
                d = json.load(f)
            # d keys: init_root_orient, init_hand_pose, init_trans, init_betas
            # shapes: (1, T_c, 3, 3) / (1, T_c, 15, 3, 3) / (1, T_c, 3) / (1, T_c, 10)
            root_rot = np.array(d['init_root_orient'], dtype=np.float32)   # (1,Tc,3,3)
            hand_rot = np.array(d['init_hand_pose'], dtype=np.float32)     # (1,Tc,15,3,3)
            trans = np.array(d['init_trans'], dtype=np.float32)           # (1,Tc,3)
            betas = np.array(d['init_betas'], dtype=np.float32)           # (1,Tc,10)

            # frame range encoded in filename: {first}_{last}.json
            fn = os.path.splitext(os.path.basename(json_path))[0]
            a, b = fn.split('_')
            fa, fb = int(a), int(b)
            # clamp into [start_idx, end_idx)
            tc = root_rot.shape[1]
            # The stored chunk covers global frames [fa, fa+tc). Validate:
            assert tc == (fb - fa + 1), f'chunk len mismatch {tc} vs {fb-fa+1} in {fn}'

            # rotmat -> angle-axis for run_mano (rotation utils expect torch tensors)
            import torch
            root_aa = rotation_matrix_to_angle_axis(
                torch.from_numpy(root_rot.reshape(-1, 3, 3))).reshape(1, tc, 3)
            hand_aa = rotation_matrix_to_angle_axis(
                torch.from_numpy(hand_rot.reshape(-1, 3, 3))).reshape(1, tc, 15 * 3)
            trans_t = torch.from_numpy(trans)
            betas_t = torch.from_numpy(betas)

            if do_flip:
                outputs = run_mano_left(trans_t, root_aa, hand_aa, betas=betas_t)
            else:
                outputs = run_mano(trans_t, root_aa, hand_aa, betas=betas_t)

            verts = outputs['vertices'][0].cpu().numpy()   # (Tc,778,3) camera space
            joints = outputs['joints'][0].cpu().numpy()    # (Tc,21,3)  camera space

            # place into global frame slice
            g0 = fa - start_idx
            g1 = g0 + tc
            model_verts[hand_id, g0:g1] = verts
            model_joints[hand_id, g0:g1] = joints
            print(f'    chunk {fn}: frames [{fa},{fb}] -> verts/joints set')

    return model_verts, model_joints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq_folder', required=True)
    ap.add_argument('--start_idx', type=int, default=0)
    ap.add_argument('--end_idx', type=int, default=600)
    ap.add_argument('--overwrite', action='store_true',
                    help='overwrite existing model_verts.npy / model_joints.npy')
    args = ap.parse_args()

    tracks_dir = os.path.join(args.seq_folder, f'tracks_{args.start_idx}_{args.end_idx}')
    os.makedirs(tracks_dir, exist_ok=True)
    vp = os.path.join(tracks_dir, 'model_verts.npy')
    jp = os.path.join(tracks_dir, 'model_joints.npy')

    if (os.path.exists(vp) and os.path.exists(jp)) and not args.overwrite:
        print(f'[rebuild] already exists; use --overwrite to regenerate. Skip.')
        return

    verts, joints = rebuild(args.seq_folder, args.start_idx, args.end_idx)
    np.save(vp, verts)
    np.save(jp, joints)
    n_r = int((joints[1].any(axis=(1, 2))).sum())
    n_l = int((joints[0].any(axis=(1, 2))).sum())
    print(f'[rebuild] saved:\n  {vp}\n  {jp}')
    print(f'[rebuild] right-hand frames with data: {n_r}, left-hand: {n_l}')


if __name__ == '__main__':
    main()
