"""
Quantitative check: do the camera-space MANO joints (right hand) fall inside
the rendered hand mask for every frame? This validates that combined_render's
MANO overlay is aligned with the mask.

Outputs per-frame and aggregate metrics:
  - joint_in_mask_rate : fraction of the 21 joints landing inside the mask
  - mean_pixel_offset  : mean 2D distance (px) between each joint and the
                         nearest mask pixel (only meaningful when outside)
Only frames that have BOTH a mask and MANO data are evaluated. Missing hands
are skipped automatically.

Usage:
  python3 verify_mano_mask_alignment.py \
      --seq_folder /mnt/.../121_C5_CellPhone_161deg --start_idx 0 --end_idx 600
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def project_points(pts3d, focal, cx, cy):
    if pts3d.ndim == 1:
        pts3d = pts3d.reshape(1, 3)
    Z = pts3d[:, 2]
    valid = Z > 0.01
    u = np.where(valid, focal * pts3d[:, 0] / np.maximum(Z, 1e-6) + cx, -1)
    v = np.where(valid, focal * pts3d[:, 1] / np.maximum(Z, 1e-6) + cy, -1)
    return np.stack([u, v], axis=-1), valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq_folder', required=True)
    ap.add_argument('--start_idx', type=int, default=0)
    ap.add_argument('--end_idx', type=int, default=600)
    ap.add_argument('--focal', type=float, default=None,
                    help='if unset, read est_focal.txt')
    args = ap.parse_args()

    tf = os.path.join(args.seq_folder, f'tracks_{args.start_idx}_{args.end_idx}')
    masks = np.load(os.path.join(tf, 'model_masks.npy'))      # (T,H,W) bool
    joints = np.load(os.path.join(tf, 'model_joints.npy'))    # (2,T,21,3)
    if args.focal is None:
        ef = os.path.join(args.seq_folder, 'est_focal.txt')
        focal = float(open(ef).read().strip())
    else:
        focal = args.focal
    H, W = masks.shape[1], masks.shape[2]
    cx, cy = W / 2.0, H / 2.0

    T = masks.shape[0]
    rate_all, offset_all = [], []
    per_hand = {0: [], 1: []}
    for i in range(T):
        mask = masks[i]
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        for hand_id in (0, 1):
            j = joints[hand_id, i]
            if not j.any() or np.isnan(j).any():
                continue
            pts2d, valid = project_points(j, focal, cx, cy)
            inside = 0
            offsets = []
            for k in range(21):
                if not valid[k]:
                    continue
                u, v = int(round(pts2d[k, 0])), int(round(pts2d[k, 1]))
                if 0 <= u < W and 0 <= v < H and mask[v, u]:
                    inside += 1
                    offsets.append(0.0)
                else:
                    # distance to nearest mask pixel
                    du = xs - u
                    dv = ys - v
                    d = np.sqrt(du * du + dv * dv).min() if len(xs) else float('inf')
                    offsets.append(d)
            if offsets:
                rate = inside / len(offsets)
                rate_all.append(rate)
                per_hand[hand_id].append(rate)
                offset_all.append(np.mean(offsets))

    rate_all = np.array(rate_all)
    offset_all = np.array(offset_all)
    print(f'focal={focal:.1f}  img={W}x{H}')
    print(f'frames evaluated (mask+MANO): {len(rate_all)}')
    print(f'joint-in-mask rate  : mean={rate_all.mean():.4f} '
          f'min={rate_all.min():.4f} median={np.median(rate_all):.4f}')
    print(f'mean pixel offset   : mean={offset_all.mean():.2f}px '
          f'median={np.median(offset_all):.2f}px max={offset_all.max():.2f}px')
    for hid in (0, 1):
        if per_hand[hid]:
            a = np.array(per_hand[hid])
            print(f'  hand {hid}: n={len(a)} rate_mean={a.mean():.4f}')


if __name__ == '__main__':
    main()
