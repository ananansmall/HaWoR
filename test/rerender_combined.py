"""
Re-render combined_render (image + MANO mesh + 21 joints + mask) using the
ALREADY-GENERATED data, WITHOUT re-running the heavy pipeline.

Key fix vs. the original demov2 combined_render:
  MANO (camera-space verts/joints) and the mask were BOTH produced inside
  hawor_motion_estimation with the same focal. That focal is recorded in
  tracks_*/mask_focal.txt. We project MANO with that focal so it stays exactly
  aligned with the mask. Using the focal-search result (img_focal) instead
  caused the mesh/joints to drift ~2x away from the mask.

Missing hands (e.g. left hand when only the right hand appears) are skipped.

Usage:
  python3 rerender_combined.py \
      --seq_folder /mnt/.../121_C5_CellPhone_161deg \
      --output_base /mnt/.../HaWoR/output/121_C5_CellPhone_161deg \
      --start_idx 0 --end_idx 600
"""
import os
import sys
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demov2 import project_points, MANO_SKELETON, FINGER_TIPS
from hawor.utils.process import get_mano_faces

_faces = get_mano_faces()
faces_right = np.array(_faces)
faces_left = faces_right[:, [0, 2, 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq_folder', required=True)
    ap.add_argument('--output_base', required=True)
    ap.add_argument('--start_idx', type=int, default=0)
    ap.add_argument('--end_idx', type=int, default=600)
    args = ap.parse_args()

    tk = os.path.join(args.seq_folder, f'tracks_{args.start_idx}_{args.end_idx}')
    img_dir = os.path.join(args.seq_folder, 'extracted_images')
    combined_dir = os.path.join(args.output_base, 'combined_render')
    os.makedirs(combined_dir, exist_ok=True)

    masks = np.load(os.path.join(tk, 'model_masks.npy'))         # (T,H,W)
    verts = np.load(os.path.join(tk, 'model_verts.npy'))         # (2,T,V,3)
    joints = np.load(os.path.join(tk, 'model_joints.npy'))       # (2,T,21,3)
    mf = os.path.join(tk, 'mask_focal.txt')
    focal = float(open(mf).read().strip()) if os.path.exists(mf) else 600.0

    img_names = sorted(
        [os.path.join(img_dir, f) for f in os.listdir(img_dir)
         if f.lower().endswith(('.jpg', '.png'))])
    H, W = masks.shape[1], masks.shape[2]
    cx, cy = W / 2.0, H / 2.0
    T = min(masks.shape[0], len(img_names), verts.shape[1])
    print(f'focal={focal}  img={W}x{H}  frames={T}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid = os.path.join(combined_dir, 'combined_render.mp4')
    writer = cv2.VideoWriter(vid, fourcc, 30, (W, H))

    for i in range(T):
        img = cv2.imread(img_names[i])
        if img is None:
            continue
        overlay = img.copy()
        # mesh
        for hand_idx, faces_draw, color in [
                (1, np.array(faces_right), (0, 200, 0)),
                (0, np.array(faces_left), (200, 100, 0))]:
            if not verts[hand_idx, i].any() or np.isnan(verts[hand_idx, i]).any():
                continue
            p2, valid = project_points(verts[hand_idx, i], focal, cx, cy)
            for face in faces_draw:
                ok, pts = True, []
                for idx in face:
                    if not valid[idx] or not (0 <= p2[idx, 0] < W and 0 <= p2[idx, 1] < H):
                        ok = False
                        break
                    pts.append([int(p2[idx, 0]), int(p2[idx, 1])])
                if ok and len(pts) >= 3:
                    cv2.fillPoly(overlay, [np.array(pts)], color)
        # joints (21)
        for hand_idx, jc in [(1, (0, 255, 0)), (0, (0, 200, 255))]:
            if not joints[hand_idx, i].any() or np.isnan(joints[hand_idx, i]).any():
                continue
            pj, vj = project_points(joints[hand_idx, i], focal, cx, cy)
            pj = np.round(pj).astype(int)
            for a, b in MANO_SKELETON:
                if vj[a] and vj[b]:
                    cv2.line(overlay, (pj[a, 0], pj[a, 1]), (pj[b, 0], pj[b, 1]), jc, 2)
            for k in range(pj.shape[0]):
                if vj[k]:
                    r = 4 if k in FINGER_TIPS else 3
                    cv2.circle(overlay, (pj[k, 0], pj[k, 1]), r, (0, 0, 255), -1)
                    cv2.circle(overlay, (pj[k, 0], pj[k, 1]), r + 1, (255, 255, 255), 1)
        # mask (cyan)
        mask = masks[i]
        if mask.any():
            mimg = np.zeros_like(img)
            mimg[mask] = (0, 255, 255)
            overlay = cv2.addWeighted(mimg, 0.2, overlay, 0.8, 0)
        result = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        cv2.imwrite(os.path.join(combined_dir, f'{i:06d}.jpg'), result)
        writer.write(result)
    writer.release()
    print(f'Saved: {combined_dir}  (jpg + combined_render.mp4)')


if __name__ == '__main__':
    main()
