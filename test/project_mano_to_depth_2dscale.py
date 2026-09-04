"""
TEST (branch C): assume the depth camera is the SAME viewpoint as the 121 source
camera (same video), only at a different resolution (688x384 vs 1920x1080).
Then the correct projection is:
  - project 121 camera-space MANO with the focal it was built with (600),
    at the ORIGINAL 1920x1080 resolution (cx=960, cy=540);
  - then rescale the 2D points to 688x384 by factor sx=688/1920, sy=384/1080.
This keeps the hand at the same relative position. We verify against the
downscaled 121 mask (which is also just a resolution rescale of the original).
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demov2 import MANO_SKELETON, FINGER_TIPS, project_points
from hawor.utils.process import get_mano_faces

faces_right = np.array(get_mano_faces())

SEQ = '/mnt/data_8THDD/lza/dataset/HOI4D_RGB/HOI4D_selected_200/121_C5_CellPhone_161deg'
RS = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/121_C5_CellPhone_161deg_vggt_omega'
TK = os.path.join(SEQ, 'tracks_0_600')

Wo, Ho = 1920, 1080          # 121 original resolution
Wd, Hd = 688, 384            # depth resolution
FOCAL_121 = 600.0
sx, sy = Wd / Wo, Hd / Ho


def main():
    joints = np.load(os.path.join(TK, 'model_joints.npy'))
    verts = np.load(os.path.join(TK, 'model_verts.npy'))
    masks = np.load(os.path.join(TK, 'model_masks.npy'))

    out_dir = os.path.join(RS, 'mano_proj_2dscale')
    os.makedirs(out_dir, exist_ok=True)

    T = masks.shape[0]
    rate = []
    n = 0
    for fi in range(min(100, T)):
        j = joints[1, fi]
        if not j.any() or np.isnan(j).any():
            continue
        v = verts[1, fi]
        m = cv2.resize(masks[fi].astype(np.uint8), (Wd, Hd), interpolation=cv2.INTER_NEAREST).astype(bool)

        # project at original res with 121 focal
        pj, vj = project_points(j, FOCAL_121, Wo / 2, Ho / 2)
        pv, vv = project_points(v, FOCAL_121, Wo / 2, Ho / 2)
        # rescale 2D to depth res
        pj2 = np.stack([pj[:, 0] * sx, pj[:, 1] * sy], -1)
        pv2 = np.stack([pv[:, 0] * sx, pv[:, 1] * sy], -1)

        color_path = os.path.join(RS, 'color', f'{fi}.jpg')
        if os.path.exists(color_path):
            bg = cv2.resize(cv2.imread(color_path), (Wd, Hd))
        else:
            dp = np.array(Image.open(os.path.join(RS, 'depth', f'{fi}.png')).convert('L'))
            bg = cv2.applyColorMap((dp / dp.max() * 255).astype(np.uint8), cv2.COLORMAP_JET)
        ov = bg.copy()
        for face in faces_right:
            ok = True
            pts = []
            for idx in face:
                if not vv[idx] or not (0 <= pv2[idx, 0] < Wd and 0 <= pv2[idx, 1] < Hd):
                    ok = False
                    break
                pts.append([int(pv2[idx, 0]), int(pv2[idx, 1])])
            if ok and len(pts) >= 3:
                cv2.fillPoly(ov, [np.array(pts)], (0, 200, 0))
        pjr = np.round(pj2).astype(int)
        for a, b in MANO_SKELETON:
            if vj[a] and vj[b] and all(0 <= pjr[k, 0] < Wd and 0 <= pjr[k, 1] < Hd for k in (a, b)):
                cv2.line(ov, tuple(pjr[a]), tuple(pjr[b]), (0, 255, 0), 2)
        for k in range(pjr.shape[0]):
            if vj[k] and 0 <= pjr[k, 0] < Wd and 0 <= pjr[k, 1] < Hd:
                r = 4 if k in FINGER_TIPS else 3
                cv2.circle(ov, tuple(pjr[k]), r, (0, 0, 255), -1)
        if m.any():
            mi = np.zeros_like(ov)
            mi[m] = (0, 255, 255)
            ov = cv2.addWeighted(mi, 0.25, ov, 0.75, 0)
            ins = sum(1 for k in range(21) if vj[k] and 0 <= pjr[k, 0] < Wd and 0 <= pjr[k, 1] < Hd and m[pjr[k, 1], pjr[k, 0]])
            rate.append(ins / 21.0)
        cv2.imwrite(os.path.join(out_dir, f'{fi:04d}.png'), ov)
        n += 1

    print(f'branch-C frames projected: {n}')
    if rate:
        print(f'branch-C joint-in-mask rate: mean={np.mean(rate):.4f} median={np.median(rate):.4f}')


if __name__ == '__main__':
    main()
