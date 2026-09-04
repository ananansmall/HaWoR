TEST (branch B): project 121 right-hand MANO onto the ReplicateAnyScene depth/color
frames via a WORLD-SPACE bridge:

  121 camera-space MANO  --SLAM c2w-->  HAWoR world
                          --coord_align R,t-->  VGGT world
                          --depth w2c-->  depth camera space
                          --depth intrinsic-->  image (688x384)

This is the geometrically correct path when the depth camera is NOT the same as
the 121 source camera. We compare against the 121 mask (downscaled to 688x384)
to judge alignment.
"""
import os
import sys
import json
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demov2 import MANO_SKELETON, FINGER_TIPS
from hawor.utils.process import get_mano_faces

faces_right = np.array(get_mano_faces())

SEQ = '/mnt/data_8THDD/lza/dataset/HOI4D_RGB/HOI4D_selected_200/121_C5_CellPhone_161deg'
RS = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/121_C5_CellPhone_161deg_vggt_omega'
TK = os.path.join(SEQ, 'tracks_0_600')


def quat_to_rotmat(q):
    # q = [qx,qy,qz,qw]
    x, y, z, w = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])
    return R


def load_slam_c2w(traj, fi):
    t = traj[fi, :3]
    q = traj[fi, 3:7]
    R_c2w = quat_to_rotmat(q)
    return R_c2w, t


def main():
    joints = np.load(os.path.join(TK, 'model_joints.npy'))
    verts = np.load(os.path.join(TK, 'model_verts.npy'))
    masks = np.load(os.path.join(TK, 'model_masks.npy'))

    K = np.loadtxt(os.path.join(RS, 'intrinsic.txt'))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Hd, Wd = 384, 688

    ca = json.load(open(os.path.join(RS, 'coordinate_alignment.json')))
    R_align = np.array(ca['R'])
    t_align = np.array(ca['t'])
    extr_dir = os.path.join(RS, 'extrinsics')

    slam = np.load(os.path.join(SEQ, 'SLAM', 'hawor_slam_w_scale_0_600.npz'), allow_pickle=True)
    traj = slam['traj']

    out_dir = os.path.join(RS, 'mano_proj_worldbridge')
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

        R_c2w, t_c2w = load_slam_c2w(traj, fi)
        # 121 cam -> HAWoR world
        jw = (R_c2w @ j.T).T + t_c2w
        vw = (R_c2w @ v.T).T + t_c2w
        # HAWoR world -> VGGT world (coordinate alignment)
        jv = (R_align @ jw.T).T + t_align
        vv = (R_align @ vw.T).T + t_align
        # VGGT world -> depth camera (per-frame w2c)
        w2c_depth = np.loadtxt(os.path.join(extr_dir, f'{fi}.txt'))
        R_d2c = w2c_depth[:3, :3]
        t_d2c = w2c_depth[:3, 3]
        jd = (R_d2c @ jv.T).T + t_d2c
        vd = (R_d2c @ vv.T).T + t_d2c

        # project with depth intrinsic
        def proj(P):
            Z = P[:, 2]
            valid = Z > 0.01
            u = np.where(valid, fx * P[:, 0] / np.maximum(Z, 1e-6) + cx, -1)
            vv_ = np.where(valid, fy * P[:, 1] / np.maximum(Z, 1e-6) + cy, -1)
            return np.stack([u, vv_], -1), valid

        pj2, vj = proj(jd)
        pv2, vv_ = proj(vd)

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
                if not vv_[idx] or not (0 <= pv2[idx, 0] < Wd and 0 <= pv2[idx, 1] < Hd):
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

    print(f'branch-B frames projected: {n}')
    if rate:
        print(f'branch-B joint-in-mask rate: mean={np.mean(rate):.4f} median={np.median(rate):.4f}')


if __name__ == '__main__':
    main()
