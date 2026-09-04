#!/usr/bin/env python3
"""
project_mano_to_depth_v2.py

把 demov2 拟合的 MANO 投影到 RAS depth 视图(688x384)，生成 depth_with_mano。
自动推断帧映射比例：ratio = hawor_frames / ras_frames，均匀插帧场景通用。
- 7号: 113/113 = 1.0 (1:1)
- 121号: 600/100 = 6.0 (每隔6帧取1帧)

用法:
    python3 project_mano_to_depth_v2.py \
        --video_path /path/to/7.mp4 \
        --img_focal 960.0
"""

import os, sys, argparse, numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- MANO skeleton & finger tips (same as demov2) ----
MANO_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGER_TIPS = {4, 8, 12, 16, 20}


def project_points(v3, focal, cx, cy):
    """Project (N,3) camera-space points to (N,2) pixel coords."""
    z = v3[:, 2:3]
    z = np.where(z <= 0, 1e-6, z)
    x = v3[:, 0:1] * focal / z + cx
    y = v3[:, 1:2] * focal / z + cy
    return np.hstack([x, y]), z


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def generate_depth_with_mano(video_path, img_focal, cx=960, cy=540):
    from hawor.utils.process import run_mano, run_mano_left, get_mano_faces

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    here = os.path.dirname(os.path.abspath(__file__))

    # ---- Load npz ----
    npz_dir = os.path.join(here, 'output', video_name, 'reconstruction')
    npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
    if not npz_files:
        print(f"[v2] error: no npz in {npz_dir}")
        return
    npz_path = os.path.join(npz_dir, npz_files[0])
    data = np.load(npz_path)
    pred_trans = data['pred_trans']
    pred_rot = data['pred_rot']
    pred_hand_pose = data['pred_hand_pose']
    pred_betas = data['pred_betas']
    pred_valid = to_numpy(data['pred_valid'])

    # ---- Detect hands from cam_space ----
    cam_space = os.path.join(here, 'output', video_name, 'cam_space')
    detected_hands = set()
    if os.path.isdir(cam_space):
        for d in os.listdir(cam_space):
            if d.startswith('hand_'):
                detected_hands.add(int(d.split('_')[1]))
    # Fallback: if no cam_space, default to left hand (0)
    if not detected_hands:
        detected_hands = {0}
        print("[v2] WARNING: no cam_space found, defaulting to left hand (0)")
    print(f"[v2] detected_hands = {detected_hands}")

    # ---- RAS depth dir ----
    ras_dir = os.path.join(here, '..', 'ReplicateAnyScene', 'output_v2', f'{video_name}_vggt_omega')
    ras_depth = os.path.join(ras_dir, 'depth')
    if not os.path.isdir(ras_depth):
        print(f"[v2] error: RAS depth dir not found: {ras_depth}")
        return

    H, W = 384, 688
    sx, sy = W / 1920.0, H / 1080.0
    faces = get_mano_faces()

    # ---- Rebuild MANO vertices/joints (camera space) ----
    hand_verts = {}
    hand_joints = {}
    for hid in detected_hands:
        if hid not in (0, 1):
            continue
        valid_mask = pred_valid[hid] if pred_valid.ndim == 2 else pred_valid[hid:hid+1]
        if not valid_mask.any():
            continue
        if hid == 1:
            out = run_mano(torch.from_numpy(pred_trans[1:2]).float(),
                           torch.from_numpy(pred_rot[1:2]).float(),
                           torch.from_numpy(pred_hand_pose[1:2]).float(),
                           betas=torch.from_numpy(pred_betas[1:2]).float())
        else:
            out = run_mano_left(torch.from_numpy(pred_trans[0:1]).float(),
                                torch.from_numpy(pred_rot[0:1]).float(),
                                torch.from_numpy(pred_hand_pose[0:1]).float(),
                                betas=torch.from_numpy(pred_betas[0:1]).float())
        hand_verts[hid] = out['vertices'][0].cpu().numpy().astype(np.float64)
        hand_joints[hid] = out['joints'][0].cpu().numpy().astype(np.float64)
        print(f"[v2] hand {hid}: verts {hand_verts[hid].shape}, joints {hand_joints[hid].shape}")

    if not hand_verts:
        print("[v2] error: no valid MANO hands")
        return

    color_map = {1: (0, 200, 0), 0: (200, 100, 0)}

    def to_visible(img):
        if img.dtype == np.uint16:
            vmin, vmax = float(img.min()), float(img.max())
            norm = np.clip((img.astype(np.float32) - vmin) / (vmax - vmin + 1e-8), 0, 1)
            gray = (norm * 255).astype(np.uint8)
        else:
            gray = img
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    out_dir = os.path.join(ras_dir, 'depth_with_mano')
    os.makedirs(out_dir, exist_ok=True)

    depth_files = sorted([f for f in os.listdir(ras_depth) if f.endswith('.png')],
                         key=lambda x: int(x.split('.')[0]))
    ras_frames = len(depth_files)

    # Auto-infer frame ratio from frame counts (uniform interpolation)
    hawor_frames = next(iter(hand_verts.values())).shape[0]
    ratio = hawor_frames / ras_frames
    print(f"[v2] HaWoR frames={hawor_frames}, RAS frames={ras_frames}, ratio={ratio:.4f}")
    print(f"[v2] mapping: hawor_frame = int(ras_frame * {ratio:.4f})")
    print(f"[v2] projecting {ras_frames} frames -> {out_dir}/")

    for f in depth_files:
        ri = int(f.split('.')[0])  # RAS frame index
        fi = int(ri * ratio)       # HaWoR frame index
        if fi >= hawor_frames:
            fi = hawor_frames - 1
        d = cv2.imread(os.path.join(ras_depth, f), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        overlay = to_visible(d)
        for hid, color in color_map.items():
            if hid not in hand_verts or fi >= hand_verts[hid].shape[0]:
                continue
            if not pred_valid[hid, fi] or np.isnan(hand_verts[hid][fi]).any():
                continue
            v3 = hand_verts[hid][fi]
            j3 = hand_joints[hid][fi]
            v2, _ = project_points(v3, img_focal, cx, cy)
            j2, _ = project_points(j3, img_focal, cx, cy)
            v2 = np.stack([v2[:, 0] * sx, v2[:, 1] * sy], -1).astype(np.int32)
            j2 = np.stack([j2[:, 0] * sx, j2[:, 1] * sy], -1).astype(np.int32)
            for tri in faces:
                pts = v2[tri]
                if np.all(pts >= 0) and np.all(pts < [W, H]):
                    cv2.fillPoly(overlay, [pts], color)
            for a, b in MANO_SKELETON:
                pa, pb = (int(j2[a, 0]), int(j2[a, 1])), (int(j2[b, 0]), int(j2[b, 1]))
                if 0 <= pa[0] < W and 0 <= pa[1] < H and 0 <= pb[0] < W and 0 <= pb[1] < H:
                    cv2.line(overlay, pa, pb, color, 2)
            for k in range(21):
                pk = (int(j2[k, 0]), int(j2[k, 1]))
                if 0 <= pk[0] < W and 0 <= pk[1] < H:
                    c = (0, 0, 255) if k in FINGER_TIPS else (0, 255, 255)
                    cv2.circle(overlay, pk, 4, c, -1)
        # Keep original depth filename (e.g. 0.png, 100.png, 0042.png)
        out_f = f
        cv2.imwrite(os.path.join(out_dir, out_f), overlay)

    print(f"[v2] done -> {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--img_focal", type=float, required=True)
    parser.add_argument("--cx", type=float, default=960)
    parser.add_argument("--cy", type=float, default=540)
    args = parser.parse_args()
    generate_depth_with_mano(args.video_path, args.img_focal, args.cx, args.cy)
