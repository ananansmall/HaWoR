"""
可视化验证脚本 - 验证 HaWoR 重建数据与原始视频的对齐关系

生成内容:
  1. 手部网格叠加在原始视频帧上 (验证 2D 对齐)
  2. 深度图可视化 (验证 3D 深度预测)
  3. 相机轨迹可视化 (验证 SLAM 轨迹)
  4. 综合验证视频

用法:
  conda run -n hawor python tools/vis_verify.py --seq example/beizi
  conda run -n hawor python tools/vis_verify.py --seq example/7
  conda run -n hawor python tools/vis_verify.py --seq example/beizi --frames 10 30 50 70
  conda run -n hawor python tools/vis_verify.py --seq example/beizi --skip_video
"""

import argparse
import os
import sys
import numpy as np
import torch
import cv2
import json
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hawor.utils.process import run_mano, run_mano_left, get_mano_faces
from hawor.utils.rotation import rotation_matrix_to_angle_axis
from lib.eval_utils.custom_utils import cam2world_convert, load_slam_cam


MANO_SKELETON = [
    [0,1],[0,5],[0,9],[0,13],[0,17],
    [1,2],[2,3],[3,4],
    [5,6],[6,7],[7,8],
    [9,10],[10,11],[11,12],
    [13,14],[14,15],[15,16],
    [17,18],[18,19],[19,20],
]

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_NAMES = {4:'Thumb', 8:'Index', 12:'Middle', 16:'Ring', 20:'Pinky'}


def find_file_by_pattern(directory, suffix):
    for f in os.listdir(directory):
        if f.endswith(suffix):
            return os.path.join(directory, f)
    return None


def load_seq_data(seq_folder):
    data = {}

    img_dir = os.path.join(seq_folder, 'extracted_images')
    img0_path = None
    for f in sorted(os.listdir(img_dir)):
        if f.endswith(('.jpg', '.png')):
            img0_path = os.path.join(img_dir, f)
            break
    img0 = cv2.imread(img0_path)
    data['img_h'], data['img_w'] = img0.shape[:2]

    with open(os.path.join(seq_folder, 'est_focal.txt')) as f:
        data['focal'] = float(f.read().strip())
    data['cx'] = data['img_w'] / 2.0
    data['cy'] = data['img_h'] / 2.0

    slam_dir = os.path.join(seq_folder, 'SLAM')
    slam_files = [d for d in os.listdir(slam_dir) if d.endswith('.npz')]
    if slam_files:
        slam_path = os.path.join(slam_dir, slam_files[0])
        data['slam'] = dict(np.load(slam_path, allow_pickle=True))
        data['R_w2c_all'], data['t_w2c_all'], data['R_c2w_all'], data['t_c2w_all'] = load_slam_cam(slam_path)
        data['scale'] = float(data['slam']['scale'])
        data['traj'] = data['slam']['traj']

    cam_dir = os.path.join(seq_folder, 'cam_space')
    data['hands'] = {}
    if os.path.exists(cam_dir):
        for hand_id in os.listdir(cam_dir):
            hand_path = os.path.join(cam_dir, hand_id)
            if not os.path.isdir(hand_path):
                continue
            for fname in os.listdir(hand_path):
                if fname.endswith('.json'):
                    parts = fname.replace('.json', '').split('_')
                    start, end = int(parts[0]), int(parts[1])
                    with open(os.path.join(hand_path, fname)) as f:
                        raw = json.load(f)
                    data['hands'][int(hand_id)] = {
                        'start': start, 'end': end, 'data': {k: torch.tensor(v) for k, v in raw.items()}
                    }

    ws_path = os.path.join(seq_folder, 'world_space_res.pth')
    if os.path.exists(ws_path):
        ws = joblib.load(ws_path)
        names = ['pred_trans', 'pred_rot', 'pred_hand_pose', 'pred_betas', 'pred_valid']
        data['world_res'] = {n: v for n, v in zip(names, ws)}

    tracks_dir = None
    for d in os.listdir(seq_folder):
        if d.startswith('tracks_') and os.path.isdir(os.path.join(seq_folder, d)):
            tracks_dir = os.path.join(seq_folder, d)
            break
    if tracks_dir:
        tracks_path = os.path.join(tracks_dir, 'model_tracks.npy')
        if os.path.exists(tracks_path):
            data['tracks'] = np.load(tracks_path, allow_pickle=True).item()
        masks_path = os.path.join(tracks_dir, 'model_masks.npy')
        if os.path.exists(masks_path):
            data['masks'] = np.load(masks_path, allow_pickle=True)

    return data


def project_points(pts3d, focal, cx, cy):
    if pts3d.ndim == 1:
        pts3d = pts3d.reshape(1, 3)
    Z = pts3d[:, 2]
    valid = Z > 0.01
    u = np.where(valid, focal * pts3d[:, 0] / np.maximum(Z, 1e-6) + cx, -1)
    v = np.where(valid, focal * pts3d[:, 1] / np.maximum(Z, 1e-6) + cy, -1)
    return np.stack([u, v], axis=-1), valid


def draw_hand_overlay(img, verts, faces, joints, focal, cx, cy, color_mesh=(0,200,0), color_skel=(0,255,255)):
    overlay = img.copy()
    h, w = img.shape[:2]

    pts2d, valid = project_points(verts, focal, cx, cy)
    for face in faces:
        all_ok = True
        pts = []
        for idx in face:
            if not valid[idx] or pts2d[idx, 0] < 0 or pts2d[idx, 0] >= w or pts2d[idx, 1] < 0 or pts2d[idx, 1] >= h:
                all_ok = False
                break
            pts.append([int(pts2d[idx, 0]), int(pts2d[idx, 1])])
        if all_ok and len(pts) >= 3:
            cv2.fillPoly(overlay, [np.array(pts)], color_mesh)

    j2d, jvalid = project_points(joints, focal, cx, cy)
    for a, b in MANO_SKELETON:
        if jvalid[a] and jvalid[b]:
            pa = (int(j2d[a, 0]), int(j2d[a, 1]))
            pb = (int(j2d[b, 0]), int(j2d[b, 1]))
            if 0 <= pa[0] < w and 0 <= pa[1] < h and 0 <= pb[0] < w and 0 <= pb[1] < h:
                cv2.line(overlay, pa, pb, color_skel, 2)
    for i in range(21):
        if jvalid[i] and 0 <= j2d[i, 0] < w and 0 <= j2d[i, 1] < h:
            cv2.circle(overlay, (int(j2d[i, 0]), int(j2d[i, 1])), 3, color_skel, -1)

    return cv2.addWeighted(overlay, 0.5, img, 0.5, 0)


def draw_depth_map(img, verts, focal, cx, cy):
    h, w = img.shape[:2]
    depth_img = np.zeros((h, w), dtype=np.float32)
    count_img = np.zeros((h, w), dtype=np.float32)

    pts2d, valid = project_points(verts, focal, cx, cy)
    Z = verts[:, 2]
    for i in range(len(verts)):
        if not valid[i]:
            continue
        u, v = int(pts2d[i, 0]), int(pts2d[i, 1])
        if 0 <= u < w and 0 <= v < h:
            depth_img[v, u] += Z[i]
            count_img[v, u] += 1

    mask = count_img > 0
    depth_img[mask] /= count_img[mask]

    depth_vis = np.zeros_like(depth_img)
    valid_depth = depth_img[mask]
    if len(valid_depth) > 0:
        dmin, dmax = valid_depth.min(), valid_depth.max()
        if dmax - dmin > 1e-6:
            depth_vis[mask] = (depth_img[mask] - dmin) / (dmax - dmin)
        else:
            depth_vis[mask] = 0.5

    depth_color = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    depth_color[~mask] = img[~mask]
    blend = cv2.addWeighted(depth_color, 0.7, img, 0.3, 0)
    return blend


def draw_depth_info_panel(img, joints_cam, focal, cx, cy, frame_idx, scale):
    h, w = img.shape[:2]
    panel = np.zeros((h, 280, 3), dtype=np.uint8)

    root = joints_cam[0]
    panel_y = 30
    cv2.putText(panel, f"Frame {frame_idx}", (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    panel_y += 30
    cv2.putText(panel, f"Root depth (tz): {root[2]:.4f}m", (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,255), 1)
    panel_y += 22

    for tip_idx in FINGER_TIPS:
        j = joints_cam[tip_idx]
        name = FINGER_NAMES[tip_idx]
        cv2.putText(panel, f"  {name}: z={j[2]:.4f}m", (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
        panel_y += 18

    panel_y += 10
    depths = joints_cam[:, 2]
    cv2.putText(panel, f"Depth range:", (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,255,200), 1)
    panel_y += 22
    cv2.putText(panel, f"  min={depths.min():.4f}m", (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
    panel_y += 18
    cv2.putText(panel, f"  max={depths.max():.4f}m", (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
    panel_y += 18
    cv2.putText(panel, f"  span={depths.max()-depths.min():.4f}m", (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

    panel_y += 30
    cv2.putText(panel, f"Focal: {focal:.0f}px", (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,200), 1)
    panel_y += 22
    cv2.putText(panel, f"Scale: {scale:.4f}", (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,200), 1)

    return panel


def draw_camera_trajectory(traj, R_c2w_all, t_c2w_all, current_frame, total_frames):
    fig_w, fig_h = 600, 400
    fig = np.ones((fig_h, fig_w, 3), dtype=np.uint8) * 30

    positions = t_c2w_all.numpy()[:, :3]

    x_range = [positions[:, 0].min() - 0.1, positions[:, 0].max() + 0.1]
    z_range = [positions[:, 2].min() - 0.1, positions[:, 2].max() + 0.1]

    def to_fig(x, z):
        fx = int((x - x_range[0]) / (x_range[1] - x_range[0]) * (fig_w - 40) + 20)
        fz = int((z - z_range[0]) / (z_range[1] - z_range[0]) * (fig_h - 40) + 20)
        return fx, fz

    for i in range(len(positions) - 1):
        p1 = to_fig(positions[i, 0], positions[i, 2])
        p2 = to_fig(positions[i+1, 0], positions[i+1, 2])
        alpha = 0.3 + 0.7 * i / len(positions)
        color = (int(100*alpha), int(100*alpha), int(200*alpha))
        cv2.line(fig, p1, p2, color, 1)

    cp = to_fig(positions[current_frame, 0], positions[current_frame, 2])
    cv2.circle(fig, cp, 6, (0, 255, 0), -1)
    cv2.circle(fig, cp, 8, (0, 255, 0), 2)

    cv2.putText(fig, "Camera Trajectory (XZ top-view)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(fig, f"Frame {current_frame}/{total_frames}", (10, fig_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

    return fig


def main():
    parser = argparse.ArgumentParser(description='HaWoR 重建数据可视化验证')
    parser.add_argument('--seq', type=str, default='example/beizi', help='示例文件夹路径')
    parser.add_argument('--frames', type=int, nargs='*', default=None, help='指定要可视化的帧号 (如 --frames 10 30 50)')
    parser.add_argument('--skip_video', action='store_true', help='跳过视频生成，只生成图片')
    parser.add_argument('--step', type=int, default=5, help='图片模式下每隔几帧取一帧')
    args = parser.parse_args()

    print("加载数据...")
    data = load_seq_data(args.seq)
    focal = data['focal']
    cx, cy = data['cx'], data['cy']
    img_h, img_w = data['img_h'], data['img_w']
    scale = data.get('scale', 0.0)
    print(f"  图像: {img_w}x{img_h}, 焦距: {focal}, 尺度: {scale:.4f}")

    out_dir = os.path.join(args.seq, 'vis_verify')
    os.makedirs(out_dir, exist_ok=True)

    faces = get_mano_faces()

    R_x = torch.tensor([[1,0,0],[0,-1,0],[0,0,-1]]).float()
    R_c2w_x = torch.einsum('ij,njk->nik', R_x, data['R_c2w_all'])
    t_c2w_x = torch.einsum('ij,nj->ni', R_x, data['t_c2w_all'])
    R_w2c_x = R_c2w_x.transpose(-1,-2)
    t_w2c_x = -torch.einsum('bij,bj->bi', R_w2c_x, t_c2w_x)

    R_w2c_np = R_w2c_x.numpy()
    t_w2c_np = t_w2c_x.numpy()

    cam_dir = os.path.join(args.seq, 'cam_space')
    detected_hands = set()
    if os.path.exists(cam_dir):
        for hid_str in os.listdir(cam_dir):
            if os.path.isdir(os.path.join(cam_dir, hid_str)):
                detected_hands.add(int(hid_str))

    if 'world_res' not in data:
        print("  错误: 没有 world_space_res.pth，无法生成验证图")
        return

    wr = data['world_res']
    pred_trans = wr['pred_trans']
    pred_rot = wr['pred_rot']
    pred_hand_pose = wr['pred_hand_pose']
    pred_betas = wr['pred_betas']
    pred_valid = wr['pred_valid']

    num_total = min(pred_trans.shape[1], R_w2c_np.shape[0])

    hand_id_map = {1: ('right', run_mano), 0: ('left', run_mano_left)}

    for hand_id in detected_hands:
        hand_side, mano_fn = hand_id_map.get(hand_id, ('right', run_mano))
        hand_dir = os.path.join(out_dir, f'hand_{hand_id}')
        os.makedirs(hand_dir, exist_ok=True)

        outputs = mano_fn(pred_trans[hand_id:hand_id+1, :num_total],
                          pred_rot[hand_id:hand_id+1, :num_total],
                          pred_hand_pose[hand_id:hand_id+1, :num_total],
                          betas=pred_betas[hand_id:hand_id+1, :num_total])
        verts_raw = outputs['vertices'][0].cpu().numpy()
        joints_raw = outputs['joints'][0].cpu().numpy()

        verts_world = (R_x.numpy() @ verts_raw.reshape(-1,3).T).T.reshape(verts_raw.shape)
        joints_world = (R_x.numpy() @ joints_raw.reshape(-1,3).T).T.reshape(joints_raw.shape)

        print(f"  手 {hand_id} ({hand_side}): 帧 0-{num_total-1}")

        target_frames = args.frames if args.frames else list(range(0, num_total, args.step))

        for idx in target_frames:
            if idx >= num_total:
                continue
            if not pred_valid[hand_id, idx]:
                continue
            if np.isnan(verts_world[idx]).any():
                continue

            img_path = os.path.join(args.seq, 'extracted_images', f'{idx:04d}.jpg')
            if not os.path.exists(img_path):
                continue

            img_orig = cv2.imread(img_path)

            v_cam = (R_w2c_np[idx] @ verts_world[idx].T).T + t_w2c_np[idx]
            j_cam = (R_w2c_np[idx] @ joints_world[idx].T).T + t_w2c_np[idx]

            img_overlay = draw_hand_overlay(img_orig.copy(), v_cam, faces, j_cam, focal, cx, cy)
            img_depth = draw_depth_map(img_orig.copy(), v_cam, focal, cx, cy)
            info_panel = draw_depth_info_panel(img_orig.copy(), j_cam, focal, cx, cy, idx, scale)
            traj_fig = draw_camera_trajectory(data['traj'], R_c2w_x, t_c2w_x, idx, num_total)

            top_row = np.hstack([img_orig, img_overlay, img_depth])
            info_resized = cv2.resize(info_panel, (280, img_h))
            traj_resized = cv2.resize(traj_fig, (600, img_h))
            bottom_row = np.hstack([info_resized, traj_resized])
            composite = np.hstack([top_row, bottom_row])

            cv2.imwrite(os.path.join(hand_dir, f'frame_{idx:04d}.png'), composite)

        print(f"    已保存图片到 {hand_dir}/")

        if not args.skip_video:
            print(f"    生成验证视频...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = os.path.join(hand_dir, 'verify.mp4')
            writer = None

            for idx in range(num_total):
                if not pred_valid[hand_id, idx]:
                    continue
                if np.isnan(verts_world[idx]).any():
                    continue

                img_path = os.path.join(args.seq, 'extracted_images', f'{idx:04d}.jpg')
                if not os.path.exists(img_path):
                    continue

                img_orig = cv2.imread(img_path)

                v_cam = (R_w2c_np[idx] @ verts_world[idx].T).T + t_w2c_np[idx]
                j_cam = (R_w2c_np[idx] @ joints_world[idx].T).T + t_w2c_np[idx]

                img_overlay = draw_hand_overlay(img_orig.copy(), v_cam, faces, j_cam, focal, cx, cy)
                img_depth = draw_depth_map(img_orig.copy(), v_cam, focal, cx, cy)
                info_panel = draw_depth_info_panel(img_orig.copy(), j_cam, focal, cx, cy, idx, scale)
                traj_fig = draw_camera_trajectory(data['traj'], R_c2w_x, t_c2w_x, idx, num_total)

                top_row = np.hstack([img_orig, img_overlay, img_depth])
                info_resized = cv2.resize(info_panel, (280, img_h))
                traj_resized = cv2.resize(traj_fig, (600, img_h))
                bottom_row = np.hstack([info_resized, traj_resized])
                composite = np.hstack([top_row, bottom_row])

                if writer is None:
                    ch, cw = composite.shape[:2]
                    writer = cv2.VideoWriter(video_path, fourcc, 30, (cw, ch))
                writer.write(composite)

            if writer is not None:
                writer.release()
            print(f"    已保存视频到 {video_path}")

    print(f"\n验证结果保存在: {out_dir}/")
    print("  每帧图片包含:")
    print("    原始图像 | 手部叠加 | 深度图 | 深度信息面板 | 相机轨迹")


if __name__ == '__main__':
    main()
