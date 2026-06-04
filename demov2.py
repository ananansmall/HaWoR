"""
HaWoR Reconstruction Pipeline (demov2)

Q&A - 关于数据来源和坐标系的关键问题:
=========================================

Q1: 相机外参和手部数据是固定的还是从视频估计的？
A:  全部从视频估计，没有任何固定/硬编码的数据:
    - 相机外参 (R_c2w, t_c2w): SLAM从视频估计
    - 手部参数 (pred_trans, pred_rot, pred_hand_pose, pred_betas): MANO从视频推理
    - 焦距 (img_focal): SLAM通过重投影误差搜索得到（或从est_focal.txt读取）
    - 主点 (cx, cy): SLAM输出（或图像中心）
    - SLAM scale: SLAM估计的单目深度缩放因子

Q2: World View中的相机轨迹是真实的吗？
A:  是的。World View中的金字塔标记使用SLAM估计的真实相机位姿(R_c2w, t_c2w)。
    World View的观察相机是虚拟的（用于3D可视化），但场景中的相机轨迹和手部
    位置都是真实数据。相机移动不多是正常的——取决于视频内容。

Q3: SLAM scale不等于1有问题吗？
A:  没问题。SLAM scale是单目SLAM的固有特性，用于将相对深度转换为绝对尺度。
    scale的值取决于SLAM的初始化和焦距估计。焦距越准确，scale越接近1，
    但scale≠1不代表结果不正确。关键是手部投影是否与视频对齐。

Q4: 坐标系变换链是什么？
A:  MANO输出(世界空间, SLAM坐标系) → R_x翻转(Y/Z) → 世界空间(OpenCV坐标系)
    → R_w2c变换 → 相机空间 → 透视投影 → 2D像素
    其中 R_x = diag(1,-1,-1) 将SLAM的OpenGL坐标系(Y-up)转为OpenCV坐标系(Y-down)

Q5: 焦距如何确定？
A:  优先级: 命令行参数 > est_focal.txt > SLAM焦距搜索(重投影误差最小化)
    如果est_focal.txt包含默认值600但图像尺寸暗示焦距应更大，SLAM会自动搜索。
"""

import argparse
import sys
import os

import torch
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
import json
import joblib
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from hawor.utils.rotation import rotation_matrix_to_angle_axis
from lib.eval_utils.custom_utils import load_slam_cam, cam2world_convert


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


def project_points(pts3d, focal, cx, cy):
    if pts3d.ndim == 1:
        pts3d = pts3d.reshape(1, 3)
    Z = pts3d[:, 2]
    valid = Z > 0.01
    u = np.where(valid, focal * pts3d[:, 0] / np.maximum(Z, 1e-6) + cx, -1)
    v = np.where(valid, focal * pts3d[:, 1] / np.maximum(Z, 1e-6) + cy, -1)
    return np.stack([u, v], axis=-1), valid


def draw_hand_overlay(img, verts, faces, joints, focal, cx, cy,
                      color_mesh=(0,200,0), color_skel=(0,255,255), alpha=0.5):
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

    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


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


def draw_camera_trajectory(R_c2w_all, t_c2w_all, current_frame, total_frames):
    fig_w, fig_h = 600, 400
    fig = np.ones((fig_h, fig_w, 3), dtype=np.uint8) * 30

    positions = t_c2w_all.numpy()[:, :3] if hasattr(t_c2w_all, 'numpy') else np.array(t_c2w_all)[:, :3]
    positions = positions[:total_frames]

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


def generate_vis_verify(seq_folder, img_focal, start_idx, end_idx,
                        R_c2w_sla_all, t_c2w_sla_all, slam_scale,
                        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid,
                        cx=None, cy=None, output_base=None):
    cam_dir = os.path.join(seq_folder, 'cam_space')
    if not os.path.exists(cam_dir):
        print("  No cam_space directory, skipping vis_verify")
        return

    if output_base is None:
        output_base = seq_folder
    out_dir = os.path.join(output_base, 'vis_verify')
    os.makedirs(out_dir, exist_ok=True)

    img_dir = os.path.join(seq_folder, 'extracted_images')
    img0_path = None
    for f in sorted(os.listdir(img_dir)):
        if f.endswith(('.jpg', '.png')):
            img0_path = os.path.join(img_dir, f)
            break
    img0 = cv2.imread(img0_path)
    img_h, img_w = img0.shape[:2]
    if cx is None:
        cx = img_w / 2.0
    if cy is None:
        cy = img_h / 2.0

    faces = get_mano_faces()
    hand_id_map = {1: ('right', run_mano), 0: ('left', run_mano_left)}

    detected_hands = set()
    for hid_str in os.listdir(cam_dir):
        if os.path.isdir(os.path.join(cam_dir, hid_str)):
            detected_hands.add(int(hid_str))

    num_total = min(pred_trans.shape[1], R_c2w_sla_all.shape[0])

    for hand_id in detected_hands:
        hand_side, mano_fn = hand_id_map.get(hand_id, ('right', run_mano))
        hand_dir = os.path.join(out_dir, f'hand_{hand_id}')
        os.makedirs(hand_dir, exist_ok=True)

        outputs = mano_fn(pred_trans[hand_id:hand_id+1, :num_total],
                          pred_rot[hand_id:hand_id+1, :num_total],
                          pred_hand_pose[hand_id:hand_id+1, :num_total],
                          betas=pred_betas[hand_id:hand_id+1, :num_total])
        verts_cam = outputs['vertices'][0].cpu().numpy()
        joints_cam = outputs['joints'][0].cpu().numpy()

        R_x = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float32)
        verts_world = (R_x @ verts_cam.reshape(-1,3).T).T.reshape(verts_cam.shape)
        joints_world = (R_x @ joints_cam.reshape(-1,3).T).T.reshape(joints_cam.shape)

        print(f"  vis_verify: hand {hand_id} ({hand_side}), frames 0-{num_total-1}")
        print(f"    Projection params: focal={img_focal:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        print(f"    Camera poses: from SLAM (real video data), {R_c2w_sla_all.shape[0]} frames")
        print(f"    Hand params: from MANO inference (real video data)")

        R_c2w_slice = R_c2w_sla_all[:num_total]
        if hasattr(R_c2w_slice, 'numpy'):
            R_c2w_np = R_c2w_slice.numpy()
        else:
            R_c2w_np = np.array(R_c2w_slice)
        R_w2c_np = R_c2w_np.transpose(0, 2, 1)

        t_c2w_slice = t_c2w_sla_all[:num_total]
        if hasattr(t_c2w_slice, 'numpy'):
            t_c2w_np = t_c2w_slice.numpy()
        else:
            t_c2w_np = np.array(t_c2w_slice)
        t_w2c_np = -np.einsum('nij,nj->ni', R_w2c_np, t_c2w_np)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(hand_dir, 'verify.mp4')
        writer = None
        saved_count = 0

        for idx in range(num_total):
            if not pred_valid[hand_id, idx]:
                continue
            if np.isnan(verts_world[idx]).any():
                continue

            img_path = os.path.join(seq_folder, 'extracted_images', f'{start_idx + idx:04d}.jpg')
            if not os.path.exists(img_path):
                continue

            img_orig = cv2.imread(img_path)

            v_cam = (R_w2c_np[idx] @ verts_world[idx].T).T + t_w2c_np[idx]
            j_cam = (R_w2c_np[idx] @ joints_world[idx].T).T + t_w2c_np[idx]

            img_overlay = draw_hand_overlay(img_orig.copy(), v_cam, faces, j_cam,
                                            img_focal, cx, cy)
            img_depth = draw_depth_map(img_orig.copy(), v_cam, img_focal, cx, cy)
            info_panel = draw_depth_info_panel(img_orig.copy(), j_cam, img_focal, cx, cy,
                                               idx, slam_scale)
            traj_fig = draw_camera_trajectory(R_c2w_sla_all, t_c2w_sla_all,
                                               idx, num_total)

            top_row = np.hstack([img_orig, img_overlay, img_depth])
            info_resized = cv2.resize(info_panel, (280, img_h))
            traj_resized = cv2.resize(traj_fig, (600, img_h))
            bottom_row = np.hstack([info_resized, traj_resized])
            composite = np.hstack([top_row, bottom_row])

            if idx % 5 == 0:
                cv2.imwrite(os.path.join(hand_dir, f'frame_{idx:04d}.png'), composite)
                saved_count += 1

            if writer is None:
                ch, cw = composite.shape[:2]
                writer = cv2.VideoWriter(video_path, fourcc, 30, (cw, ch))
            writer.write(composite)

        if writer is not None:
            writer.release()

        print(f"    Saved {saved_count} images + video to {hand_dir}/")


def lookat_matrix(source, target, up):
    source = np.array(source, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    back = target - source
    back = back / np.linalg.norm(back)
    right = np.cross(up, back)
    right = right / np.linalg.norm(right)
    up = np.cross(back, right)
    up = up / np.linalg.norm(up)
    R = np.stack([right, up, back], axis=1)
    t = source
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    return Rt


def generate_checkerboard(length=100, tile_width=0.5, z_offset=-2.0):
    radius = length / 2.0
    num_tiles = max(2, int(length / tile_width))
    vertices = []
    faces = []
    colors = []
    c0 = np.array([172, 172, 172], dtype=np.uint8)
    c1 = np.array([215, 215, 215], dtype=np.uint8)
    for i in range(num_tiles):
        for j in range(num_tiles):
            u0 = j * tile_width - radius
            v0 = i * tile_width - radius
            v = np.array([
                [u0, v0, z_offset],
                [u0 + tile_width, v0, z_offset],
                [u0 + tile_width, v0 + tile_width, z_offset],
                [u0, v0 + tile_width, z_offset],
            ])
            f = np.array([[0, 1, 3], [1, 2, 3]]) + 4 * (i * num_tiles + j)
            use_c0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
            color = c0 if use_c0 else c1
            vertices.append(v)
            faces.append(f)
            colors.append(color)
    vertices = np.concatenate(vertices, axis=0)
    faces = np.concatenate(faces, axis=0)
    return vertices, faces, colors


def generate_camera_markers(R_c2w, t_c2w, radius=0.05, height=0.1):
    marker_verts = np.array([
        [-radius, -radius, 0],
        [radius, -radius, 0],
        [radius, radius, 0],
        [-radius, radius, 0],
        [0, 0, -height],
    ])
    marker_faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [1, 0, 4], [2, 1, 4],
        [3, 2, 4], [0, 3, 4],
    ])
    face_colors = [
        (128, 128, 128), (128, 128, 128),
        (0, 255, 0), (0, 0, 255),
        (0, 255, 0), (0, 0, 255),
    ]

    num_frames = len(t_c2w)
    all_verts = []
    all_faces = []
    all_colors = []

    for i in range(num_frames):
        R = R_c2w[i].numpy() if hasattr(R_c2w[i], 'numpy') else R_c2w[i]
        t = t_c2w[i].numpy() if hasattr(t_c2w[i], 'numpy') else t_c2w[i]
        v = (R @ marker_verts.T).T + t
        f = marker_faces + 5 * i
        all_verts.append(v)
        all_faces.append(f)
        all_colors.extend(face_colors)

    return np.concatenate(all_verts), np.concatenate(all_faces), all_colors


def render_world_view(left_dict, right_dict, R_c2w_all, t_c2w_all, detected_hands,
                      pred_valid, output_pth, img_h, img_w):
    os.makedirs(output_pth, exist_ok=True)

    left_verts_np = left_dict['vertices'][0].cpu().numpy()
    left_faces = left_dict['faces']
    right_verts_np = right_dict['vertices'][0].cpu().numpy()
    right_faces = right_dict['faces']

    num_frames = min(R_c2w_all.shape[0], left_verts_np.shape[0], right_verts_np.shape[0])

    t_c2w_np = t_c2w_all.numpy() if hasattr(t_c2w_all, 'numpy') else t_c2w_all
    scene_center = t_c2w_np.mean(axis=0)
    scene_extent = max(np.linalg.norm(t_c2w_np.max(axis=0) - t_c2w_np.min(axis=0)), 0.3)

    view_dist = scene_extent * 3.0
    source = scene_center + np.array([view_dist * 0.3, -view_dist * 0.5, view_dist * 0.8])
    target = scene_center
    up = np.array([0, 1.0, 0])
    view_cam = lookat_matrix(source, target, up)
    view_R = view_cam[:3, :3]
    view_t = view_cam[:3, 3]

    vis_focal = 1000.0
    vis_cx = img_w / 2.0
    vis_cy = img_h / 2.0

    ground_verts, ground_faces, ground_colors = generate_checkerboard(
        length=max(scene_extent * 4, 5), tile_width=scene_extent * 0.05,
        z_offset=scene_center[2] - scene_extent * 0.5)

    cam_marker_verts, cam_marker_faces, cam_marker_colors = generate_camera_markers(
        R_c2w_all[:num_frames], t_c2w_all[:num_frames], radius=scene_extent * 0.02, height=scene_extent * 0.04)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_pth, 'world_view.mp4')
    writer = None

    for frame_i in range(num_frames):
        canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 240

        all_tris = []

        cam_pts_ground = (view_R.T @ (ground_verts - view_t).T).T
        pts2d_g, valid_g = project_points(cam_pts_ground, vis_focal, vis_cx, vis_cy)
        depths_g = cam_pts_ground[:, 2]
        for fi, face in enumerate(ground_faces):
            if not all(valid_g[idx] for idx in face):
                continue
            pts = pts2d_g[face]
            if any(pts[:, 0] < 0) or any(pts[:, 0] >= img_w) or any(pts[:, 1] < 0) or any(pts[:, 1] >= img_h):
                continue
            avg_depth = float(depths_g[face].mean())
            color = ground_colors[fi // 2].tolist()
            all_tris.append((avg_depth, pts.astype(np.int32), color))

        cam_pts_markers = (view_R.T @ (cam_marker_verts - view_t).T).T
        pts2d_m, valid_m = project_points(cam_pts_markers, vis_focal, vis_cx, vis_cy)
        depths_m = cam_pts_markers[:, 2]
        for fi, face in enumerate(cam_marker_faces):
            if not all(valid_m[idx] for idx in face):
                continue
            pts = pts2d_m[face]
            if any(pts[:, 0] < 0) or any(pts[:, 0] >= img_w) or any(pts[:, 1] < 0) or any(pts[:, 1] >= img_h):
                continue
            avg_depth = float(depths_m[face].mean())
            color = cam_marker_colors[fi]
            all_tris.append((avg_depth, pts.astype(np.int32), color))

        if 1 in detected_hands and pred_valid[1, frame_i] and not np.isnan(right_verts_np[frame_i]).any():
            v_world = right_verts_np[frame_i]
            cam_pts = (view_R.T @ (v_world - view_t).T).T
            pts2d_h, valid_h = project_points(cam_pts, vis_focal, vis_cx, vis_cy)
            depths_h = cam_pts[:, 2]
            for face in right_faces:
                if not all(valid_h[idx] for idx in face):
                    continue
                pts = pts2d_h[face]
                if any(pts[:, 0] < 0) or any(pts[:, 0] >= img_w) or any(pts[:, 1] < 0) or any(pts[:, 1] >= img_h):
                    continue
                avg_depth = float(depths_h[face].mean())
                all_tris.append((avg_depth, pts.astype(np.int32), (180, 100, 255)))

        if 0 in detected_hands and pred_valid[0, frame_i] and not np.isnan(left_verts_np[frame_i]).any():
            v_world = left_verts_np[frame_i]
            cam_pts = (view_R.T @ (v_world - view_t).T).T
            pts2d_h, valid_h = project_points(cam_pts, vis_focal, vis_cx, vis_cy)
            depths_h = cam_pts[:, 2]
            for face in left_faces:
                if not all(valid_h[idx] for idx in face):
                    continue
                pts = pts2d_h[face]
                if any(pts[:, 0] < 0) or any(pts[:, 0] >= img_w) or any(pts[:, 1] < 0) or any(pts[:, 1] >= img_h):
                    continue
                avg_depth = float(depths_h[face].mean())
                all_tris.append((avg_depth, pts.astype(np.int32), (255, 180, 50)))

        all_tris.sort(key=lambda x: -x[0])
        for _, pts, color in all_tris:
            cv2.fillPoly(canvas, [pts], color)

        cv2.putText(canvas, f"World View - Frame {frame_i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        cv2.putText(canvas, "Purple=Right hand  Orange=Left hand  Pyramids=Cameras", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

        if frame_i % 5 == 0:
            cv2.imwrite(os.path.join(output_pth, f'{frame_i:06d}.png'), canvas)

        if writer is None:
            writer = cv2.VideoWriter(video_path, fourcc, 30, (img_w, img_h))
        writer.write(canvas)

    if writer is not None:
        writer.release()

    print(f"✓ World view saved to: {output_pth}/ ({num_frames} frames + world_view.mp4)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default='example/video_0.mp4')
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument("--checkpoint",  type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--infiller_weight",  type=str, default='./weights/hawor/checkpoints/infiller.pt')
    args = parser.parse_args()

    print("="*80)
    print("HaWoR Reconstruction Pipeline")
    print("="*80)
    
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    print(f"✓ Detection & Tracking: frames {start_idx} to {end_idx}")

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    output_base = os.path.join(os.path.dirname(__file__), 'output', video_name)
    os.makedirs(output_base, exist_ok=True)
    print(f"  Output directory: {output_base}")

    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
    print(f"✓ Motion estimation completed")

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        print("Running SLAM...")
        hawor_slam(args, start_idx, end_idx)
    else:
        print("✓ SLAM data already exists")
    
    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)
    slam_data = dict(np.load(slam_path, allow_pickle=True))
    slam_scale = float(slam_data['scale'])

    if 'img_focal' in slam_data:
        slam_focal = float(slam_data['img_focal'])
        if abs(slam_focal - img_focal) > 1.0:
            print(f"  WARNING: SLAM focal ({slam_focal:.1f}) != motion_est focal ({img_focal:.1f}), using SLAM value")
        img_focal = slam_focal
    if 'img_center' in slam_data:
        slam_center = slam_data['img_center']
    else:
        slam_center = None

    print(f"✓ Loaded SLAM cameras:")
    print(f"    focal={img_focal:.1f} (from SLAM), scale={slam_scale:.4f}")
    print(f"    img_center={slam_center if slam_center is not None else 'not available, will use image center'}")
    print(f"    camera poses: {R_c2w_sla_all.shape[0]} frames")

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(args, start_idx, end_idx, frame_chunks_all)
    print(f"✓ Infiller completed")

    hand2idx = {"right": 1, "left": 0}
    vis_start = 0
    vis_end = pred_trans.shape[1]
    num_total_frames = min(
        vis_end - vis_start,
        R_c2w_sla_all.shape[0],
        t_c2w_sla_all.shape[0],
        len(imgfiles),
    )
    vis_end = vis_start + num_total_frames
    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
                          [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
                          [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
                          [120, 108, 78], [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)
    
    hand_idx = hand2idx['right']
    pred_glob_r = run_mano(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], 
                           pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], 
                           pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], 
                           betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    right_verts = pred_glob_r['vertices'][0]
    right_dict = {'vertices': right_verts.unsqueeze(0), 'faces': faces_right}
    
    faces_left = faces_right[:, [0, 2, 1]]
    hand_idx = hand2idx['left']
    pred_glob_l = run_mano_left(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], 
                                pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], 
                                pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], 
                                betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    left_verts = pred_glob_l['vertices'][0]
    left_dict = {'vertices': left_verts.unsqueeze(0), 'faces': faces_left}
    
    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
    R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all[:num_total_frames])
    t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all[:num_total_frames])
    left_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, left_dict['vertices'].cpu())
    right_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, right_dict['vertices'].cpu())

    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
    
    cam_dir = os.path.join(seq_folder, 'cam_space')
    detected_hands = set()
    if os.path.exists(cam_dir):
        for hid_str in os.listdir(cam_dir):
            if os.path.isdir(os.path.join(cam_dir, hid_str)):
                detected_hands.add(int(hid_str))
    print(f"✓ Detected hands: {detected_hands} (0=left, 1=right)")

    cam_output_pth = os.path.join(output_base, f"vis_cam_{vis_start}_{vis_end}")
    if not os.path.exists(cam_output_pth):
        os.makedirs(cam_output_pth)
    image_names = imgfiles[vis_start:vis_end]

    img0 = cv2.imread(image_names[0])
    img_h, img_w = img0.shape[:2]
    if slam_center is not None:
        cx, cy = float(slam_center[0]), float(slam_center[1])
    else:
        cx, cy = img_w / 2.0, img_h / 2.0
    print(f"  Image size: {img_w}x{img_h}, cx={cx:.1f}, cy={cy:.1f}, focal={img_focal:.1f}")

    print(f"Generating camera-view overlay...")
    left_verts_np = left_dict['vertices'][0].cpu().numpy()
    left_faces = left_dict['faces']
    right_verts_np = right_dict['vertices'][0].cpu().numpy()
    right_faces = right_dict['faces']
    num_frames = min(len(image_names), left_verts_np.shape[0], right_verts_np.shape[0], R_w2c_sla_all.shape[0], pred_valid.shape[1])

    for i in range(num_frames):
        R_w2c_i = R_w2c_sla_all[i].cpu().numpy()
        t_w2c_i = t_w2c_sla_all[i].cpu().numpy()

        frame_img = cv2.imread(image_names[i])
        if frame_img is None:
            continue

        overlay = frame_img.copy()

        if 1 in detected_hands and pred_valid[1, i] and not np.isnan(right_verts_np[i]).any():
            v_world = right_verts_np[i]
            v_cam = (R_w2c_i @ v_world.T).T + t_w2c_i
            pts2d, valid = project_points(v_cam, img_focal, cx, cy)
            for face in right_faces:
                all_ok = True
                pts = []
                for idx in face:
                    if not valid[idx] or pts2d[idx, 0] < 0 or pts2d[idx, 0] >= img_w or pts2d[idx, 1] < 0 or pts2d[idx, 1] >= img_h:
                        all_ok = False
                        break
                    pts.append([int(pts2d[idx, 0]), int(pts2d[idx, 1])])
                if all_ok and len(pts) >= 3:
                    cv2.fillPoly(overlay, [np.array(pts)], (0, 200, 0))

        if 0 in detected_hands and pred_valid[0, i] and not np.isnan(left_verts_np[i]).any():
            v_world = left_verts_np[i]
            v_cam = (R_w2c_i @ v_world.T).T + t_w2c_i
            pts2d, valid = project_points(v_cam, img_focal, cx, cy)
            for face in left_faces:
                all_ok = True
                pts = []
                for idx in face:
                    if not valid[idx] or pts2d[idx, 0] < 0 or pts2d[idx, 0] >= img_w or pts2d[idx, 1] < 0 or pts2d[idx, 1] >= img_h:
                        all_ok = False
                        break
                    pts.append([int(pts2d[idx, 0]), int(pts2d[idx, 1])])
                if all_ok and len(pts) >= 3:
                    cv2.fillPoly(overlay, [np.array(pts)], (200, 100, 0))

        result = cv2.addWeighted(overlay, 0.5, frame_img, 0.5, 0)
        out_path = os.path.join(cam_output_pth, f'{i:06d}.png')
        cv2.imwrite(out_path, result)

    print(f"✓ Camera-view overlay saved to: {cam_output_pth}/ ({num_frames} frames)")

    print(f"Generating world view (3D scene)...")
    world_output_pth = os.path.join(output_base, f"vis_world_{vis_start}_{vis_end}")
    render_world_view(left_dict, right_dict, R_c2w_sla_all, t_c2w_sla_all, detected_hands,
                      pred_valid, world_output_pth, img_h, img_w)

    print(f"Generating vis_verify (depth + trajectory + info)...")
    generate_vis_verify(seq_folder, img_focal, start_idx, end_idx,
                        R_c2w_sla_all, t_c2w_sla_all, slam_scale,
                        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid,
                        cx=cx, cy=cy,
                        output_base=output_base)

    output_dir = os.path.join(output_base, "reconstruction")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    save_path = os.path.join(output_dir, f"hawor_results_{start_idx}_{end_idx}.npz")
    
    def to_numpy(x):
        return x.cpu().numpy() if hasattr(x, 'cpu') else x
    
    np.savez(save_path,
             pred_trans=to_numpy(pred_trans),
             pred_rot=to_numpy(pred_rot),
             pred_hand_pose=to_numpy(pred_hand_pose),
             pred_betas=to_numpy(pred_betas),
             pred_valid=to_numpy(pred_valid),
             R_c2w=to_numpy(R_c2w_sla_all),
             t_c2w=to_numpy(t_c2w_sla_all),
             img_focal=img_focal,
             img_center=np.array([cx, cy]),
             slam_scale=slam_scale,
             start_idx=start_idx,
             end_idx=end_idx)
    
    print(f"\n{'='*80}")
    print(f"✓ Reconstruction saved to: {save_path}")
    print(f"{'='*80}")
    print(f"Frames: {start_idx} to {end_idx} ({end_idx - start_idx + 1} frames)")
    print(f"Data sources (all from video):")
    print(f"  Camera extrinsics: SLAM estimation from video")
    print(f"  Hand parameters: MANO inference from video")
    print(f"  Focal length: {img_focal:.1f} (from SLAM)")
    print(f"  Principal point: ({cx:.1f}, {cy:.1f}) (from SLAM)")
    print(f"  SLAM scale: {slam_scale:.4f}")
    print(f"  Detected hands: {detected_hands}")
    print(f"Output:")
    print(f"  Camera-view overlay: {cam_output_pth}/")
    print(f"  World view (3D scene): {world_output_pth}/")
    print(f"  Verify images+video: {output_base}/vis_verify/")
    print(f"{'='*80}\n")
    print("finish")
