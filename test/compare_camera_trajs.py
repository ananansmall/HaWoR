#!/usr/bin/env python3
"""
相机轨迹对比：demov2 (DROID-SLAM) vs VGGT-Omega

用法：
  python compare_camera_trajs.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

OUTPUT_DIR = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/output'
DEMOV2_DIR = f'{OUTPUT_DIR}/hoi4d'
VGGT_DIR = f'{OUTPUT_DIR}/hoi4d_vggt-omega'


def load_cameras(base_dir, source='npz'):
    if source == 'npz':
        path = f'{base_dir}/reconstruction/hawor_results_0_600.npz'
        data = np.load(path)
        R = data['R_c2w']
        t = data['t_c2w']
        focal = float(data['img_focal'])
    elif source == 'vggt':
        path = f'{base_dir}/vggt_omega_cam/vggt_omega_cam.npz'
        data = np.load(path)
        R = data['R_c2w']
        t = data['t_c2w']
        focal = float(data['img_focal'])
    return R, t, focal


def compute_ate(R1, t1, R2, t2):
    """计算绝对轨迹误差 (ATE)，两个轨迹已经对齐到同一坐标系"""
    errors = np.linalg.norm(t1 - t2, axis=1)
    return errors


def main():
    print("Loading Demov2 (DROID-SLAM) camera poses...")
    R_demo, t_demo, focal_demo = load_cameras(DEMOV2_DIR, source='npz')
    print(f"  Demov2: {R_demo.shape[0]} frames, focal={focal_demo:.1f}")

    print("Loading VGGT-Omega camera poses...")
    R_vggt, t_vggt, focal_vggt = load_cameras(VGGT_DIR, source='vggt')
    print(f"  VGGT: {R_vggt.shape[0]} frames, focal={focal_vggt:.1f}")

    # Demov2 用的是 DROID-SLAM 坐标系 (SLAM scale)
    # VGGT 输出经过 R_x 变换对齐到 SLAM 坐标系（见 demo_vggt_omega.py L718-722）
    # 所以可以直接对比
    print(f"\nFocal difference: {abs(focal_demo - focal_vggt):.1f}")
    print(f"Demov2 focal: {focal_demo:.1f}, VGGT focal: {focal_vggt:.1f}")

    # 计算 ATE
    ate = compute_ate(R_demo, t_demo, R_vggt, t_vggt)
    print(f"\nATE Statistics (Euclidean distance between trajectories):")
    print(f"  Mean:   {ate.mean():.6f} m")
    print(f"  Median: {np.median(ate):.6f} m")
    print(f"  Max:    {ate.max():.6f} m")
    print(f"  Min:    {ate.min():.6f} m")
    print(f"  Std:    {ate.std():.6f} m")

    # 检查是否有 NaN 或 Inf
    nan_demo = np.sum(np.isnan(t_demo))
    nan_vggt = np.sum(np.isnan(t_vggt))
    print(f"\nNaN counts: Demov2={nan_demo}, VGGT={nan_vggt}")

    # ========== 可视化 ==========
    fig = plt.figure(figsize=(16, 12))

    # 1. 3D 轨迹对比
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot(t_demo[:, 0], t_demo[:, 1], t_demo[:, 2], 'b-', alpha=0.6, label='Demov2 (DROID-SLAM)')
    ax.plot(t_vggt[:, 0], t_vggt[:, 1], t_vggt[:, 2], 'r-', alpha=0.6, label='VGGT-Omega')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory: Demov2 vs VGGT-Omega')
    ax.legend()
    ax.view_init(elev=25, azim=45)

    # 2. 轨迹误差随帧变化
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(range(len(ate)), ate, 'g-', linewidth=1)
    ax.axhline(y=ate.mean(), color='r', linestyle='--', label=f'Mean ATE = {ate.mean():.4f} m')
    ax.set_xlabel('Frame')
    ax.set_ylabel('ATE (m)')
    ax.set_title('Absolute Trajectory Error per Frame')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 旋转误差（角度）
    ax = fig.add_subplot(2, 2, 3)
    angle_errors = []
    for i in range(len(R_demo)):
        cos_theta = np.clip((np.trace(R_demo[i].T @ R_vggt[i]) - 1) / 2, -1, 1)
        angle_errors.append(np.degrees(np.arccos(cos_theta)))
    angle_errors = np.array(angle_errors)
    ax.plot(range(len(angle_errors)), angle_errors, 'm-', linewidth=1)
    ax.axhline(y=angle_errors.mean(), color='r', linestyle='--', label=f'Mean = {angle_errors.mean():.2f}°')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Rotation Error (deg)')
    ax.set_title('Rotation Error per Frame')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 速度对比（帧间位移）
    ax = fig.add_subplot(2, 2, 4)
    vel_demo = np.linalg.norm(np.diff(t_demo, axis=0), axis=1)
    vel_vggt = np.linalg.norm(np.diff(t_vggt, axis=0), axis=1)
    ax.plot(range(len(vel_demo)), vel_demo, 'b-', alpha=0.6, label='Demov2 velocity')
    ax.plot(range(len(vel_vggt)), vel_vggt, 'r-', alpha=0.6, label='VGGT velocity')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Displacement per frame (m)')
    ax.set_title('Camera Velocity (Inter-frame Displacement)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/output/camera_traj_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")

    # 保存 ATE 数据
    np.savez('/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/output/camera_traj_ate.npz',
             ate=ate, angle_errors=angle_errors,
             t_demo=t_demo, t_vggt=t_vggt)
    print(f"✓ ATE data saved to: /mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/output/camera_traj_ate.npz")


if __name__ == '__main__':
    main()