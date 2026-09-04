#!/usr/bin/env python3
"""
相机轨迹对比 + Sim3 对齐：demov2 (DROID-SLAM) vs VGGT-Omega

用法：
  python compare_camera_trajs_aligned.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

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


def compute_sim3_params(tgt, src):
    """
    计算 Sim3 变换参数 (scale, R, t)，使 src 轨迹对齐到 tgt 轨迹。
    最小化 sum_i ||tgt_i - (s * R * src_i + t)||^2
    """
    tgt = tgt.astype(np.float64)
    src = src.astype(np.float64)

    # 去中心化
    tgt_mean = tgt.mean(axis=0, keepdims=True)
    src_mean = src.mean(axis=0, keepdims=True)
    tgt_c = tgt - tgt_mean
    src_c = src - src_mean

    def loss(params):
        s = np.exp(params[0])  # 确保 scale > 0
        # Euler angles for rotation
        rot = R.from_euler('xyz', params[1:4])
        R_mat = rot.as_matrix()
        t = params[4:7]
        aligned = s * (src_c @ R_mat.T) + t
        return np.sum((tgt_c - aligned) ** 2)

    # Initial guess: scale from ratio of standard deviations, identity rotation/translation
    scale_init = np.log(np.std(tgt_c) / np.std(src_c))
    x0 = [scale_init, 0, 0, 0, 0, 0, 0]

    res = minimize(loss, x0, method='L-BFGS-B')
    params = res.x

    s = np.exp(params[0])
    rot = R.from_euler('xyz', params[1:4])
    R_mat = rot.as_matrix()
    t = params[4:7]
    return s, R_mat, t, res.fun


def apply_sim3(t, s, R_mat, t_trans):
    """Apply Sim3: t_aligned = s * R @ t + t_trans"""
    return s * (t @ R_mat.T) + t_trans


def apply_sim3_rot(R_arr, s, R_sim3, t_sim3):
    """Apply Sim3 to rotation matrices: R_aligned = R_sim3 @ R_src"""
    return R_sim3 @ R_arr


def compute_ate(t1, t2):
    errors = np.linalg.norm(t1 - t2, axis=1)
    return errors


def compute_rot_err(R1, R2):
    angle_errors = []
    for i in range(len(R1)):
        cos_theta = np.clip((np.trace(R1[i].T @ R2[i]) - 1) / 2, -1, 1)
        angle_errors.append(np.degrees(np.arccos(cos_theta)))
    return np.array(angle_errors)


def main():
    print("Loading Demov2 (DROID-SLAM) camera poses...")
    R_demo, t_demo, focal_demo = load_cameras(DEMOV2_DIR, source='npz')
    print(f"  Demov2: {R_demo.shape[0]} frames, focal={focal_demo:.1f}")

    print("Loading VGGT-Omega camera poses...")
    R_vggt, t_vggt, focal_vggt = load_cameras(VGGT_DIR, source='vggt')
    print(f"  VGGT: {R_vggt.shape[0]} frames, focal={focal_vggt:.1f}")

    # ===== Step 1: 原始对比（不对齐）=====
    print("\n" + "=" * 60)
    print("BEFORE Alignment (raw coordinates)")
    print("=" * 60)
    ate_raw = compute_ate(t_demo, t_vggt)
    rot_err_raw = compute_rot_err(R_demo, R_vggt)
    print(f"  ATE Mean: {ate_raw.mean():.6f} m")
    print(f"  Rot Error Mean: {rot_err_raw.mean():.2f}°")

    # ===== Step 2: Sim3 对齐 =====
    print("\nComputing Sim3 alignment (VGGT -> Demov2)...")
    s, R_sim3, t_sim3, loss = compute_sim3_params(t_demo, t_vggt)
    print(f"  Scale: {s:.6f}")
    print(f"  Rotation (deg): {R.from_matrix(R_sim3).as_euler('xyz', degrees=True)}")
    print(f"  Translation: {t_sim3}")
    print(f"  Optimization loss: {loss:.6f}")

    t_vggt_aligned = apply_sim3(t_vggt, s, R_sim3, t_sim3)
    R_vggt_aligned = np.array([apply_sim3_rot(r, s, R_sim3, t_sim3) for r in R_vggt])

    # ===== Step 3: 对齐后对比 =====
    print("\n" + "=" * 60)
    print("AFTER Sim3 Alignment (VGGT aligned to Demov2)")
    print("=" * 60)
    ate_aligned = compute_ate(t_demo, t_vggt_aligned)
    rot_err_aligned = compute_rot_err(R_demo, R_vggt_aligned)
    print(f"  ATE Mean:   {ate_aligned.mean():.6f} m")
    print(f"  ATE Median: {np.median(ate_aligned):.6f} m")
    print(f"  ATE Max:    {ate_aligned.max():.6f} m")
    print(f"  ATE Std:    {ate_aligned.std():.6f} m")
    print(f"  Rot Mean:   {rot_err_aligned.mean():.2f}°")
    print(f"  Rot Max:    {rot_err_aligned.max():.2f}°")
    print(f"  Rot Std:    {rot_err_aligned.std():.2f}°")

    # 计算相对轨迹误差 (RTE)
    # RTE measures how much the trajectory diverges over time
    rtes = []
    for w in [10, 50, 100, 200]:
        errors = []
        for i in range(len(t_demo) - w):
            t1_window = t_demo[i:i+w] - t_demo[i]
            t2_window = t_vggt_aligned[i:i+w] - t_vggt_aligned[i]
            # Procrustes alignment of window
            t1_c = t1_window - t1_window.mean(axis=0)
            t2_c = t2_window - t2_window.mean(axis=0)
            H = t1_c.T @ t2_c
            U, S, Vt = np.linalg.svd(H)
            R_win = U @ Vt
            err = np.mean(np.linalg.norm(t1_c @ R_win - t2_c, axis=1))
            errors.append(err)
        rtes.append(np.mean(errors))
    print(f"\n  RTE (window=10):  {rtes[0]:.6f} m")
    print(f"  RTE (window=50):  {rtes[1]:.6f} m")
    print(f"  RTE (window=100): {rtes[2]:.6f} m")
    print(f"  RTE (window=200): {rtes[3]:.6f} m")

    # ===== 可视化 =====
    fig = plt.figure(figsize=(18, 14))

    # 1. 3D 轨迹对比 - 原始
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.plot(t_demo[:, 0], t_demo[:, 1], t_demo[:, 2], 'b-', alpha=0.6, label='Demov2')
    ax.plot(t_vggt[:, 0], t_vggt[:, 1], t_vggt[:, 2], 'r-', alpha=0.6, label='VGGT (raw)')
    ax.set_title('BEFORE: Raw Coordinates')
    ax.legend(fontsize=8)
    ax.view_init(elev=25, azim=45)

    # 2. 3D 轨迹对比 - 对齐后
    ax = fig.add_subplot(2, 3, 2, projection='3d')
    ax.plot(t_demo[:, 0], t_demo[:, 1], t_demo[:, 2], 'b-', alpha=0.6, label='Demov2')
    ax.plot(t_vggt_aligned[:, 0], t_vggt_aligned[:, 1], t_vggt_aligned[:, 2], 'r-', alpha=0.6, label='VGGT (aligned)')
    ax.set_title('AFTER: Sim3 Aligned')
    ax.legend(fontsize=8)
    ax.view_init(elev=25, azim=45)

    # 3. 轨迹误差对比 (BEFORE vs AFTER)
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(range(len(ate_raw)), ate_raw, 'b-', alpha=0.5, label=f'Before (mean={ate_raw.mean():.3f}m)', linewidth=1)
    ax.plot(range(len(ate_aligned)), ate_aligned, 'r-', alpha=0.5, label=f'After (mean={ate_aligned.mean():.3f}m)', linewidth=1)
    ax.set_xlabel('Frame')
    ax.set_ylabel('ATE (m)')
    ax.set_title('ATE: Before vs After Alignment')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. 旋转误差对比
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(range(len(rot_err_raw)), rot_err_raw, 'b-', alpha=0.5, label=f'Before (mean={rot_err_raw.mean():.1f}°)', linewidth=1)
    ax.plot(range(len(rot_err_aligned)), rot_err_aligned, 'r-', alpha=0.5, label=f'After (mean={rot_err_aligned.mean():.1f}°)', linewidth=1)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Rotation Error (deg)')
    ax.set_title('Rotation Error: Before vs After')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. 对齐后的 ATE 详情
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(range(len(ate_aligned)), ate_aligned, 'g-', linewidth=1.2)
    ax.axhline(y=ate_aligned.mean(), color='r', linestyle='--', label=f'Mean = {ate_aligned.mean():.4f} m')
    ax.axhline(y=ate_aligned.mean() + 2 * ate_aligned.std(), color='orange', linestyle=':', label=f'+2σ = {ate_aligned.mean() + 2 * ate_aligned.std():.4f} m')
    ax.set_xlabel('Frame')
    ax.set_ylabel('ATE (m)')
    ax.set_title('Post-Alignment ATE Detail')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. 速度对比
    ax = fig.add_subplot(2, 3, 6)
    vel_demo = np.linalg.norm(np.diff(t_demo, axis=0), axis=1)
    vel_vggt_aligned = np.linalg.norm(np.diff(t_vggt_aligned, axis=0), axis=1)
    ax.plot(range(len(vel_demo)), vel_demo, 'b-', alpha=0.6, label='Demov2 velocity')
    ax.plot(range(len(vel_vggt_aligned)), vel_vggt_aligned, 'r-', alpha=0.6, label='VGGT velocity')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Displacement per frame (m)')
    ax.set_title('Camera Velocity (Inter-frame)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/output/camera_traj_comparison_aligned.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")

    # 保存对齐数据
    np.savez('/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/output/camera_traj_aligned.npz',
             t_demo=t_demo, t_vggt=t_vggt, t_vggt_aligned=t_vggt_aligned,
             R_demo=R_demo, R_vggt_aligned=R_vggt_aligned,
             ate_raw=ate_raw, ate_aligned=ate_aligned,
             rot_err_raw=rot_err_raw, rot_err_aligned=rot_err_aligned,
             sim3_scale=s, sim3_R=R_sim3, sim3_t=t_sim3)
    print(f"✓ Aligned data saved to: /mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/output/camera_traj_aligned.npz")


if __name__ == '__main__':
    main()