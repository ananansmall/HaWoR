#!/usr/bin/env python3
"""
轻量级测试脚本：验证 HOI4D 视频 121_C5_CellPhone_161deg 的两个 HaWoR 输出

不重新跑完整管线，直接加载已有的 reconstruction npz 数据，
对比 hoiger4d (DROID-SLAM) 和 hoiger4d_vggt-omega (VGGT-Omega) 的差异。

Usage:
    cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR
    conda activate hawor
    python3 test_hoi4d_demo.py
"""

import os
import numpy as np

# ========== 路径配置 ==========
VIDEO_NAME = "121_C5_CellPhone_161deg"
VIDEO_PATH = "/mnt/data_8THDD/lza/dataset/HOI4D_RGB/HOI4D_selected_200/121_C5_CellPhone_161deg.mp4"

HAWOR_DIR = "/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR"
OUTPUT_BASE = os.path.join(HAWOR_DIR, "output")

# 两个输出的 reconstruction npz
paths = {
    "hoi4d_droid": os.path.join(OUTPUT_BASE, "hoi4d", "reconstruction", "hawor_results_0_600.npz"),
    "hoi4d_vggt_omega": os.path.join(OUTPUT_BASE, "hoi4d_vggt-omega", "reconstruction", "hawor_results_0_600.npz"),
}

# RAS 输出目录
RAS_DEPTH_DIR = "/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/hoi4d1_vggt_omega/depth"
RAS_EXTRINSICS_DIR = "/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/hoi4d1_vggt_omega/extrinsics"
RAS_INTRINSIC = "/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/hoi4d1_vggt_omega/intrinsic.txt"


def check_file(path, desc):
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = f"OK ({size/1e6:.1f} MB)" if exists else "MISSING"
    print(f"  [{status}] {desc}: {path}")
    return exists


def load_npz_safe(path):
    """加载 npz，打印基本信息"""
    d = np.load(path, allow_pickle=True)
    print(f"\n  文件: {path}")
    print(f"  键: {list(d.keys())}")
    for k in d.keys():
        v = d[k]
        if hasattr(v, 'shape'):
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
            if v.ndim >= 2:
                print(f"         range: [{np.nanmin(v):.6f}, {np.nanmax(v):.6f}]")
        else:
            print(f"    {k}: {v}")
    return d


def compare_cameras(d1, d2, label1, label2):
    """对比两组相机外参"""
    t1 = d1['t_c2w']
    t2 = d2['t_c2w']
    n1 = np.linalg.norm(t1, axis=1)
    n2 = np.linalg.norm(t2, axis=1)

    print(f"\n  === 相机外参 t_c2w 对比 ===")
    print(f"  {label1}: norm range [{n1.min():.6f}, {n1.max():.6f}], mean={n1.mean():.6f}")
    print(f"  {label2}: norm range [{n2.min():.6f}, {n2.max():.6f}], mean={n2.mean():.6f}")

    R1 = d1['R_c2w']
    R2 = d2['R_c2w']
    rot_diff = np.zeros(len(R1))
    for i in range(len(R1)):
        diff = R1[i] @ R2[i].T
        # trace -> angle
        angle = np.arccos(np.clip((np.trace(diff) - 1) / 2, -1, 1))
        rot_diff[i] = np.degrees(angle)
    print(f"  旋转角度差: mean={rot_diff.mean():.2f}°, max={rot_diff.max():.2f}°")

    # 逐帧对比
    print(f"\n  逐帧对比 (sample):")
    for i in [0, 50, 100, 200, 300, 400, 500, 599]:
        if i >= len(t1):
            break
        print(f"    frame {i}:")
        print(f"      {label1}: t={t1[i]}, ||t||={n1[i]:.6f}")
        print(f"      {label2}: t={t2[i]}, ||t||={n2[i]:.6f}")
        print(f"      trans[1]: {label1}={d1['pred_trans'][1,i]}, {label2}={d2['pred_trans'][1,i]}")


def compare_hand_params(d1, d2, label1, label2):
    """对比手部参数"""
    print(f"\n  === 手部参数 pred_trans 对比 ===")
    tr1 = d1['pred_trans']
    tr2 = d2['pred_trans']
    n1 = np.linalg.norm(tr1, axis=-1)
    n2 = np.linalg.norm(tr2, axis=-1)

    for hand in [0, 1]:
        hname = "left" if hand == 0 else "right"
        diff = np.abs(tr1[hand] - tr2[hand])
        print(f"  Hand {hand} ({hname}):")
        print(f"    {label1} norm range: [{n1[hand].min():.6f}, {n1[hand].max():.6f}]")
        print(f"    {label2} norm range: [{n2[hand].min():.6f}, {n2[hand].max():.6f}]")
        print(f"    绝对差: mean={diff.mean():.6f}, max={diff.max():.6f}")
        print(f"    相对差: mean={(diff / (np.abs(tr1[hand]) + 1e-8)).mean():.4f}")


def check_ras_data():
    """检查 RAS 输出数据"""
    print(f"\n  === RAS 输出数据检查 ===")

    # 深度图
    if os.path.exists(RAS_DEPTH_DIR):
        depth_files = sorted([f for f in os.listdir(RAS_DEPTH_DIR) if f.endswith('.png')])
        print(f"  深度图: {len(depth_files)} 张")
        if depth_files:
            import cv2
            sample = cv2.imread(os.path.join(RAS_DEPTH_DIR, depth_files[0]), cv2.IMREAD_UNCHANGED)
            if sample is not None:
                print(f"    首帧: shape={sample.shape}, dtype={sample.dtype}, range=[{sample.min()}, {sample.max()}]")
                print(f"    深度值 / 1000 = 米制深度")

    # 外参
    if os.path.exists(RAS_EXTRINSICS_DIR):
        ext_files = sorted([f for f in os.listdir(RAS_EXTRINSICS_DIR) if f.endswith('.txt')])
        print(f"  外参: {len(ext_files)} 个")
        if ext_files:
            ext = np.loadtxt(os.path.join(RAS_EXTRINSICS_DIR, ext_files[0]))
            print(f"    首帧: shape={ext.shape}")

    # 内参
    if os.path.exists(RAS_INTRINSIC):
        intrinsic = np.loadtxt(RAS_INTRINSIC)
        print(f"  RAS 内参:\n{intrinsic}")

    # mask
    mask_path = os.path.join(HAWOR_DIR, "output", "hoi4d_vggt-omega", "tracks_0_600", "model_masks.npy")
    if os.path.exists(mask_path):
        masks = np.load(mask_path)
        print(f"  model_masks: shape={masks.shape}, dtype={masks.dtype}")


def main():
    print("=" * 70)
    print(f"HaWoR Demo 测试: {VIDEO_NAME}")
    print("=" * 70)

    # 1. 检查视频文件
    print(f"\n  === 视频文件 ===")
    check_file(VIDEO_PATH, "HOI4D 视频")

    # 2. 检查输出文件
    print(f"\n  === 输出文件检查 ===")
    for label, path in paths.items():
        check_file(path, f"{label} reconstruction")

    # 额外检查
    check_file(os.path.join(OUTPUT_BASE, "hoi4d_vggt-omega", "vggt_omega_cam", "vggt_omega_cam.npz"),
               "VGGT-Omega 相机轨迹")
    check_file(os.path.join(OUTPUT_BASE, "hoi4d", "reconstruction", "hawor_results_0_600.npz"),
               "DROID-SLAM reconstruction")

    # 3. 加载并对比
    print(f"\n  === 数据加载 ===")
    datasets = {}
    for label, path in paths.items():
        if os.path.exists(path):
            datasets[label] = np.load(path, allow_pickle=True)
            print(f"  已加载: {label}")
        else:
            print(f"  跳过 (不存在): {label}")

    if len(datasets) < 2:
        print("\n  至少需要两个数据集才能对比")
        return

    labels = list(datasets.keys())
    d1, d2 = datasets[labels[0]], datasets[labels[1]]
    l1, l2 = labels[0], labels[1]

    # 4. 对比
    compare_cameras(d1, d2, l1, l2)
    compare_hand_params(d1, d2, l1, l2)

    # 5. 焦距对比
    print(f"\n  === 焦距对比 ===")
    f1 = d1.get('img_focal', 'N/A')
    f2 = d2.get('img_focal', 'N/A')
    print(f"  {l1}: focal={f1}")
    print(f"  {l2}: focal={f2}")
    if hasattr(f1, '__float__') and hasattr(f2, '__float__'):
        print(f"  焦距差: {abs(float(f1) - float(f2)):.1f} px")

    # 6. RAS 数据检查
    check_ras_data()

    # 7. 结论
    print(f"\n  === 总结 ===")
    print(f"  两个 demo 都可以用于此视频，但它们是从视频开始的完整管线")
    print(f"  - demov2.py: 使用 DROID-SLAM 估计相机")
    print(f"  - demo_vggt_omega.py: 使用 VGGT-Omega 估计相机")
    print(f"  你已经有两者的输出结果在 output/hoi4d/ 和 output/hoi4d_vggt-omega/")
    print(f"  上述对比展示了两种相机估计方法对结果的影响")
    print(f"\n  如果要重新运行 demo:")
    print(f"    python3 demov2.py --video_path {VIDEO_PATH}")
    print(f"    python3 demo-vggt-omega/demo_vggt_omega.py --video_path {VIDEO_PATH}")
    print(f"  注意: 这需要 GPU (~47GB for VGGT-Omega) 和较长的推理时间")


if __name__ == '__main__':
    main()
