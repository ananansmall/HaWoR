import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from lib.eval_utils.eval_utils import (
    batch_compute_similarity_transform_torch,
    compute_jpe,
    first_align_joints,
    global_align_joints,
    compute_rte,
    compute_error_accel,
)


def test_pa_mpjpe():
    print("=" * 60)
    print("测试 PA-MPJPE (Procrustes-Aligned MPJPE)")
    print("=" * 60)

    T, J = 50, 21
    gt_joints = torch.randn(T, J, 3) * 0.1 + torch.tensor([0.5, 0.3, 0.2])

    pred_joints = gt_joints + torch.randn(T, J, 3) * 0.02

    pred_pa = batch_compute_similarity_transform_torch(pred_joints, gt_joints)
    pa_mpjpe = compute_jpe(gt_joints, pred_pa)
    print(f"  小噪声扰动: PA-MPJPE = {pa_mpjpe.mean():.6f} m = {pa_mpjpe.mean() * 1000:.3f} mm")

    R = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)
    t = torch.tensor([1.0, 2.0, 3.0])
    s = 2.0
    pred_transformed = s * torch.einsum('ij,tnj->tni', R, gt_joints) + t

    pred_pa = batch_compute_similarity_transform_torch(pred_transformed, gt_joints)
    pa_mpjpe = compute_jpe(gt_joints, pred_pa)
    print(f"  旋转+平移+缩放变换: PA-MPJPE = {pa_mpjpe.mean():.6f} m = {pa_mpjpe.mean() * 1000:.3f} mm (应接近0)")

    pred_random = torch.randn(T, J, 3)
    pred_pa = batch_compute_similarity_transform_torch(pred_random, gt_joints)
    pa_mpjpe = compute_jpe(gt_joints, pred_pa)
    print(f"  完全随机预测: PA-MPJPE = {pa_mpjpe.mean():.6f} m = {pa_mpjpe.mean() * 1000:.3f} mm")

    print()


def test_w_mpjpe():
    print("=" * 60)
    print("测试 W-MPJPE (World-space MPJPE)")
    print("=" * 60)

    T, J = 50, 21
    gt_joints = torch.randn(T, J, 3) * 0.1 + torch.tensor([0.5, 0.3, 0.2])

    pred_joints = gt_joints + torch.randn(T, J, 3) * 0.02

    w_j3d = first_align_joints(gt_joints, pred_joints)
    w_mpjpe = compute_jpe(gt_joints, w_j3d)
    print(f"  小噪声扰动: W-MPJPE = {w_mpjpe.mean():.6f} m = {w_mpjpe.mean() * 1000:.3f} mm")

    R = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)
    t = torch.tensor([1.0, 2.0, 3.0])
    s = 2.0
    pred_transformed = s * torch.einsum('ij,tnj->tni', R, gt_joints) + t

    w_j3d = first_align_joints(gt_joints, pred_transformed)
    w_mpjpe = compute_jpe(gt_joints, w_j3d)
    print(f"  旋转+平移+缩放变换: W-MPJPE = {w_mpjpe.mean():.6f} m = {w_mpjpe.mean() * 1000:.3f} mm")

    pred_random = torch.randn(T, J, 3)
    w_j3d = first_align_joints(gt_joints, pred_random)
    w_mpjpe = compute_jpe(gt_joints, w_j3d)
    print(f"  完全随机预测: W-MPJPE = {w_mpjpe.mean():.6f} m = {w_mpjpe.mean() * 1000:.3f} mm")

    print()


def test_wa_mpjpe():
    print("=" * 60)
    print("测试 WA-MPJPE (World-Aligned MPJPE)")
    print("=" * 60)

    T, J = 50, 21
    gt_joints = torch.randn(T, J, 3) * 0.1 + torch.tensor([0.5, 0.3, 0.2])

    pred_joints = gt_joints + torch.randn(T, J, 3) * 0.02

    wa_j3d = global_align_joints(gt_joints, pred_joints)
    wa_mpjpe = compute_jpe(gt_joints, wa_j3d)
    print(f"  小噪声扰动: WA-MPJPE = {wa_mpjpe.mean():.6f} m = {wa_mpjpe.mean() * 1000:.3f} mm")

    R = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)
    t = torch.tensor([1.0, 2.0, 3.0])
    s = 2.0
    pred_transformed = s * torch.einsum('ij,tnj->tni', R, gt_joints) + t

    wa_j3d = global_align_joints(gt_joints, pred_transformed)
    wa_mpjpe = compute_jpe(gt_joints, wa_j3d)
    print(f"  旋转+平移+缩放变换: WA-MPJPE = {wa_mpjpe.mean():.6f} m = {wa_mpjpe.mean() * 1000:.3f} mm (应接近0)")

    pred_random = torch.randn(T, J, 3)
    wa_j3d = global_align_joints(gt_joints, pred_random)
    wa_mpjpe = compute_jpe(gt_joints, wa_j3d)
    print(f"  完全随机预测: WA-MPJPE = {wa_mpjpe.mean():.6f} m = {wa_mpjpe.mean() * 1000:.3f} mm")

    print()


def test_rte():
    print("=" * 60)
    print("测试 RTE (Relative Translation Error)")
    print("=" * 60)

    T = 50
    gt_trans = torch.cumsum(torch.randn(T, 3) * 0.01, dim=0)

    pred_trans = gt_trans + torch.randn(T, 3) * 0.005

    rte = compute_rte(gt_trans, pred_trans)
    print(f"  小噪声扰动: RTE = {rte.mean():.6f} (归一化比值)")

    pred_trans_shifted = gt_trans + torch.tensor([0.1, 0.2, 0.3])

    rte = compute_rte(gt_trans, pred_trans_shifted)
    print(f"  全局平移偏移: RTE = {rte.mean():.6f} (归一化比值, 应接近0)")

    pred_trans_scaled = gt_trans * 1.5
    rte = compute_rte(gt_trans, pred_trans_scaled)
    print(f"  缩放变换: RTE = {rte.mean():.6f} (归一化比值)")

    pred_trans_random = torch.randn(T, 3)
    rte = compute_rte(gt_trans, pred_trans_random)
    print(f"  完全随机预测: RTE = {rte.mean():.6f} (归一化比值)")

    print()


def test_accel():
    print("=" * 60)
    print("测试 Accel (Acceleration Error)")
    print("=" * 60)

    T, J = 50, 21
    gt_joints = torch.randn(T, J, 3) * 0.1

    pred_joints = gt_joints.clone()

    accel = compute_error_accel(joints_gt=gt_joints, joints_pred=pred_joints)
    print(f"  完全相同输入: Accel = {accel.mean():.6f} (应接近0)")

    pred_joints = gt_joints + torch.randn(T, J, 3) * 0.005
    accel = compute_error_accel(joints_gt=gt_joints, joints_pred=pred_joints)
    print(f"  小噪声扰动: Accel = {accel.mean():.6f}")

    pred_jitter = gt_joints.clone()
    for i in range(1, T - 1):
        pred_jitter[i] += torch.randn(J, 3) * 0.05
    accel = compute_error_accel(joints_gt=gt_joints, joints_pred=pred_jitter)
    print(f"  抖动/不平稳运动: Accel = {accel.mean():.6f}")

    pred_random = torch.randn(T, J, 3)
    accel = compute_error_accel(joints_gt=gt_joints, joints_pred=pred_random)
    print(f"  完全随机预测: Accel = {accel.mean():.6f}")

    fps = 30
    accel_scaled = accel * (fps ** 2)
    print(f"  完全随机预测(缩放后, fps={fps}): Accel = {accel_scaled.mean():.3f}")

    print()


def test_all():
    print("\n" + "=" * 60)
    print("HaWoR 评价指标综合测试 (合成数据)")
    print("=" * 60 + "\n")

    torch.manual_seed(42)
    np.random.seed(42)

    test_pa_mpjpe()
    test_w_mpjpe()
    test_wa_mpjpe()
    test_rte()
    test_accel()

    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_all()
