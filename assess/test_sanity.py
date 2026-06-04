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


def test_pa_mpjpe_sanity():
    print("=" * 60)
    print("PA-MPJPE 合理性检查")
    print("=" * 60)

    T, J = 30, 21
    gt = torch.randn(T, J, 3)

    pred_pa = batch_compute_similarity_transform_torch(gt, gt)
    pa_mpjpe = compute_jpe(gt, pred_pa)
    print(f"  完全相同输入: PA-MPJPE = {pa_mpjpe.mean():.8f} (应接近0)")
    assert pa_mpjpe.mean() < 1e-5, f"PA-MPJPE for identical inputs should be ~0, got {pa_mpjpe.mean()}"

    R = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)
    t = torch.tensor([3.0, -2.0, 5.0])
    s = 1.5
    pred = s * torch.einsum('ij,tnj->tni', R, gt) + t
    pred_pa = batch_compute_similarity_transform_torch(pred, gt)
    pa_mpjpe = compute_jpe(gt, pred_pa)
    print(f"  相似变换(旋转+平移+缩放): PA-MPJPE = {pa_mpjpe.mean():.8f} (应接近0)")
    assert pa_mpjpe.mean() < 1e-3, f"PA-MPJPE for similarity-transformed inputs should be ~0, got {pa_mpjpe.mean()}"

    print("  ✓ PA-MPJPE 合理性检查通过\n")


def test_w_mpjpe_sanity():
    print("=" * 60)
    print("W-MPJPE 合理性检查")
    print("=" * 60)

    T, J = 30, 21
    gt = torch.randn(T, J, 3)

    w_j3d = first_align_joints(gt, gt)
    w_mpjpe = compute_jpe(gt, w_j3d)
    print(f"  完全相同输入: W-MPJPE = {w_mpjpe.mean():.8f} (应接近0)")
    assert w_mpjpe.mean() < 1e-5, f"W-MPJPE for identical inputs should be ~0, got {w_mpjpe.mean()}"

    R = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)
    t = torch.tensor([3.0, -2.0, 5.0])
    s = 1.5
    pred = s * torch.einsum('ij,tnj->tni', R, gt) + t
    w_j3d = first_align_joints(gt, pred)
    w_mpjpe = compute_jpe(gt, w_j3d)
    print(f"  相似变换: W-MPJPE = {w_mpjpe.mean():.8f} (仅前两帧对齐, 不一定为0)")

    print("  ✓ W-MPJPE 合理性检查通过\n")


def test_wa_mpjpe_sanity():
    print("=" * 60)
    print("WA-MPJPE 合理性检查")
    print("=" * 60)

    T, J = 30, 21
    gt = torch.randn(T, J, 3)

    wa_j3d = global_align_joints(gt, gt)
    wa_mpjpe = compute_jpe(gt, wa_j3d)
    print(f"  完全相同输入: WA-MPJPE = {wa_mpjpe.mean():.8f} (应接近0)")
    assert wa_mpjpe.mean() < 1e-5, f"WA-MPJPE for identical inputs should be ~0, got {wa_mpjpe.mean()}"

    R = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)
    t = torch.tensor([3.0, -2.0, 5.0])
    s = 1.5
    pred = s * torch.einsum('ij,tnj->tni', R, gt) + t
    wa_j3d = global_align_joints(gt, pred)
    wa_mpjpe = compute_jpe(gt, wa_j3d)
    print(f"  相似变换: WA-MPJPE = {wa_mpjpe.mean():.8f} (全局对齐, 应接近0)")
    assert wa_mpjpe.mean() < 1e-3, f"WA-MPJPE for similarity-transformed inputs should be ~0, got {wa_mpjpe.mean()}"

    print("  ✓ WA-MPJPE 合理性检查通过\n")


def test_rte_sanity():
    print("=" * 60)
    print("RTE 合理性检查")
    print("=" * 60)

    T = 30
    gt_trans = torch.cumsum(torch.randn(T, 3) * 0.01, dim=0)

    pred_shifted = gt_trans + torch.tensor([1.0, 2.0, 3.0])
    rte = compute_rte(gt_trans, pred_shifted)
    print(f"  全局平移偏移: RTE = {rte.mean():.8f} (全局对齐后应接近0)")
    assert rte.mean() < 0.05, f"RTE for globally shifted trajectory should be ~0, got {rte.mean()}"

    R = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)
    pred_rotated = torch.einsum('ij,nj->ni', R, gt_trans)
    rte = compute_rte(gt_trans, pred_rotated)
    print(f"  旋转变换: RTE = {rte.mean():.8f} (全局对齐后应接近0)")
    assert rte.mean() < 0.05, f"RTE for rotated trajectory should be ~0, got {rte.mean()}"

    print("  ✓ RTE 合理性检查通过\n")


def test_accel_sanity():
    print("=" * 60)
    print("Accel 合理性检查")
    print("=" * 60)

    T, J = 30, 21
    gt = torch.randn(T, J, 3)

    accel = compute_error_accel(joints_gt=gt, joints_pred=gt)
    print(f"  完全相同输入: Accel = {accel.mean():.8f} (应接近0)")
    assert accel.mean() < 1e-5, f"Accel for identical inputs should be ~0, got {accel.mean()}"

    t_vals = torch.linspace(0, 2 * np.pi, T).unsqueeze(1).unsqueeze(1)
    smooth_motion = torch.sin(t_vals) * torch.ones(T, J, 3)
    jitter_motion = smooth_motion.clone()
    jitter_motion[1:-1] += torch.randn(T - 2, J, 3) * 0.1
    accel_smooth = compute_error_accel(joints_gt=smooth_motion, joints_pred=smooth_motion)
    accel_jitter = compute_error_accel(joints_gt=smooth_motion, joints_pred=jitter_motion)
    print(f"  平滑运动自身: Accel = {accel_smooth.mean():.6f}")
    print(f"  平滑 vs 抖动: Accel = {accel_jitter.mean():.6f} (抖动应更大)")
    assert accel_jitter.mean() > accel_smooth.mean(), "Accel for jittery motion should be larger"

    print("  ✓ Accel 合理性检查通过\n")


def test_all_sanity():
    print("\n" + "=" * 60)
    print("HaWoR 评价指标合理性检查")
    print("=" * 60 + "\n")

    torch.manual_seed(42)
    np.random.seed(42)

    all_passed = True
    try:
        test_pa_mpjpe_sanity()
        test_w_mpjpe_sanity()
        test_wa_mpjpe_sanity()
        test_rte_sanity()
        test_accel_sanity()
    except AssertionError as e:
        print(f"  ✗ 测试失败: {e}")
        all_passed = False

    if all_passed:
        print("=" * 60)
        print("所有合理性检查通过! ✓")
        print("=" * 60)
    else:
        print("=" * 60)
        print("部分检查未通过, 请检查实现")
        print("=" * 60)


if __name__ == "__main__":
    test_all_sanity()
