import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import torch
import numpy as np
import json

from lib.eval_utils.eval_utils import (
    batch_compute_similarity_transform_torch,
    compute_jpe,
    first_align_joints,
    global_align_joints,
    compute_rte,
    compute_error_accel,
)
from hawor.utils.process import run_mano, run_mano_left
from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis


def load_reconstruction(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data


def compute_metrics_from_reconstruction(npz_path, handedness='right'):
    print("=" * 60)
    print(f"使用 HaWoR 推理结果测试评价指标")
    print(f"文件: {npz_path}")
    print(f"手部: {handedness}")
    print("=" * 60)

    data = load_reconstruction(npz_path)

    pred_trans = torch.tensor(data['pred_trans'])
    pred_rot = torch.tensor(data['pred_rot'])
    pred_hand_pose = torch.tensor(data['pred_hand_pose'])
    pred_betas = torch.tensor(data['pred_betas'])
    pred_valid = torch.tensor(data['pred_valid'])
    R_c2w = torch.tensor(data['R_c2w'])
    t_c2w = torch.tensor(data['t_c2w'])

    hand_idx = 0 if handedness == 'left' else 1
    T = pred_trans.shape[1]

    print(f"  总帧数: {T}")
    print(f"  有效帧数 (左/右): {pred_valid[0].sum()}/{pred_valid[1].sum()}")

    root_orient = pred_rot[hand_idx].unsqueeze(0)
    hand_pose = pred_hand_pose[hand_idx].unsqueeze(0)
    trans = pred_trans[hand_idx].unsqueeze(0)
    betas = pred_betas[hand_idx].unsqueeze(0)

    if handedness == 'right':
        outputs = run_mano(trans, root_orient, hand_pose.reshape(1, T, 45), betas=betas)
    else:
        outputs = run_mano_left(trans, root_orient, hand_pose.reshape(1, T, 45), betas=betas)

    pred_j3d_cam = outputs['joints'][0].detach().cpu()
    pred_trans_cam = trans[0].detach().cpu()

    gt_joints_sim = pred_j3d_cam + torch.randn_like(pred_j3d_cam) * 0.01
    gt_joints_shifted = pred_j3d_cam + torch.randn_like(pred_j3d_cam) * 0.005

    print("\n--- PA-MPJPE ---")
    pred_pa = batch_compute_similarity_transform_torch(pred_j3d_cam, gt_joints_sim)
    pa_mpjpe = compute_jpe(gt_joints_sim, pred_pa)
    print(f"  PA-MPJPE (模拟GT+小噪声): {pa_mpjpe.mean():.6f} m = {pa_mpjpe.mean() * 1000:.3f} mm")

    print("\n--- W-MPJPE ---")
    w_j3d = first_align_joints(gt_joints_sim, pred_j3d_cam)
    w_mpjpe = compute_jpe(gt_joints_sim, w_j3d)
    print(f"  W-MPJPE (模拟GT+小噪声): {w_mpjpe.mean():.6f} m = {w_mpjpe.mean() * 1000:.3f} mm")

    print("\n--- WA-MPJPE ---")
    wa_j3d = global_align_joints(gt_joints_sim, pred_j3d_cam)
    wa_mpjpe = compute_jpe(gt_joints_sim, wa_j3d)
    print(f"  WA-MPJPE (模拟GT+小噪声): {wa_mpjpe.mean():.6f} m = {wa_mpjpe.mean() * 1000:.3f} mm")

    print("\n--- RTE ---")
    gt_trans_sim = pred_trans_cam + torch.randn_like(pred_trans_cam) * 0.005
    rte = compute_rte(gt_trans_sim, pred_trans_cam)
    print(f"  RTE (模拟GT+小噪声): {rte.mean():.6f} (归一化比值)")
    print(f"  RTE (cm): {rte.mean() * 100:.3f} cm")

    print("\n--- Accel ---")
    accel = compute_error_accel(joints_gt=gt_joints_shifted, joints_pred=pred_j3d_cam)
    fps = 30
    accel_scaled = accel * (fps ** 2)
    print(f"  Accel (原始): {accel.mean():.6f}")
    print(f"  Accel (缩放后, fps={fps}): {accel_scaled.mean():.3f}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="使用HaWoR推理结果测试评价指标")
    parser.add_argument("--npz_path", type=str,
                        default="example/7/reconstruction/hawor_results_0_113.npz",
                        help="HaWoR推理结果的npz文件路径")
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"],
                        help="评估哪只手")
    args = parser.parse_args()

    npz_path = os.path.join(os.path.dirname(__file__), '..', args.npz_path)
    compute_metrics_from_reconstruction(npz_path, args.hand)
