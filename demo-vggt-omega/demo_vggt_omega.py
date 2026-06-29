#!/usr/bin/env python3
"""
HaWoR + VGGT-Omega Reconstruction Pipeline
============================================

用 VGGT-Omega 替换 DROID-SLAM 的相机估计,
坐标系与 ReplicateAnyScene mainv2.py 一致 (可选 room alignment).

Usage:
    cd /path/to/HaWoR
    conda activate hawor
    python demo-vggt-omega/demo_vggt_omega.py --video_path /path/to/video.mp4
"""

import argparse
import sys
import os
import gc
import re
import math
import subprocess
import tempfile
import shutil

import torch
import torch.nn.functional as F
import numpy as np

if not hasattr(F, 'scaled_dot_product_attention'):
    def _scaled_dot_product_attention_fallback(query, key, value, attn_mask=None,
                                                dropout_p=0.0, is_causal=False, scale=None):
        """分块计算注意力, 避免一次性分配 N×N 矩阵导致 OOM"""
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
        CHUNK = 2048
        output_chunks = []
        for i in range(0, L, CHUNK):
            end = min(i + CHUNK, L)
            q_chunk = query[..., i:end, :]
            attn_weight = q_chunk @ key.transpose(-2, -1) * scale_factor
            if is_causal and attn_mask is None:
                causal_mask = torch.triu(
                    torch.ones(end - i, S, dtype=torch.bool, device=query.device),
                    diagonal=1 + i,
                )
                attn_weight = attn_weight.masked_fill(causal_mask, float('-inf'))
            if attn_mask is not None:
                mask_chunk = attn_mask[..., i:end, :] if attn_mask.ndim >= 2 else attn_mask
                if mask_chunk.dtype == torch.bool:
                    attn_weight = attn_weight.masked_fill(~mask_chunk, float('-inf'))
                else:
                    attn_weight = attn_weight + mask_chunk
            attn_weight = torch.softmax(attn_weight, dim=-1)
            if dropout_p > 0.0 and not torch.is_inference_mode_enabled():
                attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            output_chunks.append(attn_weight @ value)
        return torch.cat(output_chunks, dim=-2)

    F.scaled_dot_product_attention = _scaled_dot_product_attention_fallback
    print(f"[compat] Patched F.scaled_dot_product_attention (chunked, PyTorch {torch.__version__})")

import cv2
import json
import joblib
from natsort import natsorted
from glob import glob
from tqdm import tqdm
from collections import defaultdict

HAWOR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPLICATE_ROOT = os.path.join(HAWOR_ROOT, '..', 'ReplicateAnyScene')
VGGT_OMEGA_ROOT = os.path.join(HAWOR_ROOT, '..', 'vggt-omega')

sys.path.insert(0, HAWOR_ROOT)
sys.path.insert(0, REPLICATE_ROOT)
sys.path.insert(0, VGGT_OMEGA_ROOT)

from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from hawor.utils.rotation import rotation_matrix_to_angle_axis
from lib.eval_utils.custom_utils import cam2world_convert
from lib.pipeline.tools import parse_chunks, parse_chunks_hand_frame
from lib.eval_utils.filling_utils import filling_postprocess, filling_preprocess
from infiller.lib.model.network import TransformerModel

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

HAS_ROOM_ALIGNMENT = False
try:
    from src.geometry_utils import align_to_room_coordinate_system, align_vggt_predictions
    from src.models import load_sam3_image_model, unload_model
    from src.object_segmentation import segment_wall_and_floor
    HAS_ROOM_ALIGNMENT = True
except ImportError:
    pass

try:
    from demov2 import (
        draw_hand_overlay, draw_depth_map,
        draw_depth_info_panel, draw_camera_trajectory,
        generate_vis_verify, generate_checkerboard,
        generate_camera_markers, render_world_view,
        lookat_matrix,
        MANO_SKELETON, FINGER_TIPS, FINGER_NAMES,
    )
    HAS_DEMOV2_VIS = True
except ImportError:
    HAS_DEMOV2_VIS = False


def project_points(pts3d, focal, cx, cy):
    """将3D点投影到2D图像平面 (针孔模型), 返回像素坐标和有效标记"""
    if pts3d.ndim == 1:
        pts3d = pts3d.reshape(1, 3)
    Z = pts3d[:, 2]
    valid = Z > 0.01
    u = np.where(valid, focal * pts3d[:, 0] / np.maximum(Z, 1e-6) + cx, -1)
    v = np.where(valid, focal * pts3d[:, 1] / np.maximum(Z, 1e-6) + cy, -1)
    return np.stack([u, v], axis=-1), valid

VGGT_OMEGA_CHECKPOINT = "/mnt/data/lza/models/vggt_omega/vggt_omega_1b_512.pt"


def load_vggt_omega_model(checkpoint_path=VGGT_OMEGA_CHECKPOINT):
    """加载 VGGT-Omega 模型权重, 返回 eval 模式的模型实例 (CPU)"""
    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


def _load_frames_from_dir(img_dir, max_frames, image_resolution=512):
    """从图片目录加载帧, 超过 max_frames 时均匀采样, 返回 (tensor, 采样索引数组)"""
    images = os.listdir(img_dir)
    images = [img for img in images if img.endswith(('.jpg', '.png', '.jpeg'))]
    images = sorted(
        images,
        key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else -1,
    )
    total_frames = len(images)
    if total_frames == 0:
        raise ValueError(f"No image files found in directory: {img_dir}")

    if total_frames > max_frames and max_frames > 0:
        indices = np.linspace(0, total_frames - 1, max_frames).astype(int)
    else:
        indices = np.arange(total_frames)

    selected = [os.path.join(img_dir, images[i]) for i in indices]
    frames = load_and_preprocess_images(selected, image_resolution=image_resolution)
    return frames, indices


def load_vggt_omega_frames(video_path, max_frames, image_resolution=512):
    """从视频文件或图片目录提取帧并预处理为 VGGT-Omega 输入格式 (S,3,H,W), 值域[0,1]"""
    if os.path.isdir(video_path):
        return _load_frames_from_dir(video_path, max_frames, image_resolution)
    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vsync", "0",
             os.path.join(temp_dir, "frame_%04d.png")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        return _load_frames_from_dir(temp_dir, max_frames, image_resolution)


def unproject_depth_to_world_points(depth_map, extrinsic, intrinsic):
    """将深度图反投影为世界坐标系3D点云 (向量化, 一次处理所有帧), 数学等价于 VGGT 的 unproject_depth_map_to_point_map"""
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))
    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    camera_points = np.stack([
        (x - cx) / fx * depth,
        (y - cy) / fy * depth,
        depth,
    ], axis=-1)
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def vggt_omega_predict(images, model):
    """VGGT-Omega 推理: 输入帧序列, 输出颜色/深度/外参/内参/世界点/置信度"""
    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions_np[key] = value

    world_points = unproject_depth_to_world_points(
        predictions_np["depth"], predictions_np["extrinsic"], predictions_np["intrinsic"],
    )
    predictions_np["world_points_from_depth"] = world_points

    colors = (predictions_np['images'].transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    depths = predictions_np['depth'].squeeze(-1)
    extrinsics = np.pad(
        predictions_np['extrinsic'],
        ((0, 0), (0, 1), (0, 0)),
        mode='constant', constant_values=0,
    )
    extrinsics[:, 3, 3] = 1
    world_points_out = predictions_np['world_points_from_depth'].copy()
    world_points_conf = predictions_np['depth_conf'].copy()
    intrinsic_out = np.mean(predictions_np['intrinsic'], axis=0)

    torch.cuda.empty_cache()
    gc.collect()

    return {
        "colors": colors,
        "depths": depths,
        "extrinsics": extrinsics,
        "world_points": world_points_out,
        "world_points_conf": world_points_conf,
        "intrinsic": intrinsic_out,
    }


def interpolate_camera_poses(R_c2w_sparse, t_c2w_sparse, sparse_indices, total_frames):
    """将稀疏的相机位姿插值到全部帧: 旋转用 SLERP, 平移用线性插值, 边界帧复制最近值"""
    from scipy.spatial.transform import Rotation, Slerp

    R_np = R_c2w_sparse if isinstance(R_c2w_sparse, np.ndarray) else R_c2w_sparse.numpy()
    t_np = t_c2w_sparse if isinstance(t_c2w_sparse, np.ndarray) else t_c2w_sparse.numpy()
    idx_np = sparse_indices if isinstance(sparse_indices, np.ndarray) else np.array(sparse_indices)

    if len(idx_np) == total_frames:
        return R_np.copy(), t_np.copy()

    if len(idx_np) == 1:
        return np.tile(R_np[0], (total_frames, 1, 1)), np.tile(t_np[0], (total_frames, 1))

    rotations = Rotation.from_matrix(R_np)
    slerp = Slerp(idx_np, rotations)

    all_indices = np.arange(total_frames)
    valid = (all_indices >= idx_np[0]) & (all_indices <= idx_np[-1])

    R_all = np.zeros((total_frames, 3, 3))
    t_all = np.zeros((total_frames, 3))

    if valid.sum() > 0:
        R_all[valid] = slerp(all_indices[valid]).as_matrix()
        for dim in range(3):
            t_all[valid, dim] = np.interp(all_indices[valid], idx_np, t_np[:, dim])

    for i in range(idx_np[0]):
        R_all[i] = R_np[0]
        t_all[i] = t_np[0]
    for i in range(idx_np[-1] + 1, total_frames):
        R_all[i] = R_np[-1]
        t_all[i] = t_np[-1]

    return R_all, t_all


from scipy.spatial.transform import Rotation as scipy_R


def _procrustes_align(R_src, t_src, R_ref, t_ref):
    """用 Procrustes 分析对齐两组相机位姿: src → ref, 返回 (R_align, t_align, scale)"""
    t_src_c = t_src - t_src.mean(axis=0)
    t_ref_c = t_ref - t_ref.mean(axis=0)
    H = t_src_c.T @ t_ref_c
    U, S, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = Vt.T @ U.T
    scale = S.sum() / (t_src_c ** 2).sum()
    t_align = t_ref.mean(axis=0) - scale * R_align @ t_src.mean(axis=0)
    return R_align, t_align, scale


def compute_focal_from_vggt_omega(intrinsic_vggt, vggt_h, vggt_w, orig_h, orig_w):
    """从 VGGT-Omega 内参推算原始分辨率下的焦距: FoV 不变, 根据分辨率缩放 fx/fy"""
    fx_vggt = intrinsic_vggt[0, 0]
    fy_vggt = intrinsic_vggt[1, 1]
    fov_w = 2 * np.arctan2(vggt_w, 2 * fx_vggt)
    fov_h = 2 * np.arctan2(vggt_h, 2 * fy_vggt)
    fx_orig = (orig_w / 2.0) / np.tan(fov_w / 2.0)
    fy_orig = (orig_h / 2.0) / np.tan(fov_h / 2.0)
    return (fx_orig + fy_orig) / 2.0


def _merge_chunk_poses(chunk_results, overlap=5):
    """合并分块推理结果: 用重叠区域首帧的相对位姿变换将后续 chunk 对齐到第一个 chunk 坐标系"""
    if len(chunk_results) == 1:
        return chunk_results[0]['R_c2w'], chunk_results[0]['t_c2w']

    R_global = chunk_results[0]['R_c2w'].copy()
    t_global = chunk_results[0]['t_c2w'].copy()

    for ci in range(1, len(chunk_results)):
        cur = chunk_results[ci]
        prev = chunk_results[ci - 1]
        cur_start = cur['global_start']
        prev_start = prev['global_start']
        prev_end = prev_start + len(prev['t_c2w'])

        overlap_start = max(cur_start, prev_start)
        overlap_end = min(cur_start + len(cur['t_c2w']), prev_end)
        n_overlap = overlap_end - overlap_start

        if n_overlap < 1:
            R_global = np.concatenate([R_global, cur['R_c2w']], axis=0)
            t_global = np.concatenate([t_global, cur['t_c2w']], axis=0)
            continue

        ov_idx_in_prev = overlap_start - prev_start
        ov_idx_in_cur = overlap_start - cur_start

        R_prev_ov = R_global[overlap_start]
        t_prev_ov = t_global[overlap_start]
        R_cur_ov = cur['R_c2w'][ov_idx_in_cur]
        t_cur_ov = cur['t_c2w'][ov_idx_in_cur]

        R_align = R_prev_ov @ R_cur_ov.T
        t_align = t_prev_ov - R_align @ t_cur_ov

        cur_R = cur['R_c2w']
        cur_t = cur['t_c2w']
        aligned_R = np.einsum('ij,njk->nik', R_align, cur_R)
        aligned_t = (R_align @ cur_t.T).T + t_align

        new_end = cur_start + len(cur_t)
        if new_end > len(t_global):
            n_new = new_end - len(t_global)
            R_global = np.concatenate([R_global, aligned_R[-n_new:]], axis=0)
            t_global = np.concatenate([t_global, aligned_t[-n_new:]], axis=0)

        fill_start = max(cur_start, prev_end - overlap)
        fill_end = min(new_end, len(t_global))
        if fill_end > fill_start:
            alpha = np.linspace(0, 1, fill_end - fill_start)[:, None]
            t_global[fill_start:fill_end] = (
                (1 - alpha) * t_global[fill_start:fill_end]
                + alpha * aligned_t[fill_start - cur_start:fill_end - cur_start]
            )
            for fi in range(fill_start, fill_end):
                a = (fi - fill_start) / max(1, fill_end - fill_start - 1)
                rv_prev = scipy_R.from_matrix(R_global[fi]).as_rotvec()
                rv_cur = scipy_R.from_matrix(aligned_R[fi - cur_start]).as_rotvec()
                rv_blend = (1 - a) * rv_prev + a * rv_cur
                R_global[fi] = scipy_R.from_rotvec(rv_blend).as_matrix()

    return R_global, t_global


def run_vggt_omega_camera(seq_folder, start_idx, end_idx, max_frames=20,
                          chunk_overlap=5, enable_room_alignment=False):
    """
    VGGT-Omega 分块相机估计: 视频按 max_frames 分块推理, 重叠区域 Procrustes 对齐合并。

    Args:
        seq_folder: 视频帧目录
        max_frames: 每块最大帧数 (默认20, 47GB GPU 安全值)
        chunk_overlap: 块间重叠帧数 (默认5, 用于 Procrustes 对齐)
        enable_room_alignment: 是否启用 Room Alignment

    Returns:
        R_c2w_all, t_c2w_all, img_focal, img_center, vggt_intrinsic, vggt_extrinsics
    """
    img_dir = os.path.join(seq_folder, 'extracted_images')
    all_imgfiles = natsorted(glob(os.path.join(img_dir, '*.jpg')))
    total_frames = len(all_imgfiles)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  Loading VGGT-Omega model...")
    model = load_vggt_omega_model().to(device)

    chunk_size = max_frames
    step = max(1, chunk_size - chunk_overlap)
    chunk_starts = list(range(0, total_frames, step))
    print(f"  Total {total_frames} frames, chunk_size={chunk_size}, overlap={chunk_overlap}, {len(chunk_starts)} chunks")

    chunk_results = []
    all_intrinsics = []
    vggt_h, vggt_w = None, None

    for ci, chunk_start in enumerate(chunk_starts):
        chunk_end = min(chunk_start + chunk_size, total_frames)
        chunk_imgfiles = all_imgfiles[chunk_start:chunk_end]
        chunk_len = len(chunk_imgfiles)

        print(f"  Chunk {ci+1}/{len(chunk_starts)}: frames {chunk_start}-{chunk_end-1} ({chunk_len} frames)")

        images_list = []
        for img_path in chunk_imgfiles:
            from PIL import Image as PILImage
            from torchvision import transforms as TF
            img = PILImage.open(img_path).convert('RGB')
            img = img.resize((512, 512), PILImage.BICUBIC)
            img = TF.ToTensor()(img)
            images_list.append(img)
        frames = torch.stack(images_list).to(device)

        if vggt_h is None:
            vggt_h, vggt_w = frames.shape[-2], frames.shape[-1]

        with torch.inference_mode():
            predictions = model(frames)

        extrinsic, intrinsic = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:]
        )
        extrinsic = extrinsic.detach().float().cpu().numpy()
        if extrinsic.shape[0] == 1:
            extrinsic = extrinsic[0]
        intrinsic = intrinsic.detach().float().cpu().numpy()
        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic[0]

        R_w2c = extrinsic[:, :3, :3]
        t_w2c = extrinsic[:, :3, 3]
        R_c2w = np.transpose(R_w2c, (0, 2, 1))
        t_c2w = -np.einsum('sij,sj->si', R_c2w, t_w2c)

        chunk_results.append({
            'R_c2w': R_c2w,
            't_c2w': t_c2w,
            'global_start': chunk_start,
        })
        all_intrinsics.append(intrinsic)

        del frames, predictions, extrinsic, intrinsic
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  VGGT-Omega inference done, merging {len(chunk_results)} chunks...")

    R_c2w_all, t_c2w_all = _merge_chunk_poses(chunk_results, overlap=chunk_overlap)
    print(f"  Merged camera poses: {R_c2w_all.shape[0]} frames")

    if enable_room_alignment and HAS_ROOM_ALIGNMENT:
        print(f"  Room alignment not yet supported in chunked mode, skipping")

    img0 = cv2.imread(all_imgfiles[0])
    orig_h, orig_w = img0.shape[:2]

    avg_intrinsic = np.mean([inst.mean(axis=0) for inst in all_intrinsics], axis=0)
    img_focal = compute_focal_from_vggt_omega(avg_intrinsic, vggt_h, vggt_w, orig_h, orig_w)
    img_center = [orig_w / 2.0, orig_h / 2.0]

    vggt_extrinsics = np.pad(
        np.concatenate([cr['R_c2w'] for cr in chunk_results], axis=0)[:1],
        ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0,
    )
    vggt_intrinsic = avg_intrinsic

    print(f"  VGGT-Omega camera: focal={img_focal:.1f}, center={img_center}")
    print(f"  Camera poses: {R_c2w_all.shape[0]} frames (chunked, no interpolation)")

    return R_c2w_all, t_c2w_all, img_focal, img_center, vggt_intrinsic, vggt_extrinsics


def vggt_omega_infiller(args, start_idx, end_idx, frame_chunks_all,
                        R_c2w_all, t_c2w_all, img_focal, img_center):
    """使用 VGGT-Omega 相机位姿进行 cam2world 转换 + Transformer infiller 填充缺失帧"""
    weight_path = args.infiller_weight
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ckpt = torch.load(weight_path, map_location=device)

    pos_dim = 3
    shape_dim = 10
    num_joints = 15
    rot_dim = (num_joints + 1) * 6
    repr_dim = 2 * (pos_dim + shape_dim + rot_dim)
    nhead = 8
    horizon = 120
    filling_model = TransformerModel(
        seq_len=horizon, input_dim=repr_dim, d_model=384,
        nhead=nhead, d_hid=2048, nlayers=8, dropout=0.05,
        out_dim=repr_dim, masked_attention_stage=True,
    )
    filling_model.to(device)
    filling_model.load_state_dict(ckpt['transformer_encoder_state_dict'])
    filling_model.eval()

    file = args.video_path
    video_root = os.path.dirname(file)
    video = os.path.basename(file).split('.')[0]
    seq_folder = os.path.join(video_root, video)
    img_folder = f"{video_root}/{video}/extracted_images"
    imgfiles = np.array(natsorted(glob(f'{img_folder}/*.jpg')))

    idx2hand = ['left', 'right']
    filling_length = 120

    R_c2w_t = torch.from_numpy(R_c2w_all).float()
    t_c2w_t = torch.from_numpy(t_c2w_all).float()

    pred_trans = torch.zeros(2, len(imgfiles), 3)
    pred_rot = torch.zeros(2, len(imgfiles), 3)
    pred_hand_pose = torch.zeros(2, len(imgfiles), 45)
    pred_betas = torch.zeros(2, len(imgfiles), 10)
    pred_valid = torch.zeros((2, pred_betas.size(1)))

    tid = [0, 1]
    for k, idx in enumerate(tid):
        frame_chunks = frame_chunks_all[idx]
        if len(frame_chunks) == 0:
            continue
        for frame_ck in frame_chunks:
            print(f"  cam2world: hand {idx}, frame {frame_ck[0]} to {frame_ck[-1]}")
            pred_path = os.path.join(
                seq_folder, 'cam_space', str(idx),
                f"{frame_ck[0]}_{frame_ck[-1]}.json"
            )
            with open(pred_path, "r") as f:
                pred_dict = json.load(f)
            data_out = {k: torch.tensor(v) for k, v in pred_dict.items()}

            R_c2w_ck = R_c2w_t[frame_ck]
            t_c2w_ck = t_c2w_t[frame_ck]
            data_world = cam2world_convert(
                R_c2w_ck, t_c2w_ck, data_out,
                'right' if idx > 0 else 'left'
            )
            pred_trans[[idx], frame_ck] = data_world["init_trans"]
            pred_rot[[idx], frame_ck] = data_world["init_root_orient"]
            pred_hand_pose[[idx], frame_ck] = data_world["init_hand_pose"].flatten(-2)
            pred_betas[[idx], frame_ck] = data_world["init_betas"]
            pred_valid[[idx], frame_ck] = 1

    frame_list = torch.tensor(list(range(pred_trans.size(1))))
    pred_valid = (pred_valid > 0).numpy()
    for k, idx in enumerate([1, 0]):
        missing = ~pred_valid[idx]
        frame = frame_list[missing]
        frame_chunks = parse_chunks_hand_frame(frame)
        print(f"  Infilling {idx2hand[idx]} hand...")
        for frame_ck in tqdm(frame_chunks):
            start_shift = -1
            while (frame_ck[0] + start_shift >= 0 and
                   pred_valid[:, frame_ck[0] + start_shift].sum() != 2):
                start_shift -= 1

            frame_start = frame_ck[0]
            filling_net_start = max(0, frame_start + start_shift)
            filling_net_end = min(len(imgfiles) - 1, filling_net_start + filling_length)
            seq_valid = pred_valid[:, filling_net_start:filling_net_end]
            filling_seq = {
                'trans': pred_trans[:, filling_net_start:filling_net_end].numpy(),
                'rot': pred_rot[:, filling_net_start:filling_net_end].numpy(),
                'hand_pose': pred_hand_pose[:, filling_net_start:filling_net_end].numpy(),
                'betas': pred_betas[:, filling_net_start:filling_net_end].numpy(),
                'valid': seq_valid,
            }
            filling_input, transform_w_canon = filling_preprocess(filling_seq)
            src_mask = torch.zeros((filling_length, filling_length), device=device).type(torch.bool)
            filling_input = torch.from_numpy(filling_input).unsqueeze(0).to(device).permute(1, 0, 2)
            T_original = len(filling_input)
            if T_original < filling_length:
                pad_length = filling_length - T_original
                last_ts = filling_input[-1, :, :]
                padding = last_ts.unsqueeze(0).repeat(pad_length, 1, 1)
                filling_input = torch.cat([filling_input, padding], dim=0)
                seq_valid_padding = np.ones((2, filling_length - T_original))
                seq_valid_padding = np.concatenate([seq_valid, seq_valid_padding], axis=1)
            else:
                seq_valid_padding = seq_valid

            T, B, _ = filling_input.shape
            valid = torch.from_numpy(seq_valid_padding).unsqueeze(0).all(dim=1).permute(1, 0)
            valid_atten = torch.from_numpy(seq_valid_padding).unsqueeze(0).all(dim=1).unsqueeze(1)
            data_mask = torch.zeros((horizon, B, 1), device=device, dtype=filling_input.dtype)
            data_mask[valid] = 1
            atten_mask = torch.ones((B, 1, horizon), device=device, dtype=torch.bool)
            atten_mask[valid_atten] = False
            atten_mask = atten_mask.unsqueeze(2).repeat(1, 1, T, 1)

            output_ck = filling_model(filling_input, src_mask, data_mask, atten_mask)
            output_ck = output_ck.permute(1, 0, 2).reshape(T, 2, -1).cpu().detach()
            output_ck = output_ck[:T_original]
            filling_output = filling_postprocess(output_ck, transform_w_canon)

            filling_seq['trans'][~seq_valid] = filling_output['trans'][~seq_valid]
            filling_seq['rot'][~seq_valid] = filling_output['rot'][~seq_valid]
            filling_seq['hand_pose'][~seq_valid] = filling_output['hand_pose'][~seq_valid]
            filling_seq['betas'][~seq_valid] = filling_output['betas'][~seq_valid]

            pred_trans[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['trans'])
            pred_rot[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['rot'])
            pred_hand_pose[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['hand_pose'])
            pred_betas[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq['betas'])
            pred_valid[:, filling_net_start:filling_net_end] = 1

    save_path = os.path.join(seq_folder, "world_space_res.pth")
    joblib.dump([pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid], save_path)
    return pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HaWoR + VGGT-Omega Reconstruction Pipeline")
    parser.add_argument("--video_path", type=str, default='example/video_0.mp4')
    parser.add_argument("--img_focal", type=float, default=None)
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument("--checkpoint", type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--infiller_weight", type=str, default='./weights/hawor/checkpoints/infiller.pt')
    parser.add_argument("--max_frames", type=int, default=20, help="VGGT-Omega 每块最大帧数 (47GB GPU 建议 20)")
    parser.add_argument("--chunk_overlap", type=int, default=5, help="块间重叠帧数 (用于 Procrustes 对齐合并)")
    parser.add_argument("--enable_room_alignment", action="store_true",
                        help="Enable room alignment (requires ReplicateAnyScene)")
    args = parser.parse_args()

    print("=" * 80)
    print("HaWoR + VGGT-Omega Reconstruction Pipeline")
    print("=" * 80)

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    print(f"✓ Detection & Tracking: frames {start_idx} to {end_idx}")

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    output_base = os.path.join(HAWOR_ROOT, 'output', f'{video_name}_vggt-omega')
    os.makedirs(output_base, exist_ok=True)
    print(f"  Output directory: {output_base}")

    R_c2w_all, t_c2w_all, img_focal, img_center, vggt_intrinsic, vggt_extrinsics = run_vggt_omega_camera(
        seq_folder, start_idx, end_idx,
        max_frames=args.max_frames,
        chunk_overlap=args.chunk_overlap,
        enable_room_alignment=args.enable_room_alignment,
    )
    print(f"✓ VGGT-Omega camera estimation completed")

    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    print(f"  GPU memory after VGGT-Omega cleanup: {gpu_mem:.2f} GB")

    if args.img_focal is not None:
        img_focal = args.img_focal
        print(f"  Using user-specified focal length: {img_focal:.1f}")

    args.img_focal = img_focal
    frame_chunks_all, img_focal_me = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
    print(f"✓ Motion estimation completed (focal={img_focal_me:.1f})")

    if abs(img_focal_me - img_focal) > 1.0:
        print(f"  NOTE: VGGT-Omega focal ({img_focal:.1f}) differs from motion_est ({img_focal_me:.1f})")
        print(f"  Using VGGT-Omega focal for consistency with camera poses")
    img_focal = img_focal

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = vggt_omega_infiller(
        args, start_idx, end_idx, frame_chunks_all,
        R_c2w_all, t_c2w_all, img_focal, img_center,
    )
    print(f"✓ Infiller completed")

    cx, cy = img_center
    hand2idx = {"right": 1, "left": 0}
    vis_start = 0
    vis_end = pred_trans.shape[1]
    num_total_frames = min(
        vis_end - vis_start,
        R_c2w_all.shape[0],
        t_c2w_all.shape[0],
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
    R_c2w_sla_all = torch.from_numpy(R_c2w_all).float()
    t_c2w_sla_all = torch.from_numpy(t_c2w_all).float()
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
    os.makedirs(cam_output_pth, exist_ok=True)
    image_names = imgfiles[vis_start:vis_end]

    img0 = cv2.imread(image_names[0])
    img_h, img_w = img0.shape[:2]
    print(f"  Image size: {img_w}x{img_h}, cx={cx:.1f}, cy={cy:.1f}, focal={img_focal:.1f}")

    print(f"Generating camera-view overlay...")
    left_verts_np = left_dict['vertices'][0].cpu().numpy()
    left_faces = left_dict['faces']
    right_verts_np = right_dict['vertices'][0].cpu().numpy()
    right_faces = right_dict['faces']
    num_frames = min(len(image_names), left_verts_np.shape[0], right_verts_np.shape[0],
                     R_w2c_sla_all.shape[0], pred_valid.shape[1])

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
        cv2.imwrite(os.path.join(cam_output_pth, f'{i:06d}.png'), result)

    print(f"✓ Camera-view overlay saved to: {cam_output_pth}/ ({num_frames} frames)")

    if HAS_DEMOV2_VIS:
        print(f"Generating world view (3D scene)...")
        world_output_pth = os.path.join(output_base, f"vis_world_{vis_start}_{vis_end}")
        render_world_view(left_dict, right_dict, R_c2w_sla_all, t_c2w_sla_all, detected_hands,
                          pred_valid, world_output_pth, img_h, img_w)

        print(f"Generating vis_verify (depth + trajectory + info)...")
        generate_vis_verify(seq_folder, img_focal, start_idx, end_idx,
                            R_c2w_sla_all, t_c2w_sla_all, 1.0,
                            pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid,
                            cx=cx, cy=cy, output_base=output_base)
    else:
        print(f"  WARNING: demov2 visualization functions not available, skipping world view and vis_verify")

    output_dir = os.path.join(output_base, "reconstruction")
    os.makedirs(output_dir, exist_ok=True)

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
             slam_scale=1.0,
             start_idx=start_idx,
             end_idx=end_idx,
             camera_source='vggt_omega')

    vggt_cam_dir = os.path.join(output_base, 'vggt_omega_cam')
    os.makedirs(vggt_cam_dir, exist_ok=True)
    np.savez(os.path.join(vggt_cam_dir, 'vggt_omega_cam.npz'),
             R_c2w=R_c2w_all,
             t_c2w=t_c2w_all,
             img_focal=img_focal,
             img_center=np.array(img_center),
             max_frames=args.max_frames)
    np.savetxt(os.path.join(vggt_cam_dir, 'intrinsic.txt'), vggt_intrinsic)
    for i, extrinsic in enumerate(vggt_extrinsics):
        np.savetxt(os.path.join(vggt_cam_dir, f'extrinsic_{i}.txt'), extrinsic)

    print(f"Copying key data files to output directory...")
    data_items = [
        ('cam_space', os.path.join(seq_folder, 'cam_space')),
        (f'tracks_{start_idx}_{end_idx}', os.path.join(seq_folder, f'tracks_{start_idx}_{end_idx}')),
        ('extracted_images', os.path.join(seq_folder, 'extracted_images')),
        ('world_space_res.pth', os.path.join(seq_folder, 'world_space_res.pth')),
        ('est_focal.txt', os.path.join(seq_folder, 'est_focal.txt')),
    ]
    for name, src in data_items:
        if not os.path.exists(src):
            continue
        dst = os.path.join(output_base, name)
        if os.path.exists(dst):
            continue
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print(f"\n{'=' * 80}")
    print(f"✓ All output saved to: {output_base}")
    print(f"{'=' * 80}")
    print(f"Frames: {start_idx} to {end_idx} ({end_idx - start_idx + 1} frames)")
    print(f"Camera source: VGGT-Omega (replaces DROID-SLAM)")
    print(f"Room alignment: {'enabled' if args.enable_room_alignment else 'disabled'}")
    print(f"VGGT-Omega max frames: {args.max_frames}")
    print(f"Focal length: {img_focal:.1f} (from VGGT-Omega FoV)")
    print(f"Principal point: ({cx:.1f}, {cy:.1f})")
    print(f"Scale: 1.0 (VGGT-Omega metric depth)")
    print(f"Detected hands: {detected_hands}")
    print(f"Data files:")
    print(f"  vggt_omega_cam/     - VGGT-Omega camera data (R_c2w, t_c2w, intrinsic)")
    print(f"  cam_space/          - Camera-space hand parameters")
    print(f"  tracks_{start_idx}_{end_idx}/  - Detection & tracking results")
    print(f"  world_space_res.pth - World-space hand parameters")
    print(f"  reconstruction/     - Combined npz with all data")
    print(f"Visualization:")
    print(f"  vis_cam_{vis_start}_{vis_end}/  - Camera-view overlay ({num_frames} frames)")
    if HAS_DEMOV2_VIS:
        print(f"  vis_world_{vis_start}_{vis_end}/ - 3D world view + world_view.mp4")
        print(f"  vis_verify/        - Verify images + video (per hand)")
    print(f"{'=' * 80}\n")
    print("finish")
