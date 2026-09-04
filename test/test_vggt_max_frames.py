#!/usr/bin/env python3
"""
测试 VGGT-Omega 在不同帧数下的显存占用和是否会 OOM

用法:
  python test_vggt_max_frames.py --video_path /path/to/video.mp4
"""

import argparse
import gc
import os
import sys
import tempfile
import subprocess

import torch
import numpy as np
from PIL import Image as PILImage
from torchvision import transforms as TF
from natsort import natsorted
from glob import glob

HAWOR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
REPLICATE_ROOT = os.path.join(HAWOR_ROOT, '..', 'ReplicateAnyScene')
VGGT_OMEGA_ROOT = os.path.join(HAWOR_ROOT, '..', 'vggt-omega')

sys.path.insert(0, HAWOR_ROOT)
sys.path.insert(0, REPLICATE_ROOT)
sys.path.insert(0, VGGT_OMEGA_ROOT)


def load_vggt_omega_model():
    """加载 VGGT-Omega 模型"""
    from vggt_omega.core.model_builder import build_model_from_cfg
    from vggt_omega.configs import get_config
    from pathlib import Path

    cfg_path = Path(VGGT_OMEGA_ROOT) / 'configs' / 'vggt_omega' / 'vggt_omega.yaml'
    cfg = get_config(str(cfg_path))
    model = build_model_from_cfg(cfg)
    return model


def test_vggt_max_frames(video_path, test_sizes):
    """测试不同帧数下的 VGGT-Omega 显存占用"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"GPU memory (total): {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 加载模型
    print("Loading VGGT-Omega model...")
    model = load_vggt_omega_model().to(device)

    # 从视频中提取帧
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    seq_folder = os.path.join(os.path.dirname(video_path), video_name)
    img_dir = os.path.join(seq_folder, 'extracted_images')
    all_imgfiles = natsorted(glob(os.path.join(img_dir, '*.jpg')))
    total_frames = len(all_imgfiles)
    print(f"Total extracted frames: {total_frames}")

    results = []

    for n_frames in test_sizes:
        if n_frames > total_frames:
            print(f"\n{'='*50}")
            print(f"SKIP: max_frames={n_frames} > total_frames={total_frames}")
            continue

        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()

        print(f"\n{'='*50}")
        print(f"TESTING: max_frames = {n_frames}")

        # 加载帧
        images_list = []
        for i in range(n_frames):
            img_path = all_imgfiles[i]
            img = PILImage.open(img_path).convert('RGB')
            img = img.resize((512, 512), PILImage.BICUBIC)
            img = TF.ToTensor()(img)
            images_list.append(img)
        frames = torch.stack(images_list).to(device)

        mem_before = torch.cuda.memory_allocated() / 1024**3
        print(f"  Memory before inference: {mem_before:.2f} GB")

        # 推理
        try:
            with torch.inference_mode():
                predictions = model(frames)

            mem_after = torch.cuda.memory_allocated() / 1024**3
            print(f"  Memory after inference:  {mem_after:.2f} GB")
            print(f"  Peak memory:             {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
            print(f"  ✓ SUCCESS")

            # 释放
            del predictions, frames
            torch.cuda.empty_cache()
            gc.collect()

            results.append({
                'n_frames': n_frames,
                'success': True,
                'peak_gb': torch.cuda.max_memory_allocated() / 1024**3,
            })

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
                print(f"  ✗ OOM - CUDA out of memory")
                mem_oom = torch.cuda.max_memory_allocated() / 1024**3
                results.append({
                    'n_frames': n_frames,
                    'success': False,
                    'peak_gb': mem_oom,
                })
            else:
                raise

            del frames
            torch.cuda.empty_cache()
            gc.collect()

    # 总结
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "✓" if r['success'] else "✗ OOM"
        print(f"  max_frames={r['n_frames']:3d}  →  {status}  peak={r['peak_gb']:.2f} GB")

    # 找最大安全帧数
    safe = [r for r in results if r['success']]
    if safe:
        max_safe = max(safe, key=lambda x: x['n_frames'])
        print(f"\n  ➡️  Maximum safe max_frames: {max_safe['n_frames']} (peak: {max_safe['peak_gb']:.2f} GB)")
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        overhead = gpu_total - max_safe['peak_gb']
        print(f"     Available overhead:       {overhead:.2f} GB")
    else:
        print("\n  ⚠️  All test sizes failed (OOM)")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str,
                        default='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/assets/example/hoi4d.mp4')
    args = parser.parse_args()

    # 测试从 20 到 40，步长 5
    test_sizes = [20, 25, 30, 35, 40]
    test_vggt_max_frames(args.video_path, test_sizes)