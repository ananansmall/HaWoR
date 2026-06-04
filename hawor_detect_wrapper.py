"""
HaWoR 手部检测 + 掩码生成 wrapper 脚本。

在 hawor conda 环境中运行，生成 model_masks.npy 后退出。
由 HaworArmSegmenter 通过子进程调用。

用法（由代码自动调用，无需手动运行）：
    conda run -n hawor python hawor_detect_wrapper.py --video_path <video> [--checkpoint <ckpt>] [--infiller_weight <weight>] [--img_focal <focal>]
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation
from easydict import EasyDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", type=str, default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--img_focal", type=float, default=None)
    args = parser.parse_args()

    hawor_args = EasyDict({
        'video_path': args.video_path,
        'input_type': 'file',
        'checkpoint': args.checkpoint,
        'infiller_weight': args.infiller_weight,
        'img_focal': args.img_focal,
    })

    print(f"[HaWoR Wrapper] Processing: {args.video_path}")

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(hawor_args)
    print(f"[HaWoR Wrapper] Detected frames: {start_idx} to {end_idx} ({len(imgfiles)} total)")

    mask_path = os.path.join(seq_folder, f"tracks_{start_idx}_{end_idx}", "model_masks.npy")
    if os.path.exists(mask_path):
        print(f"[HaWoR Wrapper] Masks already exist: {mask_path}")
    else:
        print(f"[HaWoR Wrapper] Running motion estimation...")
        frame_chunks_all, img_focal = hawor_motion_estimation(
            hawor_args, start_idx, end_idx, seq_folder
        )
        print(f"[HaWoR Wrapper] Motion estimation done, focal={img_focal}")

    if os.path.exists(mask_path):
        masks = np.load(mask_path)
        print(f"[HaWoR Wrapper] SUCCESS: mask_path={mask_path} shape={masks.shape} dtype={masks.dtype}")
    else:
        print(f"[HaWoR Wrapper] ERROR: mask not found at {mask_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
