"""
HOT3D-Clips 小切片下载+设置脚本

策略:
  1. 优先使用 /mnt/data_8THDD/lza/dataset/hot3d/ 下已有的数据
  2. 只补充下载缺失的部分（train_aria 训练clip + 元数据）
  3. 使用 wget 处理大文件下载（更稳定）

用法:
  python download_hot3d_clips_slice.py

输出: hot3d_clips_slice/
"""

import json
import os
import sys
import subprocess
import time
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(SCRIPT_DIR, "hot3d_clips_slice")
EXISTING_DATA = "/mnt/data_8THDD/lza/dataset/hot3d"

NUM_TRAIN_CLIPS = 3

EXISTING_DIRS = [
    "test_aria",
    "object_models",
    "object_models_eval",
    "object_ref_aria_dynamic_vis",
    "object_ref_aria_static",
    "object_ref_aria_static_vis",
    "object_ref_quest3_static_vis",
]

HF_BASE = "https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/main"


def find_existing_clips(prefix):
    src_dir = os.path.join(EXISTING_DATA, prefix)
    if not os.path.isdir(src_dir):
        return []
    return sorted([
        f for f in os.listdir(src_dir)
        if f.endswith(".tar") and os.path.isfile(os.path.join(src_dir, f))
    ])


def link_existing_data():
    print("\n=== Linking existing data ===")
    os.makedirs(TARGET_DIR, exist_ok=True)
    for subdir in EXISTING_DIRS:
        src = os.path.join(EXISTING_DATA, subdir)
        dst = os.path.join(TARGET_DIR, subdir)
        if not os.path.isdir(src):
            print(f"  [SKIP] {subdir} (source not found)")
            continue
        if os.path.isdir(dst):
            print(f"  [SKIP] {subdir} (already linked)")
            continue
        shutil.copytree(src, dst, copy_function=os.link, dirs_exist_ok=True)
        size_mb = sum(os.path.getsize(os.path.join(dp, f))
                      for dp, _, fn in os.walk(dst) for f in fn) / 1024 / 1024
        print(f"  [LINK] {subdir} ({size_mb:.0f}MB)")


def download_with_wget(url, local_path, timeout=120):
    """用 wget 下载文件（更稳定，支持断点续传）"""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"  [SKIP] (already exists, {os.path.getsize(local_path)//1024//1024}MB)")
        return True
    cmd = [
        "wget", "-c", "--quiet", "--show-progress",
        "--timeout", str(timeout),
        "--tries", "3",
        "-O", local_path,
        url
    ]
    print(f"  Downloading {os.path.basename(local_path)} ...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
        if result.returncode == 0 and os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            size_mb = os.path.getsize(local_path) / 1024 / 1024
            print(f"    [OK] ({size_mb:.1f}MB)")
            return True
        else:
            print(f"    [FAIL] wget exited with code {result.returncode}")
            if os.path.exists(local_path) and os.path.getsize(local_path) < 1000:
                os.remove(local_path)
            return False
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT]")
        return False


def download_metadata():
    print("\n=== Downloading metadata ===")
    meta_files = ["clip_definitions.json", "clip_splits.json", "README.md"]

    # 先从已有数据复制
    for f in meta_files:
        src = os.path.join(EXISTING_DATA, f)
        dst = os.path.join(TARGET_DIR, f)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  [COPY] {f} from existing data ({os.path.getsize(src)/1024:.0f}KB)")
            continue
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  [SKIP] {f} (already exists)")
            continue
        url = f"{HF_BASE}/{f}"
        download_with_wget(url, dst)


def download_clips(prefix="train_aria", num_clips=3):
    print(f"\n=== Downloading {num_clips} {prefix} clips ===")
    clip_dir = os.path.join(TARGET_DIR, prefix)
    os.makedirs(clip_dir, exist_ok=True)

    existing = find_existing_clips(prefix)
    if existing:
        print(f"  Found {len(existing)} existing clips in {EXISTING_DATA}/{prefix}")
        count = 0
        for clip in existing[:num_clips]:
            src = os.path.join(EXISTING_DATA, prefix, clip)
            dst = os.path.join(clip_dir, clip)
            if not os.path.exists(dst):
                os.link(src, dst)
                print(f"  [LINK] {clip}")
                count += 1
        if count > 0:
            return count

    # 从 Hugging Face 下载
    downloaded = 0
    for offset in range(2000):
        if downloaded >= num_clips:
            break
        clip_name = f"clip-{offset:06d}.tar"
        dst = os.path.join(clip_dir, clip_name)
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            downloaded += 1
            continue
        url = f"{HF_BASE}/{prefix}/{clip_name}"
        success = download_with_wget(url, dst, timeout=300)
        if success:
            downloaded += 1
        else:
            time.sleep(3)

    if downloaded == 0:
        print(f"  WARNING: Could not download any {prefix} clips. Network may be unavailable.")
    return downloaded


def show_summary():
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(TARGET_DIR):
        if not filenames:
            continue
        rel = os.path.relpath(dirpath, TARGET_DIR)
        total = sum(os.path.getsize(os.path.join(dirpath, f)) for f in filenames)
        n_tar = sum(1 for f in filenames if f.endswith(".tar"))
        total_size += total
        parts = []
        if n_tar:
            parts.append(f"{n_tar} clips")
        if total > 0:
            parts.append(f"{total/1024/1024:.0f} MB")
        if parts:
            print(f"  {rel}/: {', '.join(parts)}")
    print(f"\n  Total: {total_size/1024/1024/1024:.2f} GB")
    print(f"  Location: {TARGET_DIR}")

    # 检查缺失项
    missing = []
    if not os.path.isfile(os.path.join(TARGET_DIR, "clip_definitions.json")):
        missing.append("clip_definitions.json (元数据)")
    train_dir = os.path.join(TARGET_DIR, "train_aria")
    train_clips = [f for f in os.listdir(train_dir) if f.endswith(".tar")] if os.path.isdir(train_dir) else []
    if not train_clips:
        missing.append(f"train_aria/ 训练 clip (需要网络下载)")
    if missing:
        print(f"\n  Missing: {', '.join(missing)}")
        print(f"  Rerun the script when network is available to download missing files.")


def main():
    print("=" * 60)
    print("HOT3D-Clips Slice Setup")
    print(f"Target: {TARGET_DIR}")
    print("=" * 60)

    os.makedirs(TARGET_DIR, exist_ok=True)
    link_existing_data()
    download_metadata()
    download_clips("train_aria", NUM_TRAIN_CLIPS)
    show_summary()


if __name__ == "__main__":
    main()