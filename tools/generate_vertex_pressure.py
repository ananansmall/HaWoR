"""
将 EgoTouch 真实压力映射到 HaWoR 的 MANO 顶点（接触/压力数据生成工具）
=====================================================================

输入:
  - EgoTouch episode 的 jq_pressure.json（逐帧真实 256 传感器压力, JSONL）
  - HaWoR 重建输出的 MANO 网格 npz（verts_ras_world_left/right, (T,778,3)）

输出:
  - out_dir/vertex_pressure.npz:
      vertex_pressure_left/right : (T, 778) 逐 MANO 顶点压力
      mano_verts_left/right      : (T, 778, 3) 对应的世界系顶点（便于可视化）
      ts, frame_index            : 压力帧元信息
  - 可选: --viz_frames 指定帧的压力着色 png

映射资源（来自 TouchAnything 官方仓库, 已固化在 assets/tactile/）:
  - ta_to_mano_mapping_{left,right}_visual.json : grid(row,col) -> MANO 顶点索引列表
  - pressure_position_mapping_{left,right}.json : grid(row,col) -> 传感器索引

用法:
  python tools/generate_vertex_pressure.py \
      --pressure_json <EgoTouch/jq_pressure.json> \
      --mano_npz <HaWoR/output/*/reconstruction/mano_ras_reconstruction_*.npz> \
      --out_dir <output_dir> --hand both --viz_frames 0,50,100
"""

import argparse
import json
import os

import numpy as np


def load_pressure(json_path):
    """读取 EgoTouch jq_pressure.json（JSONL）, 返回帧列表与 ts/frame_index。"""
    frames = []
    with open(json_path) as f:
        for line in f:
            if line.strip():
                frames.append(json.loads(line))
    return frames


def load_mano_verts(npz_path):
    """读取 HaWoR 输出 npz 中的世界系 MANO 顶点 (T,778,3) 左右手。"""
    z = np.load(npz_path, allow_pickle=True)
    left = z["verts_ras_world_left"] if "verts_ras_world_left" in z.files else None
    right = z["verts_ras_world_right"] if "verts_ras_world_right" in z.files else None
    return left, right


def load_mappings(assets_dir):
    """加载 TouchAnything 官方映射资源。

    Returns:
        mappings: {"left": {"grid->mano_vid": dict, "grid->sensor_idx": dict},
                   "right": {...}}
    """
    out = {}
    for hand in ["left", "right"]:
        ta = json.load(open(os.path.join(
            assets_dir, f"ta_to_mano_mapping_{hand}_visual.json")))["positions"]
        pm = json.load(open(os.path.join(
            assets_dir, f"pressure_position_mapping_{hand}.json")))
        out[hand] = {"ta": ta, "pm": pm}
    return out


def map_frame_pressure(sensor, ta, pm, n_verts=778):
    """把单帧 256 传感器压力映射到 778 个 MANO 顶点。

    Args:
        sensor: (256,) 传感器压力数组
        ta:     grid(row,col) -> {"mano_vid": [顶点索引]}
        pm:     grid(row,col) -> 传感器索引
    Returns:
        (778,) 逐顶点压力
    """
    sensor = np.asarray(sensor, dtype=float)
    vertex_p = np.zeros(n_verts)
    for grid, info in ta.items():
        sidx = pm.get(grid)
        if sidx is None or sidx >= len(sensor):
            continue
        val = sensor[sidx]
        for vid in info["mano_vid"]:
            if vid < n_verts:
                vertex_p[vid] = val
    return vertex_p


def viz_frame(verts, vertex_p, save_path, title=""):
    """把一帧 MANO 顶点按压力着色并保存 png。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    vmax = max(float(vertex_p.max()), 1.0)
    sc = ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
                    c=vertex_p, cmap="jet", s=8, vmin=0, vmax=vmax)
    ax.set_title(title or "HaWoR MANO mesh + EgoTouch real pressure")
    fig.colorbar(sc, label="pressure")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="EgoTouch 真实压力 -> HaWoR MANO 顶点")
    ap.add_argument("--pressure_json", required=True,
                    help="EgoTouch episode 的 jq_pressure.json")
    ap.add_argument("--mano_npz", required=True,
                    help="HaWoR 输出 npz (含 verts_ras_world_left/right)")
    ap.add_argument("--hand", default="both", choices=["left", "right", "both"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--viz_frames", default="", help="逗号分隔的帧号, 渲染压力着色 png")
    ap.add_argument("--assets_dir", default=None,
                    help="映射资源目录 (默认 <repo>/assets/tactile)")
    args = ap.parse_args()

    assets_dir = args.assets_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "tactile")
    os.makedirs(args.out_dir, exist_ok=True)

    frames = load_pressure(args.pressure_json)
    mano_left, mano_right = load_mano_verts(args.mano_npz)
    mappings = load_mappings(assets_dir)

    hands = ["left", "right"] if args.hand == "both" else [args.hand]
    n_frames = len(frames)
    lm = len(mano_left) if mano_left is not None else 0
    rm = len(mano_right) if mano_right is not None else 0
    print(f"[generate_vertex_pressure] pressure frames: {n_frames}, "
          f"mano frames: {lm}/{rm}, hands: {hands}")

    result = {"ts": np.array([f.get("ts") for f in frames]),
              "frame_index": np.array([f.get("frame_index", i) for i, f in enumerate(frames)])}
    for hand in hands:
        mano_verts = mano_left if hand == "left" else mano_right
        if mano_verts is None:
            print(f"  [warn] mano npz 无 {hand} 手数据, 跳过")
            continue
        T = min(n_frames, len(mano_verts))
        if T < n_frames:
            print(f"  [warn] {hand} 手: MANO 帧数({len(mano_verts)}) < 压力帧数({n_frames}), "
                  f"截断到前 {T} 帧")
        sensor_key = f"sensor_{hand}"
        ta = mappings[hand]["ta"]
        pm = mappings[hand]["pm"]
        vp = np.zeros((T, 778))
        for t in range(T):
            vp[t] = map_frame_pressure(frames[t][sensor_key], ta, pm)
        result[f"vertex_pressure_{hand}"] = vp
        result[f"mano_verts_{hand}"] = mano_verts[:T]
        nz = (vp > 0).sum()
        print(f"  {hand}: saved ({T},778), 非零压力元素 {nz}, 峰值 {vp.max():.3f}")

    np.savez(os.path.join(args.out_dir, "vertex_pressure.npz"), **result)
    print(f"[generate_vertex_pressure] done -> {args.out_dir}/vertex_pressure.npz")

    if args.viz_frames:
        for hand in hands:
            mano_verts = mano_left if hand == "left" else mano_right
            if mano_verts is None or f"vertex_pressure_{hand}" not in result:
                continue
            vp = result[f"vertex_pressure_{hand}"]
            for fidx in args.viz_frames.split(","):
                fidx = int(fidx)
                if fidx >= len(vp):
                    continue
                save = os.path.join(args.out_dir, f"viz_{hand}_f{fidx}.png")
                viz_frame(mano_verts[fidx], vp[fidx], save,
                          title=f"{hand} hand frame {fidx}")
                print(f"  viz -> {save}")


if __name__ == "__main__":
    main()
