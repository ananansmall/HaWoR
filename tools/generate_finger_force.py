"""
EgoTouch 真实压力 -> 逐 MANO 顶点压力 -> 5 指受力聚合（含图与数据落盘到 output/）
==============================================================================

本质: 这是【纯数据后处理脚本】(numpy + matplotlib + json), **不是大模型**。
  无需任何神经网络推理, 无需 GPU, hawor 环境即可运行。
  只做两件事:
    1) 用官方映射表把 256 路传感器压力投到 MANO 的 778 个顶点
    2) 把顶点按"最近指尖"聚合, 得到 5 个手指各自的受力

输出到 <out_dir>(默认 HaWoR/output/tactile_sensor_demo/):
  - finger_force.npz:
      finger_force_<hand>  : (T,5) 每帧 5 指受力 [thumb,index,middle,ring,pinky]
      vertex_pressure_<hand> : (T,778) 逐顶点压力
      mano_verts_<hand>    : (T,778,3)
  - finger_force_curve_<hand>.png : 5 指受力随帧变化的曲线图
  - viz_<hand>_f<idx>.png         : 指定帧顶点压力着色

注意: 数值为 EgoTouch 传感器的【相对压力强度】(0~454), 并非牛顿;
      当前仅做时序/手指分布分析, 不直接等于物理力。

用法:
  python tools/generate_finger_force.py \
      --pressure_json <EgoTouch/jq_pressure.json> \
      --mano_npz <HaWoR/output/*/reconstruction/mano_ras_reconstruction_*.npz> \
      [--hand right] [--out_dir output/tactile_sensor_demo] [--viz_frames 0,77]
"""

import argparse
import json
import os

import numpy as np


# MANO 21 joints 的手指划分（指尖 joint id）
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


def load_pressure(json_path):
    frames = []
    with open(json_path) as f:
        for line in f:
            if line.strip():
                frames.append(json.loads(line))
    return frames


def load_mano(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    return {
        "left": (z["joints_ras_world_left"] if "joints_ras_world_left" in z.files else None,
                 z["verts_ras_world_left"] if "verts_ras_world_left" in z.files else None,
                 z["valid_left"] if "valid_left" in z.files else None),
        "right": (z["joints_ras_world_right"] if "joints_ras_world_right" in z.files else None,
                  z["verts_ras_world_right"] if "verts_ras_world_right" in z.files else None,
                  z["valid_right"] if "valid_right" in z.files else None),
    }


def load_mappings(assets_dir):
    out = {}
    for hand in ["left", "right"]:
        ta = json.load(open(os.path.join(
            assets_dir, f"ta_to_mano_mapping_{hand}_visual.json")))["positions"]
        pm = json.load(open(os.path.join(
            assets_dir, f"pressure_position_mapping_{hand}.json")))
        out[hand] = {"ta": ta, "pm": pm}
    return out


def map_frame_pressure(sensor, ta, pm, n_verts=778):
    """单帧 256 传感器压力 -> (778,) 顶点压力。"""
    sensor = np.asarray(sensor, float)
    vp = np.zeros(n_verts)
    for grid, info in ta.items():
        sidx = pm.get(grid)
        if sidx is None or sidx >= len(sensor):
            continue
        val = sensor[sidx]
        for vid in info["mano_vid"]:
            if vid < n_verts:
                vp[vid] = val
    return vp


def aggregate_finger_force(verts_f, joints_f, vp):
    """把顶点压力按最近指尖聚合为 (5,) 五指受力。"""
    tip_idx = [TIP[f] for f in FINGERS]
    tip_pos = joints_f[tip_idx]          # (5,3)
    d = np.linalg.norm(verts_f[:, None, :] - tip_pos[None, :, :], axis=2)  # (778,5)
    ff = np.zeros(5)
    for v in range(778):
        if vp[v] > 0:
            ff[np.argmin(d[v])] += vp[v]
    return ff


def viz_frame(verts, vertex_p, save_path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    vmax = max(float(vertex_p.max()), 1.0)
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
               c=vertex_p, cmap="jet", s=8, vmin=0, vmax=vmax)
    ax.set_title(title or "HaWoR MANO mesh + EgoTouch pressure")
    fig.colorbar(ax.collections[0], label="pressure")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="EgoTouch 压力 -> MANO 顶点 -> 5 指受力")
    ap.add_argument("--pressure_json", required=True)
    ap.add_argument("--mano_npz", required=True)
    ap.add_argument("--hand", default="right", choices=["left", "right", "both"])
    ap.add_argument("--out_dir", default="output/tactile_sensor_demo")
    ap.add_argument("--viz_frames", default="0", help="逗号分隔帧号, 顶点压力着色图")
    ap.add_argument("--assets_dir", default=None)
    args = ap.parse_args()

    assets_dir = args.assets_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "tactile")
    os.makedirs(args.out_dir, exist_ok=True)

    frames = load_pressure(args.pressure_json)
    mano_all = load_mano(args.mano_npz)
    mappings = load_mappings(assets_dir)
    hands = ["left", "right"] if args.hand == "both" else [args.hand]

    result = {}
    for hand in hands:
        joints_f, verts_f, valid_f = mano_all[hand]
        if joints_f is None or verts_f is None:
            print(f"[warn] 无 {hand} 手数据, 跳过")
            continue
        T = min(len(frames), len(verts_f))
        sensor_key = f"sensor_{hand}"
        ta, pm = mappings[hand]["ta"], mappings[hand]["pm"]
        vp_all = np.zeros((T, 778))
        ff_all = np.zeros((T, 5))
        for t in range(T):
            vp_all[t] = map_frame_pressure(frames[t].get(sensor_key), ta, pm)
            vv = valid_f[t] if valid_f is not None else True
            if vv:
                ff_all[t] = aggregate_finger_force(verts_f[t], joints_f[t], vp_all[t])
        result[f"finger_force_{hand}"] = ff_all
        result[f"vertex_pressure_{hand}"] = vp_all
        result[f"mano_verts_{hand}"] = verts_f[:T]
        nz = (vp_all > 0).sum()
        peak_thumb = ff_all[:, 0].max()
        print(f"  {hand}: {T}帧, 非零顶点压力 {nz}, 拇指受力峰值 {peak_thumb:.1f}")
        print(f"  {hand} 各指总受力: "
              + ", ".join(f"{FINGERS[i]}={ff_all[:,i].sum():.0f}" for i in range(5)))

    np.savez(os.path.join(args.out_dir, "finger_force.npz"), **result)
    print(f"data -> {args.out_dir}/finger_force.npz")

    # 曲线图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for hand in hands:
        if f"finger_force_{hand}" not in result:
            continue
        ff = result[f"finger_force_{hand}"]
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, fn in enumerate(FINGERS):
            ax.plot(range(len(ff)), ff[:, i], label=f"{fn}(tip{fn})")
        ax.set_xlabel("frame"); ax.set_ylabel("sensor pressure (rel.)")
        ax.set_title(f"{hand} hand finger force over time")
        ax.legend(); ax.grid(alpha=0.3)
        save = os.path.join(args.out_dir, f"finger_force_curve_{hand}.png")
        plt.tight_layout(); plt.savefig(save, dpi=120); plt.close(fig)
        print(f"curve -> {save}")

    # 顶点压力着色图
    for hand in hands:
        if f"vertex_pressure_{hand}" not in result:
            continue
        vp = result[f"vertex_pressure_{hand}"]
        verts = result[f"mano_verts_{hand}"]
        for fidx in args.viz_frames.split(","):
            fidx = int(fidx)
            if fidx < len(vp):
                save = os.path.join(args.out_dir, f"viz_{hand}_f{fidx}.png")
                viz_frame(verts[fidx], vp[fidx], save, title=f"{hand} hand frame {fidx}")
                print(f"viz -> {save}")


if __name__ == "__main__":
    main()