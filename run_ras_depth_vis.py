"""轻量调用 demov2.generate_ras_depth_vis，复用已有的 hawor_results_*.npz，
不重跑完整管线。等价于"用 demov2 的逻辑"生成 depth_with_mano。"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demov2

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str,
                    default='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/assets/basic_pick_place/7.mp4')
args = parser.parse_args()

# 定位已有 reconstruction npz
video_name = os.path.splitext(os.path.basename(args.video_path))[0]
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', video_name)
# 7 的 output 实际是 '7'（视频名去扩展名），reconstruction 在 7_vggt-omega 下
rec = os.path.join(out_dir, 'reconstruction', 'hawor_results_0_113.npz')
if not os.path.exists(rec):
    # 兼容 7_vggt-omega 命名
    rec = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output',
                       video_name + '_vggt-omega', 'reconstruction', 'hawor_results_0_113.npz')
print('loading', rec)
d = np.load(rec, allow_pickle=True)
pred_trans = d['pred_trans']; pred_rot = d['pred_rot']
pred_hand_pose = d['pred_hand_pose']; pred_betas = d['pred_betas']
pred_valid = d['pred_valid']
img_focal = float(d['img_focal'])
cx, cy = 1920/2.0, 1080/2.0

# detected_hands: 从 cam_space 目录推断（demov2 同一逻辑）
cam_dir = os.path.join(os.path.dirname(rec), '..', 'cam_space')
detected_hands = set()
if os.path.isdir(cam_dir):
    for h in os.listdir(cam_dir):
        if os.path.isdir(os.path.join(cam_dir, h)):
            detected_hands.add(int(h))
print('detected_hands', detected_hands, 'img_focal', img_focal)

# render_focal 用 mask_focal.txt（与 mano mask 一致）
mf = os.path.join(os.path.dirname(os.path.dirname(rec)), 'tracks_0_113', 'mask_focal.txt')
render_focal = img_focal
if os.path.exists(mf):
    render_focal = float(open(mf).read().strip())
print('render_focal', render_focal)

demov2.generate_ras_depth_vis(args, pred_trans, pred_rot, pred_hand_pose, pred_betas,
                              pred_valid, detected_hands, render_focal, cx, cy)
print('DONE')
