"""
快速测试：把 121 相机空间 MANO 投影到 RAS depth(688x384, fx=391)，
对比两种映射方式，用下采样 mask 统计关节命中率，找出真正有效的链路。
"""
import numpy as np, os, cv2
from scipy.spatial.transform import Rotation

TK = '/mnt/data_8THDD/lza/dataset/HOI4D_RGB/HOI4D_selected_200/121_C5_CellPhone_161deg/tracks_0_600'
RS = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/121_C5_CellPhone_161deg_vggt_omega'

joints = np.load(os.path.join(TK, 'model_joints.npy'))   # (2,600,21,3)
masks  = np.load(os.path.join(TK, 'model_masks.npy'))    # (600,1080,1920) uint8
slam   = np.load(os.path.join(TK, '..', 'SLAM', 'hawor_slam_w_scale_0_600.npz'), allow_pickle=True)
traj   = slam['traj']                                     # (600,7) [tx,ty,tz,qx,qy,qz,qw]  W在最后
K = np.loadtxt(os.path.join(RS, 'intrinsic.txt'))         # 3x3 fx=391...
fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
H, W = 384, 688

# 下采样 mask 到 688x384 作真值
mask_ds = np.array([cv2.resize((masks[i]>0).astype(np.uint8), (W,H), interpolation=cv2.INTER_NEAREST) for i in range(600)])

def quat_xyzw_to_R(qx,qy,qz,qw):
    # HaWoR 约定 WXYZ -> 这里手动构造，严格按 WXYZ
    r = Rotation.from_quat([qx,qy,qz,qw])  # scipy 也是 xyzw 输入
    return r.as_matrix()

# 方式A：直接假设 HaWoR 相机空间 == RAS 相机空间，仅用 RAS intrinsic 投影
# 方式B：worldbridge：HaWoR cam ->(SLAM c2w)-> HaWoR world ->(align)-> VGGT world ->(RAS w2c)-> RAS cam
align = None
ca_path = os.path.join(RS, 'coordinate_alignment.json')
if os.path.exists(ca_path):
    import json
    align = json.load(open(ca_path))
    R_align = np.array(align['R']); t_align = np.array(align['t'])

extr = {}
for i in range(100):
    M = np.loadtxt(os.path.join(RS,'extrinsics', f'{i}.txt'))
    extr[i] = M  # [R|t] 4x4

def project(p_cam):
    x,y,z = p_cam[:,0], p_cam[:,1], p_cam[:,2]
    u = fx*x/z + cx; v = fy*y/z + cy
    return np.stack([u,v],-1), z

def hit_rate(uv, i):
    uv = np.round(uv).astype(int)
    valid = (uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
    if valid.sum()==0: return 0.0
    inside = mask_ds[i][uv[valid,1], uv[valid,0]] > 0
    return inside.mean()

# --- 方式A ---
hitsA = []
for fi in range(100):
    p = joints[1, fi]  # 右手相机空间 (21,3)
    uv, z = project(p)
    hitsA.append(hit_rate(uv, fi))
print(f'[A 直接映射 HaWoR cam->RAS intrinsic] 命中率均值={np.mean(hitsA):.3f} 中值={np.median(hitsA):.3f}')

# --- 方式B：worldbridge ---
hitsB = []
for fi in range(100):
    p = joints[1, fi].astype(float)  # HaWoR cam (21,3)
    # SLAM c2w：traj[fi] = [tx,ty,tz,qx,qy,qz,qw]
    t = traj[fi, :3]
    qx,qy,qz,qw = traj[fi,3],traj[fi,4],traj[fi,5],traj[fi,6]
    Rc2w = Rotation.from_quat([qx,qy,qz,qw]).as_matrix()  # scipy xyzw
    p_world_hawor = (Rc2w @ p.T).T + t                   # c2w
    if align is not None:
        p_vggt = (R_align @ p_world_hawor.T).T + t_align
    else:
        p_vggt = p_world_hawor
    M = extr[fi]; Rw2c = M[:3,:3]; tw2c = M[:3,3]
    p_cam = (Rw2c @ p_vggt.T).T + tw2c
    uv, z = project(p_cam)
    hitsB.append(hit_rate(uv, fi))
print(f'[B worldbridge HaWoR->VGGT->RAS] 命中率均值={np.mean(hitsB):.3f} 中值={np.median(hitsB):.3f}')
