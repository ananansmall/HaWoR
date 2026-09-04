"""
测试：根据 RAS depth 视图生成 depth 3D 坐标，并把右手 MANO 的 3D 坐标
放到同一个 depth 相机坐标系下，输出 depth 3D 坐标。

思路：
1. depth (uint16) 反投影 -> depth 相机坐标系下的 3D 点云 (688x384x3)，scale 先试 0.001 (mm->m)
2. 右手 MANO 的 3D 坐标 (model_joints/verts) 已经是相机坐标系(focal=600, 1920x1080)
   需要映射/对齐到 depth 相机坐标系。
   由于 RAS color/depth 就是原视频缩小版，mano 2D 像素在 depth 视图里已知(v2脚本)，
   我们直接用 depth 内参反投影 mano 2D + depth图上对应像素的 depth 值，得到 mano 在 depth 相机的 3D。
"""
import numpy as np, os, cv2

RS = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/121_C5_CellPhone_161deg_vggt_omega'
TK = '/mnt/data_8THDD/lza/dataset/HOI4D_RGB/HOI4D_selected_200/121_C5_CellPhone_161deg/tracks_0_600'
OUT = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/121_C5_CellPhone_161deg_vggt_omega'

# ---- RAS depth 内参 (688x384) ----
fx, fy, cx, cy = 391.44, 390.72, 344.0, 192.0
H, W = 384, 688
DEPTH_SCALE = 0.001  # uint16 -> 米 (假设 mm)

# mano 2D 投影 (来自 v2 脚本结论: focal=600 投影到 1920x1080 再 resize 到 688x384)
render_focal = 600.0
W0, H0 = 1920, 1080
cx0, cy0 = W0//2, H0//2
sx, sy = W/W0, H/H0

joints = np.load(os.path.join(TK,'model_joints.npy'))  # (2,600,21,3)
verts  = np.load(os.path.join(TK,'model_verts.npy'))    # (2,600,778,3)

def mano_2d(j3d):
    x,y,z = j3d[:,0], j3d[:,1], j3d[:,2]
    u0 = render_focal*x/z + cx0
    v0 = render_focal*y/z + cy0
    return np.stack([u0*sx, v0*sy], -1)

def depth_to_pointcloud(d):
    """depth (uint16) -> 3D 点云 (H,W,3) 在 depth 相机坐标系"""
    z = d.astype(np.float64) * DEPTH_SCALE
    vs, us = np.mgrid[0:H, 0:W]
    X = (us - cx) * z / fx
    Y = (vs - cy) * z / fy
    return np.stack([X, Y, z], -1)  # (H,W,3)

# 测试几帧
test_frames = [0, 30, 60]
mano3d_list = []
pc_list = []
for fi in test_frames:
    d = cv2.imread(os.path.join(RS,'depth',f'{fi}.png'), cv2.IMREAD_UNCHANGED)
    pc = depth_to_pointcloud(d)  # (H,W,3)
    # mano 右手
    j3 = joints[1, fi].astype(float); v3 = verts[1, fi].astype(float)
    j2 = mano_2d(j3)  # (21,2) 像素在 688x384
    # 取 mano 关节对应像素的 depth 值 -> 反投影得到 mano 在 depth 相机 3D
    uv = np.round(j2).astype(int)
    uv[:,0] = np.clip(uv[:,0],0,W-1); uv[:,1] = np.clip(uv[:,1],0,H-1)
    z_mano = d[uv[:,1], uv[:,0]].astype(np.float64) * DEPTH_SCALE
    X = (uv[:,0]-cx)*z_mano/fx
    Y = (uv[:,1]-cy)*z_mano/fy
    mano3d = np.stack([X,Y,z_mano],-1)  # (21,3) depth 相机坐标系
    mano3d_list.append(mano3d)
    pc_list.append(pc[::8,::8].reshape(-1,3))  # 降采样点云
    print(f'frame {fi}: depth z range [{pc[...,2].min():.3f},{pc[...,2].max():.3f}] m')
    print(f'  mano joints z range [{z_mano.min():.3f},{z_mano.max():.3f}] m')
    print(f'  mano joint0 (wrist) 3D = {mano3d[0]}')

np.savez(os.path.join(OUT,'mano_depth3d_test.npz'),
         mano_joints_3d_depthcam=np.array(mano3d_list),   # (N,21,3)
         depth_pointcloud=np.array(pc_list),               # (N,M,3) 降采样
         test_frames=np.array(test_frames),
         depth_scale=DEPTH_SCALE)
print('saved ->', os.path.join(OUT,'mano_depth3d_test.npz'))
