#!/usr/bin/env python3
"""
HaWoR 重建数据查看器
支持交互式查看手部重建、相机轨迹、重投影效果
"""
import numpy as np
import cv2
import torch
import sys
import os
from pathlib import Path

sys.path.insert(0, '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR')

from hawor.utils.process import run_mano, run_mano_left

class HaWoRViewer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.load_data()
        
    def load_data(self):
        """加载所有重建数据"""
        print("Loading reconstruction data...")
        
        # 1. 加载重建结果
        recon_path = self.data_dir / 'reconstruction' / 'hawor_results_0_113.npz'
        self.recon_data = np.load(recon_path)
        
        # 2. 加载SLAM数据
        slam_path = self.data_dir / 'SLAM' / 'hawor_slam_w_scale_0_113.npz'
        self.slam_data = np.load(slam_path)
        
        # 3. 加载图像列表
        img_dir = self.data_dir / 'extracted_images'
        self.images = sorted([str(f) for f in img_dir.glob('*.jpg')])
        
        # 4. 加载手部掩码（如果有）
        tracks_dir = self.data_dir / 'tracks_0_113'
        if (tracks_dir / 'model_masks.npy').exists():
            self.masks = np.load(tracks_dir / 'model_masks.npy')
            print(f"✓ Loaded {len(self.masks)} hand masks")
        else:
            self.masks = None
            
        # 5. 生成MANO网格
        print("Generating MANO meshes...")
        pred_trans = torch.from_numpy(self.recon_data['pred_trans'])
        pred_rot = torch.from_numpy(self.recon_data['pred_rot'])
        pred_hand_pose = torch.from_numpy(self.recon_data['pred_hand_pose'])
        pred_betas = torch.from_numpy(self.recon_data['pred_betas'])
        
        # 右手 (index 1)
        right_result = run_mano(
            pred_trans[1:2], pred_rot[1:2], 
            pred_hand_pose[1:2], betas=pred_betas[1:2]
        )
        self.right_verts = right_result['vertices'][0].cpu().numpy()
        
        # 左手 (index 0) - 检查是否存在
        if not np.all(pred_trans[0] == 0):
            left_result = run_mano_left(
                pred_trans[0:1], pred_rot[0:1], 
                pred_hand_pose[0:1], betas=pred_betas[0:1]
            )
            self.left_verts = left_result['vertices'][0].cpu().numpy()
            self.has_left_hand = True
        else:
            self.left_verts = None
            self.has_left_hand = False
            
        print(f"✓ Right hand: {self.right_verts.shape}")
        if self.has_left_hand:
            print(f"✓ Left hand: {self.left_verts.shape}")
        else:
            print("⚠️  Left hand not detected")
            
        # 6. 获取相机参数
        self.R_c2w = self.recon_data['R_c2w']
        self.t_c2w = self.recon_data['t_c2w']
        self.focal = float(self.recon_data['img_focal'])
        
        print(f"✓ Focal length: {self.focal}")
        print(f"✓ Frames: {len(self.images)}")
        print("="*80)
    
    def project_to_image(self, vertices_3d, frame_idx):
        """将3D顶点投影到2D图像"""
        # 相机位姿
        R_c2w = self.R_c2w[frame_idx]
        t_c2w = self.t_c2w[frame_idx]
        
        # 世界坐标 -> 相机坐标
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        
        verts_cam = (R_w2c @ vertices_3d.T).T + t_w2c
        
        # 透视投影
        img = cv2.imread(self.images[frame_idx])
        h, w = img.shape[:2]
        
        K = np.array([
            [self.focal, 0, w/2],
            [0, self.focal, h/2],
            [0, 0, 1]
        ])
        
        pts_2d = []
        for v in verts_cam:
            if v[2] > 0.1:  # 在相机前方
                u = int(K[0,0] * v[0] / v[2] + K[0,2])
                v_coord = int(K[1,1] * v[1] / v[2] + K[1,2])
                if 0 <= u < w and 0 <= v_coord < h:
                    pts_2d.append((u, v_coord))
        
        return img, pts_2d
    
    def visualize_frame(self, frame_idx, save=False):
        """可视化单帧"""
        if frame_idx >= len(self.images):
            print(f"Frame {frame_idx} out of range")
            return
        
        # 投影右手
        img, right_pts = self.project_to_image(self.right_verts[frame_idx], frame_idx)
        
        # 绘制右手点
        for pt in right_pts[::5]:  # 每5个点画一个
            cv2.circle(img, pt, 2, (0, 165, 255), -1)  # 橙色
        
        # 投影左手（如果存在）
        if self.has_left_hand:
            _, left_pts = self.project_to_image(self.left_verts[frame_idx], frame_idx)
            for pt in left_pts[::5]:
                cv2.circle(img, pt, 2, (255, 0, 0), -1)  # 蓝色
        
        # 添加信息
        cv2.putText(img, f"Frame {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, "Orange: Right", (10, img.shape[0]-60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        if self.has_left_hand:
            cv2.putText(img, "Blue: Left", (10, img.shape[0]-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if save:
            output_path = self.data_dir / 'visualization' / f'frame_{frame_idx:04d}.jpg'
            output_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(output_path), img)
            print(f"Saved: {output_path}")
        
        return img
    
    def create_video(self, output_path=None, fps=10):
        """创建可视化视频"""
        if output_path is None:
            output_path = self.data_dir / 'visualization' / 'reconstruction_vis.mp4'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        # 读取第一帧获取尺寸
        first_img = cv2.imread(self.images[0])
        h, w = first_img.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        print(f"Creating video with {len(self.images)} frames...")
        for i in range(len(self.images)):
            img = self.visualize_frame(i)
            out.write(img)
            if (i+1) % 10 == 0:
                print(f"  Processed {i+1}/{len(self.images)} frames")
        
        out.release()
        print(f"✓ Video saved to: {output_path}")
        return output_path
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*80)
        print("Reconstruction Statistics")
        print("="*80)
        
        # 手部尺寸
        hand_size = np.max(np.linalg.norm(
            self.right_verts[0] - self.right_verts[0].mean(axis=0), axis=1
        ))
        print(f"Right hand size: {hand_size*100:.1f} cm")
        
        # 深度范围
        depths = self.right_verts[:, :, 2]
        print(f"Depth range: [{depths.min():.2f}m, {depths.max():.2f}m]")
        print(f"Average depth: {depths.mean():.2f}m ± {depths.std():.2f}m")
        
        # 运动幅度
        motion = np.linalg.norm(np.diff(self.right_verts, axis=0), axis=(1,2))
        print(f"Average motion: {motion.mean()*1000:.1f} mm/frame")
        
        # 相机移动
        cam_motion = np.linalg.norm(np.diff(self.t_c2w, axis=0), axis=1)
        print(f"Camera movement: {cam_motion.mean()*100:.1f} cm/frame")
        
        print("="*80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='HaWoR Reconstruction Viewer')
    parser.add_argument('--data_dir', type=str, 
                       default='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/example/7',
                       help='Path to reconstruction data directory')
    parser.add_argument('--frame', type=int, default=None,
                       help='Visualize specific frame (0-indexed)')
    parser.add_argument('--video', action='store_true',
                       help='Create visualization video')
    parser.add_argument('--stats', action='store_true',
                       help='Print reconstruction statistics')
    
    args = parser.parse_args()
    
    viewer = HaWoRViewer(args.data_dir)
    
    if args.stats:
        viewer.print_statistics()
    
    if args.frame is not None:
        img = viewer.visualize_frame(args.frame, save=True)
        print(f"\nVisualization saved to: {args.data_dir}/visualization/frame_{args.frame:04d}.jpg")
    
    if args.video:
        viewer.create_video()
    
    if not args.frame and not args.video and not args.stats:
        # 默认：生成视频 + 打印统计
        viewer.print_statistics()
        viewer.create_video()
        print("\nTip: Use --frame N to visualize specific frame")
        print("     Use --video to create video")
        print("     Use --stats to see statistics")


if __name__ == '__main__':
    main()
