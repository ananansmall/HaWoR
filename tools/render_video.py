import os
import sys
import cv2
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))

from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from lib.eval_utils.custom_utils import load_slam_cam

def render_offscreen(video_path, output_dir):
    """Render hand reconstruction video using offscreen rendering"""
    
    # Setup args
    class Args:
        def __init__(self):
            self.video_path = video_path
            self.input_type = 'file'
            self.checkpoint = './weights/hawor/checkpoints/hawor.ckpt'
            self.infiller_weight = './weights/hawor/checkpoints/infiller.pt'
            self.img_focal = None
    
    args = Args()
    
    print("="*80)
    print("HaWoR Offscreen Video Renderer")
    print("="*80)
    
    # Run reconstruction pipeline
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    print(f"✓ Detection completed: frames {start_idx} to {end_idx}")
    
    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
    print(f"✓ Motion estimation completed")
    
    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args, start_idx, end_idx)
    
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)
    print(f"✓ SLAM loaded")
    
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(args, start_idx, end_idx, frame_chunks_all)
    print(f"✓ Infiller completed")
    
    # Generate meshes
    hand2idx = {"right": 1, "left": 0}
    vis_start, vis_end = 0, pred_trans.shape[1] - 1
    
    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
                          [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
                          [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
                          [120, 108, 78], [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]
    
    # Get hand vertices
    hand_idx = hand2idx['right']
    pred_glob_r = run_mano(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], 
                           pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], 
                           pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], 
                           betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    right_verts = pred_glob_r['vertices'][0].cpu().numpy()
    
    hand_idx = hand2idx['left']
    pred_glob_l = run_mano_left(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], 
                                pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], 
                                pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], 
                                betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    left_verts = pred_glob_l['vertices'][0].cpu().numpy()
    
    # Coordinate transform (hands are already in world space after R_x)
    R_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    right_verts = np.einsum('ij,tnj->tni', R_x, right_verts)
    left_verts = np.einsum('ij,tnj->tni', R_x, left_verts)
    R_c2w = np.einsum('ij,njk->nik', R_x, R_c2w_sla_all.cpu().numpy())
    t_c2w = np.einsum('ij,nj->ni', R_x, t_c2w_sla_all.cpu().numpy())
    
    print(f"✓ Generated {len(right_verts)} frames of hand meshes")
    print(f"  Right hand shape: {right_verts.shape}")
    print(f"  Left hand shape: {left_verts.shape}")
    print(f"  Camera trajectory: {R_c2w.shape}")
    
    # Simple visualization: overlay hand keypoints on original frames
    img0 = cv2.imread(imgfiles[0])
    height, width = img0.shape[:2]
    
    # Create video writer
    os.makedirs(output_dir, exist_ok=True)
    video_path_out = os.path.join(output_dir, "hawor_reconstruction.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path_out, fourcc, 10, (width, height))
    
    print(f"Creating visualization video with {len(right_verts)} frames...")
    
    for i in range(len(right_verts)):
        # Read original frame
        if i < len(imgfiles):
            frame = cv2.imread(imgfiles[i])
            if frame is None:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Camera intrinsics
        K = np.array([
            [img_focal, 0, width / 2],
            [0, img_focal, height / 2],
            [0, 0, 1]
        ])
        
        # Get camera pose (camera to world)
        R_c2w_i = R_c2w[i]
        t_c2w_i = t_c2w[i]
        
        # Convert camera pose to world to camera
        R_w2c = R_c2w_i.T
        t_w2c = -R_w2c @ t_c2w_i
        
        # Transform hand vertices from world to camera space
        right_verts_cam = (R_w2c @ right_verts[i].T).T + t_w2c
        left_verts_cam = (R_w2c @ left_verts[i].T).T + t_w2c
        
        # Project 3D points to 2D using perspective projection
        def project_to_2d(vertices_3d):
            # Perspective projection: x' = f * X / Z, y' = f * Y / Z
            pts_2d = np.zeros((len(vertices_3d), 2))
            for j in range(len(vertices_3d)):
                X, Y, Z = vertices_3d[j]
                if Z > 0.1:  # Only project points in front of camera
                    u = K[0, 0] * X / Z + K[0, 2]
                    v = K[1, 1] * Y / Z + K[1, 2]
                    pts_2d[j] = [u, v]
                else:
                    pts_2d[j] = [-1000, -1000]  # Invalid point
            return pts_2d.astype(int)
        
        right_pts = project_to_2d(right_verts_cam)
        left_pts = project_to_2d(left_verts_cam)
        
        # Draw hand vertices as points
        for pt in right_pts[::5]:  # Sample every 5 points
            if 0 <= pt[0] < width and 0 <= pt[1] < height:
                cv2.circle(frame, tuple(pt), 3, (0, 165, 255), -1)  # Orange
        
        for pt in left_pts[::5]:
            if 0 <= pt[0] < width and 0 <= pt[1] < height:
                cv2.circle(frame, tuple(pt), 3, (255, 0, 0), -1)  # Blue
        
        # Add info text
        cv2.putText(frame, f"Frame {i}/{len(right_verts)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Orange: Right Hand", (10, height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(frame, "Blue: Left Hand", (10, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        out.write(frame)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(right_verts)} frames")
    
    out.release()
    print(f"\n{'='*80}")
    print(f"✓ Video saved to: {video_path_out}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default='./example/7.mp4')
    args = parser.parse_args()
    
    output_dir = os.path.join(os.path.dirname(args.video_path), 
                              os.path.splitext(os.path.basename(args.video_path))[0], 
                              "rendered_video")
    
    render_offscreen(args.video_path, output_dir)
