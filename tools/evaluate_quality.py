import numpy as np
import cv2
import torch
import sys
import os

sys.path.insert(0, '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR')

from hawor.utils.process import run_mano, run_mano_left

# Load data
data = np.load('/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/example/7/reconstruction/hawor_results_0_113.npz')

print("="*80)
print("HaWoR Reconstruction Quality Evaluation")
print("="*80)

# Check if left hand exists
left_trans_valid = not np.all(data['pred_trans'][1] == 0)
print(f"\nLeft hand detected: {left_trans_valid}")

if not left_trans_valid:
    print("⚠️  Left hand was NOT detected in the video")
    print("   Only right hand reconstruction is available\n")

# Generate MANO meshes for quality check
print("Generating MANO meshes for evaluation...")

pred_trans = torch.from_numpy(data['pred_trans'])
pred_rot = torch.from_numpy(data['pred_rot'])
pred_hand_pose = torch.from_numpy(data['pred_hand_pose'])
pred_betas = torch.from_numpy(data['pred_betas'])

# Right hand
right_result = run_mano(
    pred_trans[1:2], 
    pred_rot[1:2], 
    pred_hand_pose[1:2], 
    betas=pred_betas[1:2]
)
right_verts = right_result['vertices'][0].cpu().numpy()  # (N_frames, 778, 3)

print(f"✓ Right hand mesh generated: {right_verts.shape}")
print(f"  Vertices per frame: {right_verts.shape[1]}")
print(f"  Total frames: {right_verts.shape[0]}")

# Check hand size (should be ~0.1-0.2 meters for adult hand)
hand_sizes = []
for i in range(min(10, len(right_verts))):
    verts = right_verts[i]
    size = np.max(np.linalg.norm(verts - verts.mean(axis=0), axis=1))
    hand_sizes.append(size)

avg_hand_size = np.mean(hand_sizes)
print(f"\nAverage hand size: {avg_hand_size:.3f} meters")
if 0.08 < avg_hand_size < 0.25:
    print("✓ Hand size is reasonable (8-25 cm)")
else:
    print(f"⚠️  Hand size may be incorrect! Expected 0.08-0.25m")

# Check motion smoothness
print("\nChecking motion smoothness...")
if len(right_verts) > 1:
    velocities = np.diff(right_verts, axis=0)
    avg_velocity = np.mean(np.linalg.norm(velocities, axis=2))
    print(f"Average vertex velocity: {avg_velocity*1000:.2f} mm/frame")
    
    if avg_velocity < 0.01:
        print("✓ Motion is smooth")
    else:
        print("⚠️  Motion may be jittery")

# Check depth distribution
print("\nDepth analysis (world coordinates):")
depths = right_verts[:, :, 2]  # Z coordinate in world space
print(f"Depth range: [{depths.min():.2f}m, {depths.max():.2f}m]")
print(f"Average depth: {depths.mean():.2f}m")
print(f"Depth variance: {depths.std():.2f}m")

if depths.std() > 0.5:
    print("✓ Good depth variation (hand moves in 3D)")
else:
    print("⚠️  Limited depth movement")

# Camera trajectory analysis
print("\nCamera trajectory:")
t_c2w = data['t_c2w']
camera_movement = np.linalg.norm(np.diff(t_c2w, axis=0), axis=1)
print(f"Average camera movement: {camera_movement.mean():.2f}m/frame")
print(f"Total camera travel: {np.sum(camera_movement):.2f}m")

print("\n" + "="*80)
print("Evaluation Summary:")
print("="*80)
print("1. Hand size check: ", "✓ PASS" if 0.08 < avg_hand_size < 0.25 else "⚠️ FAIL")
print("2. Motion smoothness: ", "✓ PASS" if avg_velocity < 0.01 else "⚠️ CHECK")
print("3. Depth variation: ", "✓ PASS" if depths.std() > 0.5 else "⚠️ LIMITED")
print("\nTo verify visual alignment, you need to:")
print("  - Project 3D hand back to 2D images")
print("  - Compare with original hand masks in tracks_0_113/model_masks.npy")
print("="*80)
