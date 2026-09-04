"""Verify depth_with_mano correctness for 121_C5_CellPhone_161deg.

Three tests per sampled frame:
  A. green-mesh region depth vs surrounding background depth (doc methodology, video 116)
  B. IoU(green mask, near-depth hand mask)          -> 2D alignment of the EXISTING overlay
  C. IoU(green mask, re-projected model_verts) under:
       - branch C '2dscale': F=mask_focal(600) @1920x1080 then rescale to 688x384
       - direct RAS intrinsic (391 @688x384)
     -> which projection generated the existing overlay, and is it reproducible now
"""
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hawor.utils.process import get_mano_faces

RAS = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/121_C5_CellPhone_161deg_vggt_omega'
HW = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/output/121_C5_CellPhone_161deg'
VIS = os.path.join(RAS, 'depth_with_mano')
DEP = os.path.join(RAS, 'depth')

W0, H0 = 1920.0, 1080.0
W, H = 688, 384
sx, sy = W / W0, H / H0
mask_focal = float(open(os.path.join(HW, 'mask_focal.txt')).read().strip())
K = np.loadtxt(os.path.join(RAS, 'intrinsic.txt'))
kfx, kfy, kcx, kcy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

verts = np.load(os.path.join(HW, 'tracks_0_600', 'model_verts.npy'), mmap_mode='r')
faces = np.asarray(get_mano_faces())
NH = faces.shape[0]

print(f'mask_focal={mask_focal}  RAS K: fx={kfx:.1f} fy={kfy:.1f} cx={kcx:.1f} cy={kcy:.1f}')
print(f'verts {verts.shape}  faces {faces.shape}')


def rasterize(x2d, y2d):
    m = np.zeros((H, W), np.uint8)
    pts = np.stack([x2d, y2d], -1).astype(np.int32)
    ok = np.isfinite(x2d) & np.isfinite(y2d) & (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
    for f in faces:
        if not ok[f].all():
            continue
        cv2.fillPoly(m, [pts[f]], 1)
    return m


def proj_2dscale(v):
    x = mask_focal * v[..., 0] / v[..., 2] + W0 / 2
    y = mask_focal * v[..., 1] / v[..., 2] + H0 / 2
    return x * sx, y * sy


def proj_direct(v):
    x = kfx * v[..., 0] / v[..., 2] + kcx
    y = kfy * v[..., 1] / v[..., 2] + kcy
    return x, y


def green_mask(img):
    # mesh green is pure (0,200,0)-ish; JET colormap's green band has high R or high B
    b, g, r = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    return ((g > 150) & (r < 70) & (b < 70)).astype(np.uint8)


def ring_bg(mask, depth):
    k = np.ones((25, 25), np.uint8)
    ring = cv2.dilate(mask, k) - mask
    return depth[ring > 0]


frames = list(range(0, 100, 10))
ratios = [6.0, 5.94, 5.9, 5.8, 6.1]
rows = []
for fi in frames:
    vp = os.path.join(VIS, f'{fi}.png')
    dp = os.path.join(DEP, f'{fi}.png')
    if not (os.path.exists(vp) and os.path.exists(dp)):
        continue
    img = cv2.imread(vp)
    d16 = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
    dep = d16.astype(np.float32) * 0.001
    gm = green_mask(img)

    # near-depth hand mask: largest connected low-depth component overlapping green
    valid = dep > 0
    thr = np.percentile(dep[valid & (dep < 2.0)], 45) if valid.any() else 0
    near = ((dep < thr) & valid).astype(np.uint8)
    n, lab = cv2.connectedComponents(near)
    best, gi = 0, None
    ys, xs = np.where(gm > 0)
    if len(ys) and n > 1:
        cands = set(lab[gm > 0]) - {0}
        for c in cands:
            sz = int((lab == c).sum())
            if sz > best:
                best, gi = sz, c
    hand = (lab == gi).astype(np.uint8) if gi else np.zeros_like(gm)

    # depth consistency (test A)
    if gm.sum() > 50:
        gd = dep[gm > 0]
        bg = ring_bg(gm, dep)
        bg = bg[bg > 0]
        a_g, a_b = np.median(gd), np.median(bg) if len(bg) else np.nan
        a_ok = len(bg) > 50 and abs(a_g - a_b) < 0.10
    else:
        a_g, a_b, a_ok = np.nan, np.nan, False

    # B: green vs near-depth hand
    inter = (gm & hand).sum()
    union = (gm | hand).sum()
    iou_hand = inter / union if union else np.nan

    # C: reprojection under candidate projections / ratios
    best = {'iou': -1}
    for ratio in ratios:
        hi = int(round(fi * ratio))
        for h in [hi, hi - 2, hi + 2, hi - 4, hi + 4]:
            if h < 0 or h >= verts.shape[1]:
                continue
            v = np.asarray(verts[1, h], np.float32)
            if np.isnan(v).any():
                continue
            for name, fn in [('2dscale_F600', proj_2dscale), ('direct_K391', proj_direct)]:
                x, y = fn(v)
                mm = rasterize(x, y)
                if mm.sum() < 50:
                    continue
                iou = (mm & gm).sum() / max((mm | gm).sum(), 1)
                if iou > best['iou']:
                    best = {'iou': iou, 'proj': name, 'ratio': ratio, 'hi': h,
                            'mask': mm, 'iou_hand': (mm & hand).sum() / max((mm | hand).sum(), 1)}

    rows.append((fi, gm.sum(), a_g, a_b, a_ok, iou_hand,
                 best.get('proj'), best.get('ratio'), best['iou'], best.get('iou_hand', np.nan),
                 best.get('mask')))
    print(f'frame {fi:3d}: green_px={gm.sum():6d} | A: hand_z={a_g:.3f} bg_z={a_b:.3f} ok={a_ok} '
          f'| B: IoU(green,depth-hand)={iou_hand:.3f} '
          f'| C: best={best.get("proj")}@ratio{best.get("ratio")} IoU(green,reproj)={best["iou"]:.3f} '
          f'IoU(reproj,depth-hand)={best.get("iou_hand", np.nan):.3f}')

# save a side-by-side for one frame: original green vs best reprojection mask vs depth-hand
# save zoom overlay for frame 30: original mesh vs best reprojection
fi = 30
r = [x for x in rows if x[0] == fi][0]
img = cv2.imread(os.path.join(VIS, f'{fi}.png'))
b_, g_, r_ = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
gm = ((g_ > 150) & (r_ < 70) & (b_ < 70)).astype(np.uint8)
mm = r[10] if r[10] is not None else np.zeros((int(H), int(W)), np.uint8)
canvas = np.zeros((H, W, 3), np.uint8)
canvas[..., 1] = gm * 120            # existing overlay -> green
canvas[..., 2] = mm * 120            # reprojection -> red (overlap -> yellow)
cv2.imwrite('/tmp/verify_121_overlay.png', canvas)
print('\nsaved /tmp/verify_121_overlay.png  (green=existing, red=reprojection, yellow=overlap)')
ok_A = sum(1 for x in rows if x[4])
ok_B = sum(1 for x in rows if x[5] > 0.6)
ok_C = sum(1 for x in rows if x[8] > 0.6)
print(f'\nSUMMARY: frames={len(rows)}  A(depth-consistent)={ok_A}  B(green~hand IoU>0.6)={ok_B}  C(reproj IoU>0.6)={ok_C}')
