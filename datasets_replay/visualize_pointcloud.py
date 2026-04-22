"""
Point Cloud Visualizer for MIKASA-Robo Dataset
================================================
Reprojects the base_camera depth image from a saved .npz episode into a 3D
world-space point cloud using the stored camera intrinsic / extrinsic matrices,
colours each point by its segmentation ID (or by RGB), and displays the result.

Coordinate-system pipeline
--------------------------
  1. Pixel (u, v) + depth d  →  OpenCV camera frame  (X right, Y down, Z fwd)
  2. Flip Y and Z            →  OpenGL camera frame   (X right, Y up,   Z bwd)
  3. × cam2world_gl          →  World frame

Usage
-----
    conda activate mikasa-robo
    pip install open3d          # only needed once

    python3 datasets_replay/visualize_pointcloud.py \\
        --npz data/MIKASA-Robo/vis/ShellGameTouch-v0/traj_seed0.npz \\
        --frame 0

    # colour by raw RGB instead of segmentation
    python3 datasets_replay/visualize_pointcloud.py \\
        --npz data/MIKASA-Robo/vis/ShellGameTouch-v0/traj_seed0.npz \\
        --frame 0 --color rgb
"""

import argparse
import colorsys
import numpy as np

# ── optional Open3D (falls back to matplotlib) ────────────────────────────────
try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False
    print("[warn] open3d not found.  Install with:  pip install open3d")
    print("       Falling back to matplotlib 3D scatter plot.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: segmentation ID → pseudo-colour RGB in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def seg_id_to_color(seg_ids: np.ndarray) -> np.ndarray:
    """Map a 1-D array of integer seg IDs to (N, 3) float32 RGB in [0, 1].

    ID == 0 (background / table) → dark grey.
    All other IDs get a saturated hue derived from (id * 37) mod 180.
    """
    colors = np.zeros((len(seg_ids), 3), dtype=np.float32)
    for uid in np.unique(seg_ids):
        mask = seg_ids == uid
        if uid == 0:
            colors[mask] = [0.15, 0.15, 0.15]
        else:
            h = ((int(uid) * 37) % 180) / 180.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
            colors[mask] = [r, g, b]
    return colors


# ─────────────────────────────────────────────────────────────────────────────
# Core: depth image → 3D world-space point cloud
# ─────────────────────────────────────────────────────────────────────────────

def depth_to_pointcloud(
    depth:         np.ndarray,   # (H, W)      float32, metres
    cam_intrinsic: np.ndarray,   # (3, 3)      OpenCV K matrix
    cam2world:     np.ndarray,   # (4, 4)      OpenGL cam-to-world
    seg:           np.ndarray | None = None,  # (H, W) int32
    rgb:           np.ndarray | None = None,  # (H, W, 3) uint8
    max_depth:     float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    points : (N, 3) float32  – 3-D world coordinates (metres)
    colors : (N, 3) float32  – RGB in [0, 1]
    """
    H, W = depth.shape
    fx, fy = float(cam_intrinsic[0, 0]), float(cam_intrinsic[1, 1])
    cx, cy = float(cam_intrinsic[0, 2]), float(cam_intrinsic[1, 2])

    # ── pixel grid ────────────────────────────────────────────────────────────
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)          # (H, W)

    d = depth.astype(np.float32)

    # ── validity mask ─────────────────────────────────────────────────────────
    valid = (d > 0.001) & (d < max_depth)
    N = int(valid.sum())
    if N == 0:
        print("[warn] No valid depth pixels found.")
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    # ── step 1: unproject to OpenCV camera frame ──────────────────────────────
    x_cv = (uu[valid] - cx) * d[valid] / fx   # (N,)
    y_cv = (vv[valid] - cy) * d[valid] / fy   # (N,)
    z_cv = d[valid]                             # (N,)

    # ── step 2: OpenCV  →  OpenGL camera convention ───────────────────────────
    #   OpenCV: X right, Y ↓, Z into scene
    #   OpenGL: X right, Y ↑, Z out of screen  →  flip Y and Z
    x_gl = x_cv
    y_gl = -y_cv
    z_gl = -z_cv

    # ── step 3: homogeneous × cam2world ──────────────────────────────────────
    ones   = np.ones(N, dtype=np.float64)
    pts_h  = np.stack([x_gl, y_gl, z_gl, ones], axis=0)          # (4, N)
    pts_w  = (cam2world.astype(np.float64) @ pts_h).T             # (N, 4)
    points = pts_w[:, :3].astype(np.float32)                      # (N, 3)

    # ── colours ───────────────────────────────────────────────────────────────
    if seg is not None:
        colors = seg_id_to_color(seg[valid].ravel())
    elif rgb is not None:
        colors = rgb[valid].astype(np.float32) / 255.0
    else:
        colors = np.full((N, 3), 0.7, dtype=np.float32)

    return points, colors


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation back-ends
# ─────────────────────────────────────────────────────────────────────────────

def visualize_open3d(points: np.ndarray, colors: np.ndarray) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    # World-frame coordinate axes (0.15 m scale)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.15, origin=[0.0, 0.0, 0.0]
    )

    print("\nOpen3D viewer controls:")
    print("  Left-drag  : rotate   |  Scroll   : zoom")
    print("  Mid-drag   : pan      |  Q / Esc  : quit\n")
    o3d.visualization.draw_geometries(
        [pcd, frame],
        window_name="MIKASA-Robo  ·  point cloud coloured by segmentation",
        width=1280, height=720,
    )


def visualize_matplotlib(points: np.ndarray, colors: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    # Downsample to keep matplotlib responsive
    step = max(1, len(points) // 40_000)

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[::step, 0], points[::step, 1], points[::step, 2],
        c=colors[::step], s=0.5, linewidths=0,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Point cloud – coloured by segmentation ID")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# NPZ loading
# ─────────────────────────────────────────────────────────────────────────────

def load_frame(npz_path: str, frame_idx: int):
    ep = np.load(npz_path)

    def _squeeze_batch(arr):
        """Remove any leading size-1 batch dimension."""
        while arr.ndim >= 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr

    rgb_all   = _squeeze_batch(ep["rgb"])    # (T, H, W, 3) uint8
    depth_all = _squeeze_batch(ep["depth"])  # (T, H, W, 1) float32
    seg_all   = _squeeze_batch(ep["seg"])    # (T, H, W, 1) int32

    T = rgb_all.shape[0]
    if frame_idx >= T:
        raise IndexError(f"frame {frame_idx} out of range (episode has {T} steps)")

    rgb   = rgb_all[frame_idx]                # (H, W, 3)
    depth = depth_all[frame_idx, :, :, 0]    # (H, W)
    seg   = seg_all[frame_idx, :, :, 0]      # (H, W)

    cam_intrinsic = ep["cam_intrinsic"].astype(np.float64)   # (3, 3)
    cam2world     = ep["cam2world"].astype(np.float64)       # (4, 4)

    return rgb, depth, seg, cam_intrinsic, cam2world


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Reproject depth + seg from a MIKASA-Robo .npz into a 3D point cloud"
    )
    p.add_argument("--npz",       required=True,
                   help="Path to episode .npz  (must contain depth, seg, "
                        "cam_intrinsic, cam2world)")
    p.add_argument("--frame",     type=int, default=0,
                   help="Frame index to visualise (default: 0)")
    p.add_argument("--max-depth", type=float, default=5.0,
                   help="Clip depth at this value in metres (default: 5.0)")
    p.add_argument("--color",     choices=["seg", "rgb"], default="seg",
                   help="Point colour source – 'seg' (segmentation) or 'rgb'")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading : {args.npz}  frame={args.frame}")
    rgb, depth, seg, cam_intrinsic, cam2world = load_frame(args.npz, args.frame)

    H, W = depth.shape
    unique_ids = np.unique(seg).tolist()
    print(f"  image size    : {W}×{H}")
    print(f"  depth range   : [{depth[depth > 0].min():.3f}, {depth.max():.3f}] m")
    print(f"  seg IDs       : {unique_ids}")
    print(f"  cam_intrinsic :\n{cam_intrinsic}")
    print(f"  cam2world     :\n{cam2world}")

    points, colors = depth_to_pointcloud(
        depth         = depth,
        cam_intrinsic = cam_intrinsic,
        cam2world     = cam2world,
        seg           = seg   if args.color == "seg" else None,
        rgb           = rgb   if args.color == "rgb" else None,
        max_depth     = args.max_depth,
    )
    print(f"\nPoint cloud : {len(points):,} points")

    if HAS_O3D:
        visualize_open3d(points, colors)
    else:
        visualize_matplotlib(points, colors)


if __name__ == "__main__":
    main()
