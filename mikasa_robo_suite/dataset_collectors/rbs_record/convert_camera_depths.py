import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import h5py
import numpy as np


def depth_to_camera_points(
    depth: np.ndarray, K: np.ndarray, far_mm: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one depth image to camera-space points.

    Returns:
        pts_cam: (N,3)
        valid_mask: (H*W,) bool mask used during flattening
    """
    H, W = depth.shape
    if far_mm is not None:
        depth = np.where((depth == 0) | (depth == -32768), far_mm, depth)

    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v, indexing="xy")

    z_mm = depth.reshape(-1).astype(np.float32)
    valid = (z_mm > 0) & np.isfinite(z_mm)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), valid

    uu = uu.reshape(-1)[valid].astype(np.float32)
    vv = vv.reshape(-1)[valid].astype(np.float32)
    z_m = z_mm[valid]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x = (uu - cx) * z_m / fx
    y = (cy - vv) * z_m / fy
    z_cam = -z_m
    return np.stack([x, y, z_cam], axis=1), valid


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    q = quat
    if q.ndim == 1:
        q = q[None, ...]
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    rot = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = 1 - 2 * (yy + zz)
    rot[:, 0, 1] = 2 * (xy - wz)
    rot[:, 0, 2] = 2 * (xz + wy)
    rot[:, 1, 0] = 2 * (xy + wz)
    rot[:, 1, 1] = 1 - 2 * (xx + zz)
    rot[:, 1, 2] = 2 * (yz - wx)
    rot[:, 2, 0] = 2 * (xz - wy)
    rot[:, 2, 1] = 2 * (yz + wx)
    rot[:, 2, 2] = 1 - 2 * (xx + yy)
    if rot.shape[0] == 1:
        return rot[0]
    return rot


def find_h5_with_id_poses(folder: Path):
    for p in folder.glob("*.h5"):
        try:
            with h5py.File(p, "r") as f:
                for k in f.keys():
                    g = f[k]
                    if isinstance(g, h5py.Group) and "id_poses" in g.keys():
                        return p, k
        except Exception:
            continue
    return None, None


def _load_tracking_context(folder: Path):
    """Load reusable tracking context once per folder (for speed).

    Notes
    -----
    Original dev_wjj logic hard-coded `seg[0]` and `pose[0]` as the
    reference frame for every anchor. On envs like MIKASA-Robo
    ShellGameTouch-v0 where the target objects are NOT visible at t=0
    (they get dropped into the scene by `_initialize_episode` during the
    first physics steps), that produces anchor files with zero pixels
    assigned to those objects — their motion is never tracked.

    To fix this we keep the full `seg` array in context and let
    `track_anchor_file_exact` pick the reference frame per-anchor
    (default: the anchor index itself). This is a strict superset of
    dev_wjj's behaviour (anchor=0 → identical output) and is what
    actually makes dev_wjj's scheme work on MIKASA-Robo envs.
    """
    seg_candidates = [folder / "seg.npy", folder / "segmentation.npy"]
    seg_path = None
    for c in seg_candidates:
        if c.exists():
            seg_path = c
            break
    if seg_path is None:
        raise FileNotFoundError(f"seg.npy not found in {folder}")

    seg = np.load(seg_path, allow_pickle=False)
    if getattr(seg, "ndim", 0) >= 3 and seg.shape[0] > 1:
        seg_all = seg
        seg0 = seg[0]
    else:
        seg_all = seg[None] if getattr(seg, "ndim", 0) == 2 else seg
        seg0 = seg if seg.ndim == 2 else seg[0]
    seg_flat = seg0.reshape(-1)
    pixel_shape = seg0.shape if seg0.ndim == 2 else None

    h5_path, traj_group = find_h5_with_id_poses(folder)
    if h5_path is None:
        raise FileNotFoundError(f"No .h5 with id_poses found in {folder}")

    sid_data = {}
    T = None
    with h5py.File(h5_path, "r") as f_h5:
        if traj_group is None:
            raise RuntimeError("traj_group is None")
        traj = f_h5[traj_group]
        if not isinstance(traj, h5py.Group):
            raise RuntimeError(f"{traj_group} is not group")
        id_poses = traj["id_poses"]
        if not isinstance(id_poses, h5py.Group):
            raise RuntimeError("id_poses is not group")

        for key in id_poses.keys():
            g = id_poses[key]
            if not isinstance(g, h5py.Group):
                continue

            if "camera_position" in g and "camera_quaternion" in g:
                pos_arr = np.asarray(g["camera_position"])
                quat_arr = np.asarray(g["camera_quaternion"])
            elif "position" in g and "quaternion" in g:
                pos_arr = np.asarray(g["position"])
                quat_arr = np.asarray(g["quaternion"])
            else:
                continue

            if pos_arr.ndim == 3 and pos_arr.shape[1] > 1:
                pos_arr = pos_arr[:, 0, :]
            if quat_arr.ndim == 3 and quat_arr.shape[1] > 1:
                quat_arr = quat_arr[:, 0, :]

            pos_arr = pos_arr.astype(np.float32)
            quat_arr = quat_arr.astype(np.float32)

            if T is None:
                T = int(pos_arr.shape[0])

            sid_data[str(int(key))] = {
                "pos": pos_arr,
                "rot": quat_to_rot_matrix(quat_arr),
            }

    if T is None:
        raise RuntimeError("Cannot determine T from id_poses")

    return {
        "seg_flat": seg_flat,      # kept for backward compat (seg[0] flat)
        "seg_all": seg_all,        # (T,H,W) full segmentation video
        "pixel_shape": pixel_shape,
        "sid_data": sid_data,
        "T": int(T),
        "unique_sids": np.unique(seg_flat),
    }


def track_anchor_file_exact(
    anchor_path: Path,
    out_path: Path,
    context: dict | None = None,
    anchor_array: np.ndarray | None = None,
    anchor_idx: int | None = None,
):
    """Tracking logic ported from dev_wjj's anchor_tracker.py, with a
    small but crucial fix: use **the anchor's own frame** (seg + id_pose)
    as the reference, instead of always using frame 0.

    Parameters
    ----------
    anchor_path : Path
        Path to the `scene_point_flow_ref{XXXXX}.anchor.npy` on disk.
    out_path : Path
        Where to save the (T, H, W, 3) tracked tensor.
    anchor_idx : int | None
        Frame index used as the reference for this anchor. If None the
        index is parsed from `anchor_path.stem` (`ref{XXXXX}`); if it
        still can't be determined, 0 is used to preserve dev_wjj
        behaviour.

    Why this changes
    ----------------
    dev_wjj's original implementation always used `seg[0]` + `pose[0]`
    to (a) decide pixel→sid ownership, and (b) compute the local-frame
    template of each tracked actor. On environments like MIKASA-Robo's
    ShellGameTouch-v0 the target actors (mugs) are not visible at t=0
    (`_initialize_episode` drops them in after a few physics steps), so
    `seg[0]==mug_sid` is empty and the mugs are silently skipped for
    every anchor. Using the anchor's own frame makes the tracker see
    whatever is actually rendered at that anchor.
    """
    folder = anchor_path.parent
    if anchor_array is None:
        anchor = np.load(anchor_path, allow_pickle=False)
    else:
        anchor = anchor_array

    if context is None:
        context = _load_tracking_context(folder)

    # Infer anchor_idx from the filename (scene_point_flow_ref{XXXXX}.anchor.npy)
    if anchor_idx is None:
        import re as _re
        m = _re.search(r"ref(\d+)", anchor_path.stem)
        anchor_idx = int(m.group(1)) if m else 0

    seg_all = context.get("seg_all", None)
    pixel_shape = context["pixel_shape"]
    sid_data = context["sid_data"]
    T = context["T"]

    # Build per-anchor seg_flat + unique_sids (instead of hard-coded seg[0])
    if seg_all is not None and seg_all.ndim >= 3 and seg_all.shape[0] > 1:
        ref_idx = min(int(anchor_idx), int(seg_all.shape[0]) - 1)
        seg_flat = seg_all[ref_idx].reshape(-1)
    else:
        # Fallback: legacy behaviour
        seg_flat = context["seg_flat"]
        ref_idx = 0
    unique_sids = np.unique(seg_flat)

    if anchor.ndim == 3 and anchor.shape[2] == 3 and pixel_shape is not None and anchor.shape[0:2] == pixel_shape:
        H, W = pixel_shape
        pts = anchor.reshape(-1, 3)
        pixel_shape = (H, W)
    elif anchor.ndim == 2 and anchor.shape[1] == 3:
        pts = anchor
        if pixel_shape is not None:
            H, W = pixel_shape
            if pts.shape[0] != seg_flat.shape[0]:
                raise ValueError("Anchor N does not match seg H*W")
            pixel_shape = (H, W)
        else:
            pixel_shape = None
    else:
        raise ValueError(f"Unsupported anchor/seg shapes: {anchor.shape}")

    N = pts.shape[0]
    p_local = np.full((N, 3), np.nan, dtype=np.float32)

    for sid in unique_sids:
        if sid == 0:
            continue
        sid_str = str(int(sid))
        if sid_str not in sid_data:
            continue

        pose = sid_data[sid_str]
        # Use pose at the ANCHOR frame (not pose[0]) as the reference,
        # matching the pixel→sid mapping derived from seg[ref_idx].
        pos_ref_idx = min(ref_idx, int(pose["pos"].shape[0]) - 1)
        pos_ref = pose["pos"][pos_ref_idx]
        R_arr = pose["rot"]
        R_ref = R_arr[pos_ref_idx] if R_arr.ndim == 3 else R_arr

        T0 = np.eye(4, dtype=np.float32)
        T0[:3, :3] = R_ref
        T0[:3, 3] = pos_ref
        T0_inv = np.linalg.inv(T0)

        mask = seg_flat == sid
        if not np.any(mask):
            continue
        pts_sel = pts[mask]
        homo = np.concatenate(
            [pts_sel[:, :3], np.ones((pts_sel.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        local_sel = (T0_inv @ homo.T).T[:, :3]
        p_local[mask] = local_sel

    frames = np.zeros((T, N, 3), dtype=np.float32)
    # Background pixels (sid==0) are not tracked — replicate anchor-frame
    # positions across all t.
    bg_mask = seg_flat == 0
    if np.any(bg_mask):
        frames[:, bg_mask, :] = pts[bg_mask][None, :, :]

    for sid in unique_sids:
        if sid == 0:
            continue
        sid_str = str(int(sid))
        mask = seg_flat == sid
        if sid_str not in sid_data or not np.any(mask):
            continue

        pose = sid_data[sid_str]
        R_ts = pose["rot"]
        t_ts = pose["pos"]
        local_sel = p_local[mask]
        if np.isnan(local_sel).all():
            continue
        local_h = np.concatenate(
            [local_sel, np.ones((local_sel.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        for t in range(T):
            R = R_ts[t] if R_ts.ndim == 3 else R_ts
            tvec = t_ts[t]
            Tt = np.eye(4, dtype=np.float32)
            Tt[:3, :3] = R
            Tt[:3, 3] = tvec
            res = (Tt @ local_h.T).T[:, :3]
            frames[t, mask, :] = res

    if pixel_shape is not None:
        H, W = pixel_shape
        frames = frames.reshape((T, H, W, 3))

    np.save(out_path, frames.astype(np.float32))
    print(f"Tracked -> {out_path} shape={frames.shape} ref_frame={ref_idx} "
          f"(sids at ref: {int((np.bincount(seg_flat.astype(np.int64), minlength=1) > 0).sum())})")

def process_folder(folder: Path, out_name_template: str = "scene_point_flow_ref{idx:05d}.anchor.npy"):
    depth_path = folder / "depth_video.npy"
    if not depth_path.exists():
        return 0
    depth = np.load(depth_path, allow_pickle=False)
    # depth can be (T,H,W) or (H,W)
    if depth.ndim == 3:
        T, H, W = depth.shape
    elif depth.ndim == 2:
        T = 1
        depth = depth[None, ...]
    else:
        raise ValueError(f"Unsupported depth shape: {depth.shape}")

    intr_path = folder / "cam_intrinsics.npy"
    if intr_path.exists():
        K_all = np.load(intr_path, allow_pickle=False)
        if K_all.ndim == 3:
            # per-frame intrinsics
            def get_K(i):
                return K_all[i]
        elif K_all.ndim == 2:
            def get_K(i):
                return K_all
        else:
            raise ValueError(f"Unsupported intrinsics shape: {K_all.shape}")
    else:
        # fallback: try to read a generic file or fail
        raise FileNotFoundError(f"Missing cam_intrinsics.npy in {folder}")

    # choose frames: 0, round(1/4*(T-1)), round(2/4*(T-1)), round(3/4*(T-1)), T-1
    idxs = []
    if T >= 1:
        for i in range(5):
            idx = int(round(i * (T - 1) / 4.0))
            idxs.append(idx)
        # unique and sorted
        idxs = sorted(set(idxs))

    saved = 0
    # 目录级缓存，避免每个 ref 都重复读取 seg/h5/id_poses
    tracking_context = _load_tracking_context(folder)

    for idx in idxs:
        K = get_K(idx)
        frame_depth = depth[idx]
        pts_cam, valid = depth_to_camera_points(frame_depth, K)

        # 保存为 (H, W, 3) 格式，而不是 (N, 3)
        anchor_hw = np.zeros((H, W, 3), dtype=np.float32)
        anchor_flat = anchor_hw.reshape(-1, 3)
        anchor_flat[valid] = pts_cam.astype(np.float32)

        out_path = folder / out_name_template.format(idx=idx)
        np.save(out_path, anchor_hw)
        print(f"Saved {out_path} ({anchor_hw.shape[0]}x{anchor_hw.shape[1]} points)")

        # 代码层面融合：每生成一个 anchor，立刻 tracking 成对应的 scene_point_flow_refXXXXX.npy
        if out_path.name.endswith(".anchor.npy"):
            tracked_out_path = out_path.with_name(out_path.name.replace(".anchor.npy", ".npy"))
        else:
            tracked_out_path = out_path.with_suffix(".npy")
        track_anchor_file_exact(
            out_path,
            tracked_out_path,
            context=tracking_context,
            anchor_array=anchor_hw,
            anchor_idx=int(idx),     # use this anchor's own frame as reference
        )

        saved += 1
    return saved


def _process_folder_worker(folder_path: str):
    folder = Path(folder_path)
    try:
        n = process_folder(folder)
        return folder_path, n, None
    except Exception as e:
        return folder_path, 0, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="camera_data", help="camera_data 根目录，包含若干子文件夹")
    parser.add_argument("--workers", type=int, default=1, help="并行处理子目录的进程数，默认1")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    # 支持直接传入单个traj目录（包含 depth_video.npy 或 depth.npy）
    if (root / "depth_video.npy").exists() or (root / "depth.npy").exists():
        n = process_folder(root)
        print(f"Done. Saved {n} files.")
        return

    folders = [str(p) for p in sorted(root.iterdir()) if p.is_dir()]
    total = 0

    if args.workers <= 1:
        for fp in folders:
            p = Path(fp)
            try:
                n = process_folder(p)
                if n > 0:
                    total += n
            except Exception as e:
                print(f"Skipped {p}: {e}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_process_folder_worker, fp) for fp in folders]
            for fut in as_completed(futures):
                folder_path, n, err = fut.result()
                if err is not None:
                    print(f"Skipped {folder_path}: {err}")
                else:
                    total += n
    print(f"Done. Saved {total} files.")


if __name__ == "__main__":
    main()
