#!/usr/bin/env python3
"""
Batch depth_video Blosc2 compression with int16(mm)+Δ_t (最佳方案 B).

目标：
  - 给定一个 root 目录，递归遍历其子目录；
  - 若子目录中存在 depth_video.npy，则：
      1) 将 depth_video.npy 视作 (T,H,W) float[m]，直接在 (T,H,W) 上按
         int16(mm_step)+Δ_t 最佳方案进行 Blosc2 压缩（不做维度转置）：
             - 定点化: int16_mm，scale = 1000 / mm_step_mm
               （1 个整数单位 ≈ mm_step_mm 毫米，可通过参数配置，默认 5mm）
             - Δ_t: XOR delta 沿时间轴 T (axis=0)
             - chunk: B=(256,256,3,16) 在 depth 情况下退化为 (16,256,256)
             - codec: zstd, clevel=5, filter=bitshuffle
             - verify: 解压验证，保证最大绝对误差 ≤ mm_step_mm
      2) 压缩结果写入 depth_video_int16mm_dt.b2nd。
  - 如需**顺便清理已有文件**（scene_point_video.npy 及之前的压缩产物），可加参数控制。

注意：
  - 这里不再调用 sceneflow/blosc2/compress.py，而是直接内联其核心逻辑。
  - 压缩比统计统一按原始 float32 体积（与 scenepoint 实验一致）。
"""

import argparse
import os
from pathlib import Path
from typing import Optional
import time
import numpy as np
import blosc2
import json
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024
    return f"{x:.2f}B"


def load_cam_poses(path: Path) -> np.ndarray:
    """(N, 4, 4) float32, T_wc (camera to world)."""
    T = np.load(str(path))
    if T.ndim == 3 and T.shape[1:] == (4, 4):
        return np.asarray(T).astype(np.float32)
    if T.ndim == 3 and T.shape[0:2] == (4, 4):
        return np.transpose(T, (2, 0, 1)).astype(np.float32)
    raise ValueError(f"Unsupported cam_poses shape: {T.shape}")


def load_cam_intrinsics(path: Path, num_frames: int) -> np.ndarray:
    """(N, 3, 3) float32. 若单帧 (3,3) 则广播到 N."""
    K = np.load(str(path))
    if K.ndim == 2 and K.shape == (3, 3):
        return np.repeat(K[None, :, :].astype(np.float32), num_frames, axis=0)
    if K.ndim == 3 and K.shape[1:] == (3, 3):
        return np.asarray(K).astype(np.float32)
    raise ValueError(f"Unsupported cam_intrinsics shape: {K.shape}")


def rescale_intrinsics_to_resolution(
    K_raw: np.ndarray, out_w: int, out_h: int
) -> np.ndarray:
    """
    将 DUSt3R 内部分辨率的内参缩放到目标分辨率 (out_w, out_h)。

    约定:
      sx = (out_w / 2) / cx_raw
      sy = (out_h / 2) / cy_raw
      fx' = fx_raw * sx, cx' = cx_raw * sx
      fy' = fy_raw * sy, cy' = cy_raw * sy
    """
    K_raw = np.asarray(K_raw, dtype=np.float64)
    K_rescaled = K_raw.copy()
    cx_raw, cy_raw = K_raw[0, 2], K_raw[1, 2]
    sx = (out_w / 2.0) / cx_raw
    sy = (out_h / 2.0) / cy_raw
    K_rescaled[0, 0] *= sx
    K_rescaled[0, 2] *= sx
    K_rescaled[1, 1] *= sy
    K_rescaled[1, 2] *= sy
    return K_rescaled

def world_z_depth_to_scene_points_frame(
    depth_wz: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
) -> np.ndarray:
    """
    depth_wz: (H, W) world-frame Z（即 scene_point_video[t,:,:,2]），0/nan 无效
    K: (3,3), T_wc: (4,4)
    返回 (H, W, 3) world XYZ，无效处为 (0,0,0)
    """
    H, W = depth_wz.shape
    K64 = np.asarray(K, dtype=np.float64)
    fx, fy = float(K64[0, 0]), float(K64[1, 1])
    cx, cy = float(K64[0, 2]), float(K64[1, 2])
    T = np.asarray(T_wc, dtype=np.float64)

    uu = np.arange(W, dtype=np.float64)
    vv = np.arange(H, dtype=np.float64)
    u, v = np.meshgrid(uu, vv)

    Z_world = np.asarray(depth_wz, dtype=np.float64)
    valid = np.isfinite(Z_world) & (Z_world > 0)

    denom = (
        T[2, 0] * (u - cx) / fx
        + T[2, 1] * (v - cy) / fy
        + T[2, 2]
    )
    small = np.abs(denom) < 1e-8
    denom_safe = denom.copy()
    denom_safe[small] = np.nan

    z_c = np.where(valid, (Z_world - T[2, 3]) / denom_safe, np.nan)
    valid = valid & np.isfinite(z_c) & (z_c > 0)

    x_c = (u - cx) / fx * z_c
    y_c = (v - cy) / fy * z_c
    ones = np.ones_like(x_c)
    P_cam = np.stack([x_c, y_c, z_c, ones], axis=-1)  # (H,W,4)
    P_world = (P_cam @ T.T)[:, :, :3]  # (H,W,3)

    out = np.zeros((H, W, 3), dtype=np.float32)
    out[valid] = P_world[valid].astype(np.float32)
    return out

def find_depth_dirs(root: Path):
    """在 root 下递归查找包含 depth_video.npy 的目录。"""
    for dp, dn, fn in os.walk(root):
        if "depth_video.npy" in fn:
            yield Path(dp)


def compress_depth_dir(
    seg_dir: Path,
    dry_run: bool = False,
    mm_step_mm: float = 5.0,
    delete_existing: bool = False,
    staging_dir: Optional[Path] = None,
    defer_nfs_write: bool = False,
) -> dict:
    """
    对单个 seg 目录执行：
      - 将 depth_video.npy 压缩为 depth_video_int16mm_dt.b2nd

    Args:
        seg_dir: Final NFS output directory (compressed output written here).
        staging_dir: Directory where raw .npy files were written (e.g.
            /dev/shm tmpfs).  Defaults to ``seg_dir`` for backward compat.

    返回统计信息：{raw_mb, comp_mb, ratio, path}
    """
    src_dir = staging_dir if staging_dir is not None else seg_dir
    depth_path = src_dir / "depth_video.npy"
    spv_path = src_dir / "scene_point_video.npy"

    # If staging_dir is set and differs from seg_dir, write compressed
    # output to staging first (tmpfs), then move to seg_dir (NFS).
    _staging_output = (staging_dir is not None and staging_dir != seg_dir)
    _write_dir = staging_dir if _staging_output else seg_dir
    out_b2nd = _write_dir / "depth_video_int16mm_dt.b2nd"

    stats = {
        "seg_dir": str(seg_dir),
        "raw_mb": 0.0,
        "comp_mb": 0.0,
        "ratio": 0.0,
        "skipped": False,
    }

    if not depth_path.exists():
        return stats

    # 统计「压缩前」体积：scene_point_video.npy + depth_video.npy
    depth_file_bytes = os.path.getsize(depth_path)
    spv_bytes = os.path.getsize(spv_path) if spv_path.exists() else 0
    raw_bytes = depth_file_bytes + spv_bytes
    stats["raw_mb"] = raw_bytes / 1e6
    print(
        f"[SEG] {seg_dir.name}: depth+scene_point raw={stats['raw_mb']:.2f} MB "
        f"(depth={human_bytes(depth_file_bytes)}, "
        f"scene_point={'none' if spv_bytes == 0 else human_bytes(spv_bytes)})"
    )

    # 若存在已有产出（scene point / depth video 压缩相关），且用户指定删除，则先删除再压缩
    if delete_existing:
        meta_path = seg_dir / "depth_video_int16mm_dt.meta.json"
        dec_npy_path = seg_dir / "depth_video_int16mm_dt_decompressed.npy"
        to_remove = [
            (spv_path, "scene_point_video.npy"),
            (out_b2nd, "depth_video_int16mm_dt.b2nd"),
            (meta_path, "depth_video_int16mm_dt.meta.json"),
            (dec_npy_path, "depth_video_int16mm_dt_decompressed.npy"),
        ]
        for p, label in to_remove:
            if p.exists() and not dry_run:
                try:
                    p.unlink()
                    print(f"[DEL] {label}: {p}")
                except Exception as e:
                    print(f"[WARN] 删除 {label} 失败: {e}")

    # 2) 加载 depth_video.npy：保持为 (T,H,W)
    import time as _time
    _t_total_start = _time.perf_counter()

    _t0 = _time.perf_counter()
    depth = np.load(str(depth_path), mmap_mode=None)
    _t_load_npy = _time.perf_counter() - _t0

    if depth.ndim != 3:
        raise SystemExit(f"[ERR] depth_video 期望 shape (T,H,W)，当前: {depth.shape} @ {depth_path}")
    T, H, W = depth.shape

    # 保持 (T,H,W) 布局，在 axis=0 上做 Δ_t
    # 原始文件为 fp16，这里 raw 体积口径以 depth.nbytes 为准（而不是内部 float32）
    _t0 = _time.perf_counter()
    arr_thw = depth.astype(np.float32)  # (T,H,W)
    _t_cast_f32 = _time.perf_counter() - _t0

    # depth 自身字节数（用于吞吐量统计，压缩比使用 raw_bytes）
    orig_bytes = int(depth.nbytes)
    orig_mb = orig_bytes / 1e6
    print(f"[SEG] {seg_dir.name}: depth shape={depth.shape}, depth_raw(fp16)={orig_mb:.2f} MB")

    if dry_run:
        return stats

    # 3) int16(mm_step) + Δ_t XOR 编码（沿 T 轴, axis=0）
    #    参照 sceneflow/blosc2/compress.py 的 int16_mm + xor_delta_int16 流程
    arr = arr_thw  # (T,H,W)
    # 定点化: 1 单位 ≈ mm_step_mm 毫米
    if mm_step_mm <= 0:
        raise SystemExit(f"[ERR] mm_step_mm 必须为正数, 当前={mm_step_mm}")
    scale = 1000.0 / mm_step_mm  # depth[m] * scale -> int16
    _t0 = _time.perf_counter()
    arr_f64 = arr.astype(np.float64)
    work_i16 = np.round(arr_f64 * scale).astype(np.int16)  # (T,H,W)
    _t_quantize = _time.perf_counter() - _t0
    print(
        f"[INFO] float32 range: min={float(arr.min()):.6f} max={float(arr.max()):.6f}, "
        f"mm_step={mm_step_mm} (scale={scale:.1f})"
    )
    i16_min, i16_max = int(work_i16.min()), int(work_i16.max())
    overflow = i16_min < -32768 or i16_max > 32767
    print(f"[INFO] int16(mm): scale={scale:.1f}, range=[{i16_min}, {i16_max}], overflow={overflow}")

    # XOR delta 沿时间轴 (axis=0)
    _t0 = _time.perf_counter()
    u = work_i16.view(np.uint16)
    d = np.empty_like(u)
    d[0, ...] = u[0, ...]
    d[1:, ...] = u[1:, ...] ^ u[:-1, ...]
    work_delta = d.view(np.int16)  # (T,H,W) int16 delta
    _t_xor_delta = _time.perf_counter() - _t0

    # 4) Blosc2 压缩：chunk=(16,256,256), codec=zstd:5, filter=bitshuffle
    Tt, Hh, Ww = work_delta.shape
    chunk = (min(16, Tt), min(256, Hh), min(256, Ww))
    print(
        f"[INFO] Using chunk={chunk}, codec=zstd, clevel=5, filter=bitshuffle, "
        f"int16_mm+Δ_t (xor)"
    )

    cparams = blosc2.CParams(
        codec=blosc2.Codec.ZSTD,
        clevel=5,
        filters=[blosc2.Filter.BITSHUFFLE],
    )

    os.makedirs(os.path.dirname(out_b2nd) or ".", exist_ok=True)
    _t0 = _time.perf_counter()
    nd = blosc2.asarray(work_delta, chunks=chunk, cparams=cparams)
    nd.save(out_b2nd)
    _t_blosc2_encode = _time.perf_counter() - _t0

    # 写入元数据，记录定点 scale 和 mm_step
    meta = {
        "scale": scale,
        "mm_step_mm": mm_step_mm,
        "orig_dtype": str(depth.dtype),
        "shape": list(depth.shape),
    }
    meta_path = _write_dir / "depth_video_int16mm_dt.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 统计「压缩后」体积：b2nd + meta.json
    b2nd_bytes = os.path.getsize(out_b2nd)
    meta_bytes = os.path.getsize(meta_path)
    comp_bytes = b2nd_bytes + meta_bytes
    comp_mb = comp_bytes / 1e6
    stats["comp_mb"] = comp_mb
    stats["ratio"] = raw_bytes / comp_bytes if comp_bytes > 0 else 0.0
    enc_mb_s = (orig_bytes / 1e6) / _t_blosc2_encode if _t_blosc2_encode > 0 else float("inf")

    print(f"[RESULT] Compressed file: {out_b2nd}")
    print(
        f"[RESULT] Compressed size: total={human_bytes(comp_bytes)} "
        f"(b2nd={human_bytes(b2nd_bytes)}, meta={human_bytes(meta_bytes)})"
    )
    print(f"[RESULT] Compression ratio ((depth+scene_point)/b2nd+meta): {stats['ratio']:.2f}x")
    print(f"[RESULT] Encode throughput: {enc_mb_s:.2f} MB/s (based on raw_fp16 bytes)")

    # 5) 可选：解压 + 验证（保证量化误差 ≤ mm_step_mm）
    _t0 = _time.perf_counter()
    nd2 = blosc2.open(out_b2nd)
    rec_delta = nd2[:]  # (T,H,W) int16 delta
    _t_blosc2_decode = _time.perf_counter() - _t0
    dec_mb_s = (orig_bytes / 1e6) / _t_blosc2_decode if _t_blosc2_decode > 0 else float("inf")
    print(f"[RESULT] Decode throughput: {dec_mb_s:.2f} MB/s (based on raw_fp16 bytes)")

    # XOR 反解码（沿 axis=0）
    if rec_delta.dtype != np.int16:
        raise SystemExit(f"[FAIL] 解码数据类型错误，期望 int16，得到 {rec_delta.dtype}")
    _t0 = _time.perf_counter()
    u_rec = rec_delta.view(np.uint16)
    u_out = np.empty_like(u_rec)
    u_out[0, ...] = u_rec[0, ...]
    for t in range(1, u_rec.shape[0]):
        u_out[t, ...] = u_rec[t, ...] ^ u_out[t - 1, ...]
    rec_i16 = u_out.view(np.int16)
    _t_xor_inverse = _time.perf_counter() - _t0

    _t0 = _time.perf_counter()
    rec_f = rec_i16.astype(np.float32) / scale
    max_diff = float(np.max(np.abs(rec_f.astype(np.float64) - arr.astype(np.float64))))
    _t_verify = _time.perf_counter() - _t0
    tol_m = mm_step_mm / 1000.0  # 允许的最大误差（约等于一个量化步长）
    if max_diff > tol_m + 1e-6:
        raise SystemExit(
            f"[FAIL] Verification failed. max_abs_diff={max_diff:.6f} m "
            f"(> {tol_m:.6f} m, mm_step={mm_step_mm})"
        )
    print(
        f"[PASS] Verification OK (int16_mm+Δ_t, max_abs_diff={max_diff:.6f} m, "
        f"mm_step={mm_step_mm})."
    )

    _t_total_compress = _time.perf_counter() - _t_total_start

    # Store detailed timing in stats
    stats["timing_breakdown_s"] = {
        "load_npy": round(_t_load_npy, 4),
        "cast_f32": round(_t_cast_f32, 4),
        "quantize_int16": round(_t_quantize, 4),
        "xor_delta_encode": round(_t_xor_delta, 4),
        "blosc2_encode": round(_t_blosc2_encode, 4),
        "blosc2_decode_verify": round(_t_blosc2_decode, 4),
        "xor_inverse_verify": round(_t_xor_inverse, 4),
        "verify_diff": round(_t_verify, 4),
        "total": round(_t_total_compress, 4),
    }
    print(f"[TIMING] depth compress breakdown: {stats['timing_breakdown_s']}")

    # Move compressed output from staging (tmpfs) to final directory (NFS)
    # after verification succeeds.  Skip when defer_nfs_write.
    if _staging_output and not defer_nfs_write:
        import shutil
        final_b2nd = seg_dir / "depth_video_int16mm_dt.b2nd"
        final_meta = seg_dir / "depth_video_int16mm_dt.meta.json"
        shutil.move(str(out_b2nd), str(final_b2nd))
        shutil.move(str(meta_path), str(final_meta))

    # 若用户指定 delete_existing，则在压缩和校验成功后删除原始 depth_video.npy
    if delete_existing and depth_path.exists() and not dry_run:
        try:
            depth_path.unlink()
            print(f"[DEL] depth_video.npy: {depth_path}")
        except Exception as e:
            print(f"[WARN] 删除 depth_video.npy 失败: {e}")

    return stats


def compare_depth_dir(seg_dir: Path):
    """
    对单个 seg 目录进行 depth 压缩对比：
      - 使用原始 depth_video.npy 和 解压后的 depth_video_int16mm_dt_decompressed.npy
        分别反投影恢复点云（第一帧），与 scene_point_video.npy 第一帧做 3D 误差热力图。
    依赖文件：
      - depth_video.npy
      - depth_video_int16mm_dt_decompressed.npy
      - cam_poses.npy
      - cam_intrinsics.npy
      - scene_point_video.npy
    """
    depth_orig_path = seg_dir / "depth_video.npy"
    depth_dec_path = seg_dir / "depth_video_int16mm_dt_decompressed.npy"
    poses_path = seg_dir / "cam_poses.npy"
    intr_path = seg_dir / "cam_intrinsics.npy"
    spv_path = seg_dir / "scene_point_video.npy"

    for p in (depth_orig_path, depth_dec_path, poses_path, intr_path, spv_path):
        if not p.exists():
            print(f"[SKIP] {seg_dir}: 缺少文件 {p.name}，无法做 compare。")
            return

    depth_orig = np.load(str(depth_orig_path)).astype(np.float32)  # (T,H,W)
    depth_dec = np.load(str(depth_dec_path)).astype(np.float32)    # (T,H,W)
    if depth_orig.shape != depth_dec.shape:
        print(f"[WARN] {seg_dir}: 原始 depth shape={depth_orig.shape} 与 解压 depth shape={depth_dec.shape} 不一致，仅对重叠部分比较。")
        T = min(depth_orig.shape[0], depth_dec.shape[0])
        H = min(depth_orig.shape[1], depth_dec.shape[1])
        W = min(depth_orig.shape[2], depth_dec.shape[2])
        depth_orig = depth_orig[:T, :H, :W]
        depth_dec = depth_dec[:T, :H, :W]

    T, H, W = depth_orig.shape
    T_wc = load_cam_poses(poses_path)
    K_seq_raw = load_cam_intrinsics(intr_path, num_frames=T)
    spv = np.load(str(spv_path)).astype(np.float32)  # (T,H,W,3)
    if spv.ndim != 4 or spv.shape[3] != 3:
        print(f"[SKIP] {seg_dir}: scene_point_video shape 异常: {spv.shape}")
        return
    if spv.shape[0] < 1:
        print(f"[SKIP] {seg_dir}: scene_point_video 没有帧。")
        return

    # 只比较第一帧
    depth0_orig = depth_orig[0]
    depth0_dec = depth_dec[0]
    Twc0 = T_wc[0]
    # 将 DUSt3R 内参缩放到当前 depth 分辨率 (W,H)
    K0_raw = K_seq_raw[0]
    K0 = rescale_intrinsics_to_resolution(K0_raw, out_w=W, out_h=H)
    gt0 = spv[0]
    if gt0.shape[:2] != (H, W):
        print(f"[WARN] {seg_dir}: GT shape {gt0.shape} vs depth ({H},{W})，按重叠区域比较。")
        Hc = min(H, gt0.shape[0])
        Wc = min(W, gt0.shape[1])
        depth0_orig = depth0_orig[:Hc, :Wc]
        depth0_dec = depth0_dec[:Hc, :Wc]
        gt0 = gt0[:Hc, :Wc, :]
        H, W = Hc, Wc

    scene_orig = world_z_depth_to_scene_points_frame(depth0_orig, K0, Twc0)
    scene_dec = world_z_depth_to_scene_points_frame(depth0_dec, K0, Twc0)

    valid = np.isfinite(gt0).all(axis=-1) & (np.abs(gt0).sum(axis=-1) > 0)
    if not np.any(valid):
        print(f"[SKIP] {seg_dir}: GT 第一帧无有效点，无法比较。")
        return

    err_orig = np.linalg.norm(scene_orig.astype(np.float64) - gt0.astype(np.float64), axis=-1)
    err_dec = np.linalg.norm(scene_dec.astype(np.float64) - gt0.astype(np.float64), axis=-1)

    if plt is None:
        print("[WARN] 未安装 matplotlib，跳过误差热力图生成。")
        return

    # 在一个大图中画两套：原始 depth 与 压缩+解压 depth
    v_orig = err_orig[valid]
    v_dec = err_dec[valid]
    if v_orig.size == 0 or v_dec.size == 0:
        print(f"[WARN] {seg_dir}: 无有效误差像素。")
        return

    vmax = max(
        float(np.percentile(v_orig, 99.0)),
        float(np.percentile(v_dec, 99.0)),
    )
    if vmax <= 0:
        vmax = max(float(v_orig.max()), float(v_dec.max())) + 1e-6

    # 共同的直方图上限和 bin
    v_all = np.concatenate([v_orig, v_dec], axis=0)
    n_bins = min(80, max(20, int(v_all.size ** 0.3)))
    hist_max = float(np.percentile(v_all, 99.5))
    if hist_max <= 0:
        hist_max = float(v_all.max()) + 1e-6
    bins = np.linspace(0, hist_max, n_bins + 1)

    fig, axes = plt.subplots(
        2, 2, figsize=(10, 7), dpi=120, height_ratios=[1.2, 1]
    )
    (ax_h_orig, ax_h_dec), (ax_hist_orig, ax_hist_dec) = axes

    im0 = ax_h_orig.imshow(err_orig, cmap="jet", vmin=0, vmax=vmax)
    ax_h_orig.set_title("Depth(orig) → pointcloud vs scene_point (frame 0)")
    ax_h_orig.axis("off")
    plt.colorbar(im0, ax=ax_h_orig, fraction=0.046, pad=0.04)

    im1 = ax_h_dec.imshow(err_dec, cmap="jet", vmin=0, vmax=vmax)
    ax_h_dec.set_title("Depth(int16_mm+Δ_t) → pointcloud vs scene_point (frame 0)")
    ax_h_dec.axis("off")
    plt.colorbar(im1, ax=ax_h_dec, fraction=0.046, pad=0.04)

    ax_hist_orig.hist(v_orig, bins=bins, color="steelblue", edgecolor="none", alpha=0.8)
    ax_hist_orig.axvline(0.01, color="red", linestyle="--", linewidth=1.5, label="1 cm")
    ax_hist_orig.set_xlabel("3D L2 error (m)")
    ax_hist_orig.set_ylabel("Count")
    ax_hist_orig.set_title("Error dist (orig depth)")
    ax_hist_orig.legend(loc="upper right")
    ax_hist_orig.grid(True, alpha=0.3)

    ax_hist_dec.hist(v_dec, bins=bins, color="darkorange", edgecolor="none", alpha=0.8)
    ax_hist_dec.axvline(0.01, color="red", linestyle="--", linewidth=1.5, label="1 cm")
    ax_hist_dec.set_xlabel("3D L2 error (m)")
    ax_hist_dec.set_ylabel("Count")
    ax_hist_dec.set_title("Error dist (int16_mm+Δ_t depth)")
    ax_hist_dec.legend(loc="upper right")
    ax_hist_dec.grid(True, alpha=0.3)

    plt.tight_layout()
    out_big = seg_dir / "depth_vs_spv_error_compare.png"
    os.makedirs(out_big.parent, exist_ok=True)
    plt.savefig(out_big)
    plt.close(fig)

    print(f"[CMP] {seg_dir}: 保存对比误差大图: {out_big}")

def main():
    ap = argparse.ArgumentParser(
        description=(
            "递归压缩或解压 root 下各 seg 目录中的 depth_video.npy / depth_video_int16mm_dt.b2nd，"
            "压缩使用 Blosc2 int16(mm)+Δ_t 最佳方案 (B, 256,256,1,16, zstd:5:bitshuffle)。"
        )
    )
    ap.add_argument(
        "--mode",
        choices=["compress", "decompress", "compare"],
        default="compress",
        help=(
            "工作模式: "
            "compress=压缩 depth_video.npy, "
            "decompress=解压 depth_video_int16mm_dt.b2nd 为 .npy, "
            "compare=对比压缩前后 depth 恢复点云与 scene_point_video 的误差热力图。"
        ),
    )
    ap.add_argument(
        "--root",
        required=False,
        help="根目录，递归查找包含 depth_video.npy 的子目录（例如 Any4D 输出目录）。"
             "若同时提供 --seg_dir，则仅对 seg_dir 生效，--root 可省略。",
    )
    ap.add_argument(
        "--seg_dir",
        required=False,
        help="仅处理指定的单个子目录（该目录下须包含 depth_video.npy 或 depth_video_int16mm_dt.b2nd）。",
    )
    ap.add_argument(
        "--delete-existing",
        action="store_true",
        help=(
            "在 compress 模式下，若存在 scene_point_video.npy 以及已有的 "
            "depth_video_int16mm_dt.b2nd/meta/decompressed.npy，则先删除（默认保留）。"
        ),
    )
    ap.add_argument(
        "--mm-step-mm",
        type=float,
        default=5.0,
        help="定点量化的毫米步长（默认 5.0mm，数值越大压缩越强、精度越低）。",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的操作，不实际删除或压缩/解压。",
    )
    args = ap.parse_args()

    total_raw = 0.0
    total_comp = 0.0
    n_seg = 0

    seg_dirs: list[Path] = []
    if args.seg_dir:
        seg = Path(args.seg_dir).resolve()
        if not seg.exists():
            raise SystemExit(f"seg_dir 不存在: {seg}")
        seg_dirs = [seg]
        print(f"[INFO] 仅处理单个 seg_dir={seg}")
    else:
        if not args.root:
            raise SystemExit("--root 或 --seg_dir 至少提供一个")
        root = Path(args.root).resolve()
        if not root.exists():
            raise SystemExit(f"root 不存在: {root}")
        print(f"[INFO] root={root}")
        seg_dirs = sorted(find_depth_dirs(root))

    for seg_dir in seg_dirs:
        if args.mode == "compress":
            stats = compress_depth_dir(
                seg_dir,
                dry_run=args.dry_run,
                mm_step_mm=args.mm_step_mm,
                delete_existing=args.delete_existing,
            )
            if not stats["skipped"]:
                total_raw += stats["raw_mb"]
                total_comp += stats["comp_mb"]
                if stats["raw_mb"] > 0:
                    n_seg += 1
        elif args.mode == "decompress":
            b2nd_path = seg_dir / "depth_video_int16mm_dt.b2nd"
            if not b2nd_path.exists():
                print(f"[SKIP] {seg_dir} 不存在 depth_video_int16mm_dt.b2nd")
                continue
            out_npy = seg_dir / "depth_video_int16mm_dt_decompressed.npy"
            print(f"[DECOMP] {b2nd_path} -> {out_npy}")
            if args.dry_run:
                continue
            t0 = time.time()
            # 读取 meta 获取 scale/mm_step，如果不存在则回退到默认 scale=1000 (1mm)
            meta_path = seg_dir / "depth_video_int16mm_dt.meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                scale = float(meta.get("scale", 200.0))
                mm_step_mm = float(meta.get("mm_step_mm", 5.0))
            else:
                scale = 200.0  # 1000/5
                mm_step_mm = 5.0
                print(f"[WARN] 未找到 meta.json, 使用默认 scale={scale}, mm_step_mm={mm_step_mm}")

            nd = blosc2.open(b2nd_path)
            rec_delta = nd[:]  # (T,H,W) int16 delta
            t1 = time.time()

            # XOR 反解码（沿 axis=0）
            if rec_delta.dtype != np.int16:
                raise SystemExit(f"[FAIL] 解码数据类型错误，期望 int16，得到 {rec_delta.dtype}")
            u_rec = rec_delta.view(np.uint16)
            u_out = np.empty_like(u_rec)
            u_out[0, ...] = u_rec[0, ...]
            for t in range(1, u_rec.shape[0]):
                u_out[t, ...] = u_rec[t, ...] ^ u_out[t - 1, ...]
            rec_i16 = u_out.view(np.int16)
            rec_f = rec_i16.astype(np.float32) / scale  # (T,H,W) float32, 单位 m
            np.save(out_npy, rec_f.astype(np.float16))
            print(f"[OK] 解压完成，用时 {t1 - t0:.2f}s, 保存为 {out_npy}")
        else:  # compare
            if args.dry_run:
                print(f"[CMP-DRY] 将对 {seg_dir} 做原始 depth 与 解压 depth 的点云误差对比。")
                continue
            compare_depth_dir(seg_dir)

    if args.mode == "compress" and n_seg > 0 and not args.dry_run:
        ratio = total_raw / total_comp if total_comp > 0 else 0.0
        print(
            f"\n[SUMMARY] 处理 seg 目录: {n_seg} 个, "
            f"合计原始={total_raw/1000:.2f} GB, 压缩后={total_comp/1000:.2f} GB, "
            f"整体压缩比={ratio:.2f}x"
        )
    elif args.mode == "compress" and args.dry_run:
        print("\n[SUMMARY] dry-run 模式，仅展示将要执行的操作。")
    elif args.mode == "compress":
        print("\n[SUMMARY] 未找到任何 depth_video.npy。")


if __name__ == "__main__":
    main()

