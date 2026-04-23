#!/usr/bin/env python3
"""
Compress / decompress / verify multi-reference scene point flow.

Supports the current production format: 4ch RGBA v3, 10-bit, H.265 CRF=0,
as well as 8-bit FFV1 RGBA (legacy) and other codec/bit-depth combinations.

Encoding scheme (ref-based delta, v3, no keyframe sidecar):
  1. delta[t] = flow[t] - anchor   (each frame vs fixed anchor, NO accumulation)
  2. Per-frame scale -> alpha channel (quantized to QMAX = 2^bits - 1)
  3. Normalize delta by *quantized* scale (v3 key improvement), then -> RGB
  4. Encode RGBA as video. For codecs without alpha (H.265, VP9): spatial tile
     (T, H, W*2, 3) = [RGB | AAA]

Decoding:
  1. Decode video -> RGBA (un-tile if needed)
  2. scale_q[t] = alpha[t] / QMAX * max_scale   (from JSON sidecar)
  3. delta[t] = (RGB[t] / QMID - 1) * scale_q[t]
  4. flow[t] = anchor + delta[t]

Output files per flow:
  scene_point_flow_refXXXXX_v3_10b_h265_crf0.mp4   -- compressed flow video
  scene_point_flow_refXXXXX_v3_10b_h265_crf0.json  -- sidecar (max_scale, bits, orig_w, ...)
  scene_point_flow_refXXXXX.anchor.npy              -- anchor point cloud (ref frame)

Usage:
  # Compress all flows in one video output dir
  python flow_compress.py compress --out_dir /path/to/video_output

  # Compress all videos under out_root
  python flow_compress.py compress --out_root /path/to/out_root

  # Verify round-trip accuracy
  python flow_compress.py verify --out_dir /path/to/video_output

  # Decompress one compressed flow back to npy
  python flow_compress.py decompress \\
      --video /path/to/scene_point_flow_ref00000_v3_10b_h265_crf0.mp4 \\
      --anchor /path/to/scene_point_flow_ref00000.anchor.npy

  # Decompress (legacy: anchor from scene_point_video.npy)
  python flow_compress.py decompress \\
      --video /path/to/flow_ref00000_v3_10b_h265_crf0.mp4 \\
      --spv /path/to/scene_point_video.npy --ref_idx 0
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCALE_EPS = 1e-9
SCALE_FLOOR_DEFAULT = 0.01  # Per-frame scale floor: frames with max|delta| below this
                            # are treated as static (alpha=0, rgb=mid-gray).
                            # Prevents noise amplification in near-static frames that
                            # would otherwise produce high-entropy RGB and bloat H.265.


def _qmax(bits: int) -> int:
    return (1 << bits) - 1  # 255 (8b) or 1023 (10b)


def _qmid(bits: int) -> float:
    return _qmax(bits) / 2.0  # 127.5 or 511.5


def _container(codec: str) -> str:
    if codec == "ffv1":
        return ".mkv"
    if codec == "libvpx-vp9":
        return ".webm"
    return ".mp4"


# ---------------------------------------------------------------------------
# Spatial tiling  (for codecs without native alpha: H.265, VP9)
# ---------------------------------------------------------------------------

def _rgba_to_tiled_rgb(rgba: np.ndarray) -> np.ndarray:
    """(T,H,W,4) -> (T,H,W*2,3). Works for uint8 or uint16."""
    T, H, W, _ = rgba.shape
    tiled = np.zeros((T, H, W * 2, 3), dtype=rgba.dtype)
    tiled[:, :, :W, :] = rgba[:, :, :, :3]
    tiled[:, :, W:, 0] = rgba[:, :, :, 3]
    tiled[:, :, W:, 1] = rgba[:, :, :, 3]
    tiled[:, :, W:, 2] = rgba[:, :, :, 3]
    return tiled


def _tiled_rgb_to_rgba(tiled: np.ndarray, orig_w: int) -> np.ndarray:
    """(T,H,W*2,3) -> (T,H,orig_w,4). Inverse of _rgba_to_tiled_rgb."""
    T, H, W2, _ = tiled.shape
    rgba = np.zeros((T, H, orig_w, 4), dtype=tiled.dtype)
    rgba[:, :, :, :3] = tiled[:, :, :orig_w, :]
    rgba[:, :, :, 3] = tiled[:, :, orig_w:, 0]
    return rgba


# ---------------------------------------------------------------------------
# Video I/O  (supports 8-bit and 10-bit, native RGBA and tiled RGB)
# ---------------------------------------------------------------------------

def _encode(frames: np.ndarray, path: str, codec: str, pix_fmt: str,
            crf: Optional[int], native_alpha: bool, bits: int,
            preset: str = "ultrafast", threads: Optional[int] = None):
    if threads is None:
        threads = max(1, int(os.environ.get("RBS_FFMPEG_THREADS", "2")))
    """Encode frames to video file via ffmpeg pipe."""
    T, H, W, C = frames.shape
    if bits == 10:
        feed = (frames.astype(np.uint16) * 64).tobytes()  # left-shift to 16-bit
        in_fmt = "rgba64le" if (native_alpha and C == 4) else "rgb48le"
    else:
        feed = frames.tobytes()
        in_fmt = "rgba" if (native_alpha and C == 4) else "rgb24"

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", in_fmt,
           "-s", f"{W}x{H}", "-r", "1", "-i", "pipe:0"]
    if codec == "ffv1":
        cmd += ["-c:v", "ffv1", "-level", "3", "-pix_fmt", pix_fmt]
    elif codec == "libx265":
        cmd += ["-c:v", "libx265", "-preset", preset,
                "-x265-params", f"pools={threads}:frame-threads=1",
                "-crf", str(crf), "-pix_fmt", pix_fmt, "-tag:v", "hvc1"]
    elif codec == "libvpx-vp9":
        cmd += ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0",
                "-pix_fmt", pix_fmt, "-row-mt", "1"]
    cmd.append(path)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, err = p.communicate(input=feed)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed ({codec}): {err.decode()}")


def _decode(path: str, native_alpha: bool, bits: int) -> np.ndarray:
    """Decode video to numpy array via ffmpeg pipe."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames",
         "-of", "json", path], capture_output=True, text=True)
    info = json.loads(probe.stdout)["streams"][0]
    W, H = int(info["width"]), int(info["height"])
    nb = info.get("nb_frames", "N/A")
    if nb in ("N/A", "0"):
        c = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames",
             "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames",
             "-of", "json", path], capture_output=True, text=True)
        nb = json.loads(c.stdout)["streams"][0]["nb_read_frames"]
    T = int(nb)

    if bits == 10:
        out_fmt = "rgba64le" if native_alpha else "rgb48le"
    else:
        out_fmt = "rgba" if native_alpha else "rgb24"
    C = 4 if native_alpha else 3

    r = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", path,
         "-f", "rawvideo", "-pix_fmt", out_fmt, "pipe:1"],
        capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {r.stderr.decode()}")

    if bits == 10:
        raw = np.frombuffer(r.stdout, dtype=np.uint16).reshape(T, H, W, C)
        return (raw // 64).astype(np.uint16)  # shift back to 10-bit [0, 1023]
    else:
        return np.frombuffer(r.stdout, dtype=np.uint8).reshape(T, H, W, C)


# ---------------------------------------------------------------------------
# Build RGBA frames (v3 scheme, 8/10 bit)
# ---------------------------------------------------------------------------

def _build_rgba_v3(deltas: np.ndarray, per_frame_scale: np.ndarray,
                   max_scale: float, bits: int,
                   scale_floor: float = 0.0) -> np.ndarray:
    """Build (T,H,W,4) quantized RGBA frames using v3 scheme.

    Args:
        scale_floor: Per-frame scale floor.  Frames whose per_frame_scale is
            below this value are treated as static (alpha=0, rgb=mid-gray).
            This suppresses model noise in near-static frames, producing flat
            frames that H.265 can compress trivially.  Default 0.0 (disabled).
    """
    T, H, W, _ = deltas.shape
    QMAX = _qmax(bits)
    QMID = _qmid(bits)
    dtype = np.uint16 if bits > 8 else np.uint8
    _floor = max(scale_floor, SCALE_EPS)
    is_zero = (per_frame_scale < _floor)

    # Quantize scale -> alpha
    alpha = np.clip(np.round(per_frame_scale / max_scale * QMAX),
                    0, QMAX).astype(dtype)
    alpha[is_zero] = 0
    alpha[~is_zero & (alpha == 0)] = 1

    # Dequantize scale (decoder will use this exact value)
    scale_q = alpha.astype(np.float64) / QMAX * max_scale
    scale_q_safe = np.where(scale_q < SCALE_EPS, 1.0, scale_q)

    # Normalize by quantized scale, then quantize RGB
    s = scale_q_safe[:, None, None, None]
    normalized = deltas / s
    rgb = np.clip(np.round((normalized + 1.0) * QMID), 0, QMAX).astype(dtype)
    rgb[is_zero] = int(round(QMID))

    frames = np.zeros((T, H, W, 4), dtype=dtype)
    frames[:, :, :, :3] = rgb
    frames[:, :, :, 3] = alpha[:, None, None]
    return frames


# ---------------------------------------------------------------------------
# Decompress  (reads any variant produced by this tool or the inference pipeline)
# ---------------------------------------------------------------------------

def decompress_one_flow(
    video_path: Path,
    anchor_pts: np.ndarray,
    sidecar: Optional[dict] = None,
) -> np.ndarray:
    """Decompress compressed flow video back to (T, H, W, 3) float32.

    Args:
        video_path: Path to the .mp4/.mkv/.webm compressed flow video.
        anchor_pts: (H, W, 3) float -- anchor point cloud for this reference frame.
        sidecar: Optional sidecar dict. If None, reads from .json next to video.

    Returns:
        flow: (T, H, W, 3) float32 -- reconstructed scene point flow.
    """
    video_path = Path(video_path)
    if sidecar is None:
        sp = video_path.with_suffix(".json")
        if sp.exists():
            sidecar = json.loads(sp.read_text(encoding="utf-8"))
        else:
            sidecar = {}

    max_scale = sidecar.get("max_scale", 1.0)
    native_alpha = sidecar.get("native_alpha", False)
    bits = sidecar.get("bits", 8)
    orig_w = sidecar.get("orig_w", None)

    QMAX = _qmax(bits)
    QMID = _qmid(bits)

    # Decode video
    raw = _decode(str(video_path), native_alpha, bits)

    # Un-tile if needed
    if native_alpha:
        rgba = raw
    else:
        if orig_w is None:
            # Infer: tiled width = 2 * orig_w
            _, H, W2, _ = raw.shape
            orig_w = W2 // 2
        rgba = _tiled_rgb_to_rgba(raw, orig_w)

    T, H, W, _ = rgba.shape

    # Reconstruct flow
    alpha_vals = rgba[:, 0, 0, 3].astype(np.float64)  # uniform per frame
    scale_t = alpha_vals / QMAX * max_scale

    norm = rgba[:, :, :, :3].astype(np.float32) / QMID - 1.0
    delta = norm * scale_t[:, None, None, None].astype(np.float32)

    anchor = anchor_pts.astype(np.float32)
    flow = anchor[None] + delta

    # Zero-scale frames (reference frame itself): use anchor directly
    zero_mask = (alpha_vals == 0)
    if zero_mask.any():
        flow[zero_mask] = anchor[None]

    return flow


# ---------------------------------------------------------------------------
# Compress
# ---------------------------------------------------------------------------

def compress_one_flow(
    flow_npy_path: Path,
    anchor_pts: np.ndarray,
    codec: str = "libx265",
    crf: int = 0,
    bits: int = 10,
    pix_fmt: Optional[str] = None,
    suffix: Optional[str] = None,
    delete_npy: bool = False,
    scale_floor: Optional[float] = None,
) -> dict:
    """Compress one flow .npy into a video file.

    Args:
        flow_npy_path: Path to the flow .npy file.
        anchor_pts: (H, W, 3) float -- anchor point cloud.
        codec: Video codec (libx265, ffv1, libvpx-vp9).
        crf: CRF value (0 = near-lossless). Ignored for ffv1.
        bits: Quantization bits (8 or 10).
        pix_fmt: Pixel format override. If None, auto-selected.
        suffix: Filename suffix override. If None, auto-generated.
        delete_npy: Delete original .npy after successful compression.
        scale_floor: Per-frame scale floor. Frames whose max |delta| is
            below this value are treated as static (alpha=0, rgb=mid-gray),
            suppressing model noise and enabling H.265 to compress them
            trivially. Defaults to SCALE_FLOOR_DEFAULT (0.01 = 10mm).
            Set to 0 to disable.

    Returns:
        Sidecar dict (also written to .json file).
    """
    if scale_floor is None:
        scale_floor = SCALE_FLOOR_DEFAULT

    flow_npy_path = Path(flow_npy_path)
    flow = np.load(str(flow_npy_path)).astype(np.float32)
    T, H, W, _ = flow.shape
    anchor = anchor_pts.astype(np.float32)

    # Delta relative to anchor
    deltas = flow - anchor[np.newaxis, :, :, :]
    deltas[~np.isfinite(deltas)] = 0.0

    # Per-frame scale
    per_frame_scale = np.abs(deltas).reshape(T, -1).max(axis=1)
    max_scale = float(per_frame_scale.max())
    if max_scale < SCALE_EPS:
        max_scale = SCALE_EPS

    # Count frames that will be treated as static
    _effective_floor = max(scale_floor, SCALE_EPS)
    n_floored = int((per_frame_scale < _effective_floor).sum())

    # Build RGBA
    rgba = _build_rgba_v3(deltas, per_frame_scale, max_scale, bits,
                          scale_floor=scale_floor)

    # Determine native alpha or tiling
    native_alpha = (codec == "ffv1")

    # Auto pix_fmt
    if pix_fmt is None:
        if codec == "ffv1":
            pix_fmt = "gbrap10le" if bits == 10 else "gbrap"
        elif codec == "libx265":
            pix_fmt = "yuv444p10le" if bits == 10 else "yuv444p"
        elif codec == "libvpx-vp9":
            pix_fmt = "yuv444p10le" if bits == 10 else "yuv444p"

    # Auto suffix
    if suffix is None:
        b_str = f"{bits}b" if bits != 8 else ""
        if codec == "ffv1":
            suffix = f"v3_{b_str}_rgba" if b_str else "v3_rgba"
        elif codec == "libx265":
            suffix = f"v3_{b_str}_h265_crf{crf}" if b_str else f"v3_h265_crf{crf}"
        else:
            c = codec.replace("libvpx-", "")
            suffix = f"v3_{b_str}_{c}_crf{crf}" if b_str else f"v3_{c}_crf{crf}"

    ext = _container(codec)
    vid_path = flow_npy_path.parent / f"{flow_npy_path.stem}_{suffix}{ext}"
    json_path = vid_path.with_suffix(".json")

    # Encode
    if native_alpha:
        feed = rgba
    else:
        feed = _rgba_to_tiled_rgb(rgba)

    t0 = time.time()
    _encode(feed, str(vid_path), codec, pix_fmt,
            crf if codec != "ffv1" else None, native_alpha, bits)
    enc_time = time.time() - t0

    npy_size = os.path.getsize(str(flow_npy_path))
    vid_size = os.path.getsize(str(vid_path))

    sidecar = {
        "encoding": "ref_delta_4ch_v3",
        "max_scale": max_scale,
        "native_alpha": native_alpha,
        "bits": bits,
        "orig_w": W,
        "codec": codec,
        "crf": crf if codec != "ffv1" else None,
        "pix_fmt": pix_fmt,
        "shape_THW": [T, H, W],
        "npy_size_mb": round(npy_size / 1e6, 2),
        "video_size_mb": round(vid_size / 1e6, 2),
        "compression_ratio": round(npy_size / max(vid_size, 1), 2),
        "encode_time_s": round(enc_time, 3),
        "scale_floor": scale_floor,
        "frames_floored": n_floored,
    }
    json_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    if delete_npy:
        flow_npy_path.unlink()

    return sidecar


# ---------------------------------------------------------------------------
# Discover flow entries
# ---------------------------------------------------------------------------

def _discover_flow_entries(out_dir: Path) -> List[dict]:
    """Discover flow files from meta.json, metadata.json, or glob fallback."""
    # Multi-ref meta.json
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        entries = meta.get("outputs", {}).get("scene_point_flows", [])
        if entries:
            return entries

    # Single-ref metadata.json
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        flow_info = meta.get("outputs", {}).get("scene_point_flow")
        if flow_info and isinstance(flow_info, dict):
            ref_idx = meta.get("reference_frame_idx", 0)
            return [{"reference_frame": ref_idx, "file": flow_info["file"]}]

    # Glob fallback
    entries = []
    for fp in sorted(out_dir.glob("scene_point_flow*.npy")):
        if ".anchor." in fp.name or ".kf." in fp.name:
            continue
        name = fp.stem
        if name == "scene_point_flow":
            entries.append({"reference_frame": 0, "file": fp.name})
        elif name.startswith("scene_point_flow_ref"):
            try:
                ref_idx = int(name.split("ref")[-1])
                entries.append({"reference_frame": ref_idx, "file": fp.name})
            except ValueError:
                pass
    return entries


# ---------------------------------------------------------------------------
# Batch compress
# ---------------------------------------------------------------------------

def compress_out_dir(out_dir: Path, **kw) -> List[dict]:
    """Compress all flow npy files in a video output directory."""
    out_dir = Path(out_dir)

    # Need anchor: try .anchor.npy first, fallback to scene_point_video.npy
    spv_path = out_dir / "scene_point_video.npy"

    flow_entries = _discover_flow_entries(out_dir)
    if not flow_entries:
        return []

    results = []
    for entry in flow_entries:
        ref_idx = entry["reference_frame"]
        fname = entry["file"]
        fp = out_dir / fname
        if not fp.exists():
            continue

        # Find anchor
        anchor_path = out_dir / f"scene_point_flow_ref{ref_idx:05d}.anchor.npy"
        if anchor_path.exists():
            anchor = np.load(str(anchor_path)).astype(np.float32)
        elif spv_path.exists():
            spv = np.load(str(spv_path), mmap_mode="r")
            anchor = spv[ref_idx].astype(np.float32)
        else:
            print(f"    SKIP {fname}: no anchor found")
            continue

        print(f"    {fname} ...", end="", flush=True)
        s = compress_one_flow(fp, anchor, **kw)
        print(f" {s['npy_size_mb']:.0f}MB -> {s['video_size_mb']:.1f}MB "
              f"({s['compression_ratio']:.1f}x, {s['encode_time_s']:.1f}s)")
        results.append(s)
    return results


# ---------------------------------------------------------------------------
# Verify (round-trip)
# ---------------------------------------------------------------------------

def verify_out_dir(out_dir: Path, **kw):
    """Compress (if needed) and verify round-trip accuracy for flows in out_dir."""
    out_dir = Path(out_dir)
    spv_path = out_dir / "scene_point_video.npy"

    entries = _discover_flow_entries(out_dir)

    for entry in entries[:2]:
        ref_idx = entry["reference_frame"]
        fname = entry["file"]
        fp = out_dir / fname
        if not fp.exists():
            continue

        # Find anchor
        anchor_path = out_dir / f"scene_point_flow_ref{ref_idx:05d}.anchor.npy"
        if anchor_path.exists():
            anchor = np.load(str(anchor_path)).astype(np.float32)
        elif spv_path.exists():
            spv = np.load(str(spv_path), mmap_mode="r")
            anchor = spv[ref_idx].astype(np.float32)
        else:
            continue

        orig = np.load(str(fp)).astype(np.float32)
        T = orig.shape[0]

        # Find existing compressed file or compress now
        vid_path = None
        for pat in [f"{fp.stem}_v3_10b_h265_crf0.mp4",
                    f"{fp.stem}_v3_rgba.mkv",
                    f"{fp.stem}_v3_10b_rgba.mkv"]:
            candidate = out_dir / pat
            if candidate.exists():
                vid_path = candidate
                break
        if vid_path is None:
            compress_one_flow(fp, anchor, **kw)
            # Find the file we just created
            for f in sorted(out_dir.glob(f"{fp.stem}_v3_*")):
                if f.suffix in (".mp4", ".mkv", ".webm"):
                    vid_path = f
                    break
        if vid_path is None:
            continue

        recon = decompress_one_flow(vid_path, anchor_pts=anchor)

        ok = np.isfinite(orig) & np.isfinite(recon)
        if ok.sum() == 0:
            continue
        o_flat = orig[ok].reshape(-1, 3)
        r_flat = recon[ok].reshape(-1, 3)
        l2 = np.sqrt(np.sum((o_flat - r_flat) ** 2, axis=-1))
        err = np.abs(orig[ok] - recon[ok])

        npy_mb = os.path.getsize(str(fp)) / 1e6
        vid_mb = os.path.getsize(str(vid_path)) / 1e6

        print(f"\n  {fname} (ref={ref_idx}, T={T}):")
        print(f"    Original:   {npy_mb:.0f} MB")
        print(f"    Compressed: {vid_mb:.1f} MB ({npy_mb/vid_mb:.1f}x)")
        print(f"    Per-axis err: mean={err.mean()*1000:.3f}mm, "
              f"P99={np.percentile(err,99)*1000:.3f}mm, "
              f"max={err.max()*1000:.3f}mm")
        print(f"    L2 err:       mean={l2.mean()*1000:.3f}mm, "
              f"P99={np.percentile(l2,99)*1000:.3f}mm, "
              f"max={l2.max()*1000:.3f}mm")

        pfl = []
        for t in range(T):
            m = np.isfinite(orig[t]).all(-1) & np.isfinite(recon[t]).all(-1)
            if m.sum() == 0:
                pfl.append(0.0)
                continue
            e = np.sqrt(np.sum((orig[t][m] - recon[t][m]) ** 2, axis=-1))
            pfl.append(float(e.mean()))
        pfl = np.array(pfl)
        print(f"    Per-frame L2: first={pfl[0]*1000:.3f}mm, "
              f"mid={pfl[T//2]*1000:.3f}mm, last={pfl[-1]*1000:.3f}mm, "
              f"max={pfl.max()*1000:.3f}mm")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compress/decompress/verify scene point flow files.")
    sub = ap.add_subparsers(dest="cmd")

    # --- compress ---
    p_c = sub.add_parser("compress", help="Compress flow .npy to video")
    p_c.add_argument("--out_dir", type=str, default=None,
                     help="Single video output directory")
    p_c.add_argument("--out_root", type=str, default=None,
                     help="Root containing multiple video output directories")
    p_c.add_argument("--codec", default="libx265",
                     choices=["libx265", "ffv1", "libvpx-vp9"],
                     help="Video codec (default: libx265)")
    p_c.add_argument("--crf", type=int, default=0,
                     help="CRF value for lossy codecs (default: 0, near-lossless)")
    p_c.add_argument("--bits", type=int, default=10, choices=[8, 10],
                     help="Quantization bit depth (default: 10)")
    p_c.add_argument("--delete_npy", action="store_true",
                     help="Delete original .npy after compression")

    # --- verify ---
    p_v = sub.add_parser("verify", help="Verify round-trip accuracy")
    p_v.add_argument("--out_dir", required=True)

    # --- decompress ---
    p_d = sub.add_parser("decompress", help="Decompress flow video to .npy")
    p_d.add_argument("--video", required=True,
                     help="Path to compressed flow video (.mp4/.mkv/.webm)")
    p_d.add_argument("--anchor", type=str, default=None,
                     help="Path to .anchor.npy file")
    p_d.add_argument("--spv", type=str, default=None,
                     help="Path to scene_point_video.npy (legacy, use with --ref_idx)")
    p_d.add_argument("--ref_idx", type=int, default=0,
                     help="Reference frame index (only with --spv)")
    p_d.add_argument("--output", default=None,
                     help="Output .npy path (default: <video_stem>_decompressed.npy)")

    args = ap.parse_args()

    if args.cmd == "compress":
        kw = dict(codec=args.codec, crf=args.crf, bits=args.bits,
                  delete_npy=args.delete_npy)
        if args.out_dir:
            results = compress_out_dir(Path(args.out_dir), **kw)
            if results:
                tn = sum(r["npy_size_mb"] for r in results)
                tc = sum(r["video_size_mb"] for r in results)
                print(f"\n  Total: {tn:.0f}MB -> {tc:.1f}MB ({tn/tc:.1f}x)")
        elif args.out_root:
            ar = []
            for done in sorted(Path(args.out_root).rglob("_DONE")):
                d = done.parent
                print(f"\n  {d.relative_to(Path(args.out_root))}:")
                ar.extend(compress_out_dir(d, **kw))
            if ar:
                tn = sum(r["npy_size_mb"] for r in ar)
                tc = sum(r["video_size_mb"] for r in ar)
                print(f"\n=== Total: {tn/1000:.1f}GB -> {tc/1000:.1f}GB "
                      f"({tn/tc:.1f}x) ===")
        else:
            print("--out_dir or --out_root required")
            sys.exit(1)

    elif args.cmd == "verify":
        verify_out_dir(Path(args.out_dir))

    elif args.cmd == "decompress":
        vid = Path(args.video)
        # Load anchor
        if args.anchor:
            anchor = np.load(args.anchor).astype(np.float32)
        elif args.spv:
            spv = np.load(args.spv, mmap_mode="r")
            anchor = spv[args.ref_idx].astype(np.float32)
        else:
            # Try to find .anchor.npy next to the video
            # Parse ref index from filename: scene_point_flow_refXXXXX_...
            stem = vid.stem
            anchor_path = None
            for part in stem.split("_"):
                if part.startswith("ref") and part[3:].isdigit():
                    ref_name = f"scene_point_flow_{part}.anchor.npy"
                    candidate = vid.parent / ref_name
                    if candidate.exists():
                        anchor_path = candidate
                    break
            if anchor_path is None:
                print("Error: provide --anchor or --spv + --ref_idx")
                sys.exit(1)
            anchor = np.load(str(anchor_path)).astype(np.float32)

        flow = decompress_one_flow(vid, anchor_pts=anchor)
        out = args.output or str(vid.with_name(vid.stem + "_decompressed.npy"))
        np.save(out, flow.astype(np.float16))
        print(f"Saved {flow.shape} float16 to {out}")

    else:
        ap.print_help()


if __name__ == "__main__":
    main()
