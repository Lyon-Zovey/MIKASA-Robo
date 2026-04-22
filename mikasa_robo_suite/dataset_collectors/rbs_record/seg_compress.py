#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np


def _default_compress_output(seg_npy: Path, method: str) -> Path:
    if method == "b2nd":
        return seg_npy.with_name(f"{seg_npy.stem}.b2nd")
    if method == "npz":
        return seg_npy.with_name(f"{seg_npy.stem}.npz")
    raise ValueError(f"Unsupported method: {method}")


def compress_seg(
    seg_npy: Path,
    method: str = "b2nd",
    out_path: Optional[Path] = None,
    clevel: int = 5,
    verify: bool = True,
    delete_source: bool = False,
) -> Path:
    seg_npy = Path(seg_npy)
    if not seg_npy.exists():
        raise FileNotFoundError(seg_npy)

    arr = np.load(seg_npy, allow_pickle=False)
    out_path = Path(out_path) if out_path is not None else _default_compress_output(seg_npy, method)

    if method == "npz":
        np.savez_compressed(out_path, seg=arr)
    elif method == "b2nd":
        try:
            import blosc2
        except Exception as e:
            raise RuntimeError("blosc2 is required for method='b2nd'") from e

        chunks = None
        if arr.ndim == 3:
            chunks = (min(16, arr.shape[0]), min(256, arr.shape[1]), min(256, arr.shape[2]))
        elif arr.ndim == 2:
            chunks = (min(256, arr.shape[0]), min(256, arr.shape[1]))

        cparams = blosc2.CParams(
            codec=blosc2.Codec.ZSTD,
            clevel=int(clevel),
            filters=[blosc2.Filter.BITSHUFFLE],
        )
        nd = blosc2.asarray(arr, chunks=chunks, cparams=cparams)
        nd.save(str(out_path))

        meta = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "method": "b2nd",
            "codec": "zstd",
            "clevel": int(clevel),
        }
        meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported method: {method}")

    if verify:
        rec = decompress_seg(out_path)
        if not np.array_equal(arr, rec):
            raise RuntimeError("Verification failed: decompressed seg is not identical to source")

    if delete_source:
        seg_npy.unlink(missing_ok=True)

    return out_path


def decompress_seg(comp_path: Path, out_path: Optional[Path] = None) -> np.ndarray:
    comp_path = Path(comp_path)
    if not comp_path.exists():
        raise FileNotFoundError(comp_path)

    if comp_path.suffix == ".npz":
        z = np.load(comp_path, allow_pickle=False)
        if "seg" in z:
            arr = z["seg"]
        else:
            first_key = list(z.keys())[0]
            arr = z[first_key]
    elif comp_path.suffix == ".b2nd":
        try:
            import blosc2
        except Exception as e:
            raise RuntimeError("blosc2 is required to decompress .b2nd") from e
        arr = blosc2.open(str(comp_path))[:]
    else:
        raise ValueError(f"Unsupported compressed file suffix: {comp_path.suffix}")

    if out_path is not None:
        out_path = Path(out_path)
        np.save(out_path, arr)

    return arr


def main():
    parser = argparse.ArgumentParser(description="Compress/decompress seg.npy")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_c = sub.add_parser("compress", help="compress seg.npy")
    p_c.add_argument("--seg", type=str, help="Path to seg.npy")
    p_c.add_argument("--seg-dir", type=str, help="Directory containing seg.npy")
    p_c.add_argument("--method", type=str, default="b2nd", choices=["b2nd", "npz"])
    p_c.add_argument("--out", type=str, default=None, help="Output compressed file path")
    p_c.add_argument("--clevel", type=int, default=5)
    p_c.add_argument("--no-verify", action="store_true")
    p_c.add_argument("--delete-source", action="store_true")

    p_d = sub.add_parser("decompress", help="decompress seg compressed file")
    p_d.add_argument("--in-file", type=str, required=True, help="Input compressed file (.b2nd/.npz)")
    p_d.add_argument("--out", type=str, default=None, help="Output .npy path (default: seg_decompress.npy beside input)")

    args = parser.parse_args()

    if args.cmd == "compress":
        if args.seg is not None:
            seg_npy = Path(args.seg)
        elif args.seg_dir is not None:
            seg_npy = Path(args.seg_dir) / "seg.npy"
        else:
            raise SystemExit("Please provide --seg or --seg-dir")

        out = Path(args.out) if args.out is not None else None
        out_path = compress_seg(
            seg_npy=seg_npy,
            method=args.method,
            out_path=out,
            clevel=args.clevel,
            verify=not args.no_verify,
            delete_source=args.delete_source,
        )
        raw_bytes = seg_npy.stat().st_size if seg_npy.exists() else 0
        comp_bytes = out_path.stat().st_size
        ratio = (raw_bytes / comp_bytes) if (raw_bytes > 0 and comp_bytes > 0) else 0.0
        print(f"compressed: {out_path}")
        if ratio > 0:
            print(f"ratio: {ratio:.2f}x")

    elif args.cmd == "decompress":
        in_file = Path(args.in_file)
        out = Path(args.out) if args.out is not None else in_file.with_name("seg_decompress.npy")
        arr = decompress_seg(in_file, out)
        print(f"decompressed: {out}")
        print(f"shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
