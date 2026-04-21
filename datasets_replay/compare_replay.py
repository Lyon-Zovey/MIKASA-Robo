"""
Compare Replay Script for MIKASA-Robo
========================================
Runs BOTH replay methods on the same episode, then stitches them into a single
side-by-side comparison video so you can see the difference directly.

Left  panel → replay_dataset  : raw pixels stored in .npz (no simulation)
Right panel → action_replay   : live SAPIEN simulation fed with stored actions

Usage:
    conda activate mikasa-robo
    cd ~/Sim_Data/MIKASA-Robo

    python3 compare_replay.py --episode 0 --seed 0
    python3 compare_replay.py --episode 3 --seed 0 --fps 10
"""

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

# ─── Defaults ────────────────────────────────────────────────────────────────

DATA_DIR   = "/home/CNF2026716696/Sim_Data/MIKASA_Data/RememberShape3-v0"
VIDEO_DIR  = "/home/CNF2026716696/Sim_Data/MIKASA-Robo/videos"
ENV_ID     = "RememberShape3-v0"
TARGET_H   = 512          # common height for both panels
FONT       = cv2.FONT_HERSHEY_SIMPLEX


# ─── Step 1: run replay_dataset.py (stored pixels) ───────────────────────────

def run_replay_dataset(episode: int, fps: int) -> str:
    """Returns the path to the saved video."""
    out_dir  = os.path.join(VIDEO_DIR, f"replay_{ENV_ID}")
    out_path = os.path.join(out_dir, f"episode_{episode:04d}.mp4")
    print(f"\n[1/3] replay_dataset.py  →  {out_path}")
    cmd = [
        sys.executable, "replay_dataset.py",
        "--episode", str(episode),
        "--fps",     str(fps),
        "--quiet",
    ]
    subprocess.run(cmd, check=True)
    return out_path


# ─── Step 2: run action_replay.py (live simulation) ──────────────────────────

def run_action_replay(episode: int, seed: int, fps: int) -> str:
    """Returns the path to the saved video."""
    out_dir  = os.path.join(VIDEO_DIR, f"action_replay_{ENV_ID}")
    out_path = os.path.join(out_dir, f"train_data_{episode}.mp4")
    # action_replay.py names the file after the .npz basename: train_data_<N>.mp4
    print(f"\n[2/3] action_replay.py   →  {out_path}")
    cmd = [
        sys.executable, "action_replay.py",
        "--episode", str(episode),
        "--seed",    str(seed),
        "--fps",     str(fps),
    ]
    subprocess.run(cmd, check=True)
    return out_path


# ─── Step 3: stitch both videos side by side ─────────────────────────────────

def read_video_frames(path: str):
    """Read all frames from an mp4, return list of BGR np.ndarray."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def resize_to_height(frame: np.ndarray, h: int) -> np.ndarray:
    """Resize frame keeping aspect ratio, target height = h."""
    oh, ow = frame.shape[:2]
    w = int(ow * h / oh)
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)


def add_label(frame: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    """Add a header bar with a label above the frame."""
    bar_h = 36
    bar   = np.zeros((bar_h, frame.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, 22), FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (8, bar_h - 4), FONT, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
    return np.concatenate([bar, frame], axis=0)


def stitch_comparison(
    left_frames,
    right_frames,
    out_path: str,
    fps: int,
    episode: int,
    seed: int,
):
    """Create side-by-side video. Pads the shorter sequence with its last frame."""
    T = max(len(left_frames), len(right_frames))

    # pad shorter side
    if len(left_frames)  < T: left_frames  += [left_frames[-1]]  * (T - len(left_frames))
    if len(right_frames) < T: right_frames += [right_frames[-1]] * (T - len(right_frames))

    # resize both to TARGET_H
    left_frames  = [resize_to_height(f, TARGET_H) for f in left_frames]
    right_frames = [resize_to_height(f, TARGET_H) for f in right_frames]

    # separator
    sep = np.full((TARGET_H + 36, 6, 3), 80, dtype=np.uint8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sample = left_frames[0]
    total_w = left_frames[0].shape[1] + 6 + right_frames[0].shape[1]
    total_h = TARGET_H + 36   # +36 for label bar

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (total_w, total_h))

    for t, (lf, rf) in enumerate(zip(left_frames, right_frames)):
        lf = add_label(
            lf,
            "replay_dataset  (stored pixels, NO simulation)",
            f"episode={episode}  t={t:02d}"
        )
        rf = add_label(
            rf,
            f"action_replay   (live SAPIEN sim, seed={seed})",
            f"episode={episode}  t={t:02d}  [scene may differ from original]"
        )
        frame = np.concatenate([lf, sep, rf], axis=1)
        writer.write(frame)

    writer.release()
    print(f"\n[3/3] Comparison video   →  {out_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run both replay methods and produce a side-by-side comparison video"
    )
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--seed",    type=int, default=0,
                   help="Seed for action_replay (original seed unknown)")
    p.add_argument("--fps",     type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    ep, seed, fps = args.episode, args.seed, args.fps

    print(f"=== Compare Replay  episode={ep}  seed={seed}  fps={fps} ===")

    # 1. stored-pixel replay → video
    left_video  = run_replay_dataset(episode=ep, fps=fps)

    # 2. action replay in sim → video
    right_video = run_action_replay(episode=ep, seed=seed, fps=fps)

    # 3. stitch
    out_path = os.path.join(
        VIDEO_DIR, "comparison",
        f"compare_ep{ep:04d}_seed{seed}.mp4"
    )
    left_frames  = read_video_frames(left_video)
    right_frames = read_video_frames(right_video)

    if not left_frames:
        print(f"ERROR: could not read {left_video}")
        return
    if not right_frames:
        print(f"ERROR: could not read {right_video}")
        return

    stitch_comparison(left_frames, right_frames, out_path, fps, ep, seed)

    print(f"\n=== Done ===")
    print(f"  Left  (stored pixels) : {left_video}")
    print(f"  Right (live sim)      : {right_video}")
    print(f"  Comparison            : {out_path}")
    print()
    print("Key differences to look for:")
    print("  • Left panel  — exact original episode, 128×128 sensor cameras (top+wrist)")
    print("  • Right panel — 512×512 human-render view, scene may differ (seed mismatch)")
    print("  • If seed matches original: robot motion AND scene should look identical")
    print("  • If seed differs:  robot motion is same, but object positions are different")


if __name__ == "__main__":
    main()
