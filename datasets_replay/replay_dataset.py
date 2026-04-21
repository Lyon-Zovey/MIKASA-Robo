"""
Dataset Replay Script for MIKASA-Robo
======================================
Directly reads .npz episodes (no simulation needed) and:
  1. Saves a side-by-side video (top camera | wrist camera + reward/action overlay)
  2. Prints per-step stats (reward, success, action norm)

Usage:
    conda activate mikasa-robo
    python3 replay_dataset.py                          # replay episode 0 with defaults
    python3 replay_dataset.py --episode 5              # replay episode 5
    python3 replay_dataset.py --episode 3 --fps 10     # slower playback
    python3 replay_dataset.py --all --max_episodes 10  # batch-render first 10 episodes
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR  = "/home/CNF2026716696/Sim_Data/MIKASA_Data/RememberShape3-v0"
VIDEO_DIR = "/home/CNF2026716696/Sim_Data/MIKASA-Robo/videos/replay_RememberShape3-v0"
FPS_DEFAULT = 15
FONT  = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 0.38
THICKNESS = 1


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_episode(path: str) -> dict:
    ep = np.load(path)
    return {k: ep[k] for k in ep.keys()}


def draw_overlay(frame: np.ndarray, t: int, reward: float,
                 success: int, action: np.ndarray) -> np.ndarray:
    """Overlay step info on a BGR frame (in-place copy)."""
    frame = frame.copy()
    color_ok  = (80, 220, 80)
    color_bad = (80, 80, 220)
    color_txt = (240, 240, 240)

    lines = [
        f"step   : {t:3d}",
        f"reward : {reward:.3f}",
        f"success: {'YES' if success else 'no'}",
        f"act_l2 : {float(np.linalg.norm(action)):.3f}",
    ]
    y0 = 14
    for i, line in enumerate(lines):
        col = color_ok if (i == 2 and success) else color_txt
        cv2.putText(frame, line, (4, y0 + i * 14),
                    FONT, SCALE, (0, 0, 0), THICKNESS + 1, cv2.LINE_AA)
        cv2.putText(frame, line, (4, y0 + i * 14),
                    FONT, SCALE, col, THICKNESS, cv2.LINE_AA)
    return frame


def make_side_by_side(top: np.ndarray, wrist: np.ndarray,
                      t: int, reward: float, success: int,
                      action: np.ndarray) -> np.ndarray:
    """
    top / wrist: RGB uint8 (H, W, 3)
    Returns a BGR frame ready for VideoWriter.
    """
    top_bgr   = cv2.cvtColor(top,   cv2.COLOR_RGB2BGR)
    wrist_bgr = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)

    # no upscaling — output at original 128×128
    top_bgr   = draw_overlay(top_bgr,   t, reward, success, action)
    wrist_bgr = draw_overlay(wrist_bgr, t, reward, success, action)

    # add thin separator
    sep = np.zeros((128, 4, 3), dtype=np.uint8)
    frame = np.concatenate([top_bgr, sep, wrist_bgr], axis=1)

    # header bar with camera labels
    header = np.zeros((20, frame.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, "Top Camera",   (30,  14), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(header, "Wrist Camera", (162, 14), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    frame = np.concatenate([header, frame], axis=0)

    return frame


def replay_episode(npz_path: str, out_video: str, fps: int = FPS_DEFAULT,
                   verbose: bool = True) -> dict:
    ep = load_episode(npz_path)
    T  = ep['rgb'].shape[0]

    frames_top   = ep['rgb'][:, :, :, :3]   # (T, 128, 128, 3)
    frames_wrist = ep['rgb'][:, :, :, 3:]   # (T, 128, 128, 3)
    rewards  = ep['reward']                  # (T,)
    successes = ep['success']                # (T,)
    actions  = ep['action']                  # (T, 8)

    # VideoWriter setup (260 wide = 128 + 4 + 128, 148 tall = 20 + 128)
    os.makedirs(os.path.dirname(out_video), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video, fourcc, fps, (260, 148))

    stats = {"total_reward": 0.0, "success_steps": 0, "final_success": False}

    for t in range(T):
        frame = make_side_by_side(
            frames_top[t], frames_wrist[t],
            t, rewards[t], int(successes[t]), actions[t]
        )
        writer.write(frame)

        stats["total_reward"]  += float(rewards[t])
        stats["success_steps"] += int(successes[t])

        if verbose:
            print(f"  t={t:2d}  reward={rewards[t]:.4f}  "
                  f"success={int(successes[t])}  "
                  f"act_norm={np.linalg.norm(actions[t]):.4f}")

    stats["final_success"] = bool(successes[-1])
    writer.release()
    return stats


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MIKASA-Robo dataset replay (no simulation)")
    p.add_argument("--data_dir",   default=DATA_DIR)
    p.add_argument("--video_dir",  default=VIDEO_DIR)
    p.add_argument("--episode",    type=int, default=0,
                   help="Episode index to replay (default: 0)")
    p.add_argument("--fps",        type=int, default=FPS_DEFAULT)
    p.add_argument("--all",        action="store_true",
                   help="Render all episodes (or up to --max_episodes)")
    p.add_argument("--max_episodes", type=int, default=None,
                   help="Max episodes to process when --all is set")
    p.add_argument("--quiet",      action="store_true",
                   help="Suppress per-step output")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir  = Path(args.data_dir)
    video_dir = Path(args.video_dir)

    if args.all:
        files = sorted(data_dir.glob("train_data_*.npz"),
                       key=lambda p: int(p.stem.split("_")[-1]))
        if args.max_episodes:
            files = files[:args.max_episodes]
        print(f"Batch replay: {len(files)} episodes → {video_dir}")
        all_stats = []
        for f in files:
            ep_idx = int(f.stem.split("_")[-1])
            out    = video_dir / f"episode_{ep_idx:04d}.mp4"
            print(f"\n[Episode {ep_idx}]  {f.name}")
            stats = replay_episode(str(f), str(out), args.fps,
                                   verbose=not args.quiet)
            print(f"  → total_reward={stats['total_reward']:.4f}  "
                  f"success_steps={stats['success_steps']}  "
                  f"final_success={stats['final_success']}")
            all_stats.append(stats)

        n_success = sum(s["final_success"] for s in all_stats)
        print(f"\n=== Summary: {n_success}/{len(all_stats)} episodes ended with success ===")

    else:
        ep_idx = args.episode
        npz    = data_dir / f"train_data_{ep_idx}.npz"
        if not npz.exists():
            print(f"ERROR: {npz} not found.")
            return
        out = video_dir / f"episode_{ep_idx:04d}.mp4"
        print(f"Replaying episode {ep_idx}  →  {out}")
        stats = replay_episode(str(npz), str(out), args.fps,
                               verbose=not args.quiet)
        print(f"\n=== Episode {ep_idx} done ===")
        print(f"  total_reward  : {stats['total_reward']:.4f}")
        print(f"  success_steps : {stats['success_steps']} / 60")
        print(f"  final_success : {stats['final_success']}")
        print(f"  video saved   : {out}")


if __name__ == "__main__":
    main()
