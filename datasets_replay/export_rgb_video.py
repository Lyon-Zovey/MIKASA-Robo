"""
Export RGB frames directly from a stored .npz episode to an mp4 video.
No simulation re-run needed — uses the rgb field recorded during collection.

Usage:
    python3 datasets_replay/export_rgb_video.py
    python3 datasets_replay/export_rgb_video.py --episode 3 --fps 15 --scale 4
    python3 datasets_replay/export_rgb_video.py --all --n 10
"""
import argparse, os
import numpy as np
import cv2

ENV_ID   = "ShellGameTouch-v0"
DATA_DIR = "data/MIKASA-Robo/unbatched/ShellGameTouch-v0"
VIDEO_DIR = "videos/rgb_export/ShellGameTouch-v0"


def overlay_stats(frame: np.ndarray, t: int, reward: float, success: int) -> np.ndarray:
    frame = frame.copy()
    color = (0, 220, 0) if success else (200, 200, 200)
    cv2.putText(frame, f"t={t:02d}  r={reward:.3f}", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    if success:
        cv2.putText(frame, "SUCCESS", (4, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1, cv2.LINE_AA)
    return frame


def export_episode(npz_path: str, video_dir: str, fps: int, scale: int) -> str:
    ep      = np.load(npz_path)
    rgb_all = ep['rgb']          # (T, H, W, 6)  uint8  — two cameras stacked on channel axis
    rewards = ep['reward']       # (T,)
    success = ep['success']      # (T,)
    T, H, W, C = rgb_all.shape

    # Split into two camera views (each 3-channel RGB)
    cam1 = rgb_all[:, :, :, :3]  # base camera
    cam2 = rgb_all[:, :, :, 3:]  # wrist camera

    os.makedirs(video_dir, exist_ok=True)
    ep_name = os.path.splitext(os.path.basename(npz_path))[0]
    out_path = os.path.join(video_dir, f"{ep_name}.mp4")

    # Side-by-side: [cam1 | cam2], scaled up
    out_w = W * 2 * scale
    out_h = H * scale
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    for t in range(T):
        f1 = cv2.resize(cam1[t], (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
        f2 = cv2.resize(cam2[t], (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
        frame = np.concatenate([f1, f2], axis=1)               # side-by-side
        frame = overlay_stats(frame, t, float(rewards[t]), int(success[t]))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    suc_steps = int(success.sum())
    print(f"  [{ep_name}]  T={T}  success_steps={suc_steps}  → {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Export stored rgb data to mp4")
    p.add_argument("--data_dir",  default=DATA_DIR)
    p.add_argument("--video_dir", default=VIDEO_DIR)
    p.add_argument("--episode",   type=int, default=0,
                   help="Episode index to export (ignored if --all)")
    p.add_argument("--all",       action="store_true",
                   help="Export all episodes in data_dir")
    p.add_argument("--n",         type=int, default=None,
                   help="With --all: export only the first N episodes")
    p.add_argument("--fps",       type=int, default=15)
    p.add_argument("--scale",     type=int, default=4,
                   help="Upscale factor (default 4 → 512×512 side-by-side)")
    args = p.parse_args()

    if args.all:
        files = sorted(
            [f for f in os.listdir(args.data_dir) if f.endswith('.npz')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        if args.n is not None:
            files = files[: args.n]
        print(f"Exporting {len(files)} episodes from {args.data_dir}")
        for fname in files:
            export_episode(os.path.join(args.data_dir, fname),
                           args.video_dir, args.fps, args.scale)
    else:
        npz = os.path.join(args.data_dir, f"train_data_{args.episode}.npz")
        if not os.path.exists(npz):
            print(f"ERROR: {npz} not found.")
            return
        print(f"Exporting episode {args.episode}")
        export_episode(npz, args.video_dir, args.fps, args.scale)

    print(f"\nVideos saved to: {args.video_dir}")


if __name__ == "__main__":
    main()
