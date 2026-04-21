"""
Action Replay Script for MIKASA-Robo
======================================
Loads stored actions from a .npz episode, feeds them back into the live
SAPIEN simulation, and records the resulting video from the human-render
camera (512×512).

Since the original seed is not stored in the .npz files, the initial scene
configuration will differ from the original episode — the robot executes the
same motion sequence but in a (potentially different) randomised scene.

Usage:
    conda activate mikasa-robo
    cd ~/Sim_Data/MIKASA-Robo

    # replay episode 0 with a random seed
    python3 action_replay.py

    # specify episode index and seed
    python3 action_replay.py --episode 3 --seed 42

    # change task / dataset path
    python3 action_replay.py --env_id RememberColor3-v0 \\
        --data_dir /home/.../RememberColor3-v0 --episode 0
"""

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch

import cv2

import mikasa_robo_suite                                          # registers all envs
from mikasa_robo_suite.utils.wrappers import StateOnlyTensorToDictWrapper


# ─── Defaults ────────────────────────────────────────────────────────────────

ENV_ID    = "RememberShape3-v0"
DATA_DIR  = "/home/CNF2026716696/Sim_Data/MIKASA_Data/RememberShape3-v0"
VIDEO_DIR = "/home/CNF2026716696/Sim_Data/MIKASA-Robo/videos/action_replay_RememberShape3-v0"
EPISODE_TIMEOUT = 60          # RememberShape3-v0 episode length


# ─── Core ────────────────────────────────────────────────────────────────────

def run_action_replay(env_id: str, npz_path: str, video_dir: str,
                      seed: int, fps: int = 15, gui: bool = False) -> dict:

    # 1. Load stored actions
    ep = np.load(npz_path)
    actions_np = ep['action']          # (T, 8)  float32
    T = actions_np.shape[0]
    # Use the seed stored in the file for exact scene reproduction.
    # Fall back to the caller-supplied seed if the file pre-dates this feature.
    if 'seed' in ep:
        seed = int(ep['seed'][0])
        print(f"  Loaded {T} actions from {os.path.basename(npz_path)}  (seed={seed} from file)")
    else:
        print(f"  Loaded {T} actions from {os.path.basename(npz_path)}  "
              f"(no seed in file — using caller seed={seed}, scene may differ)")

    # 2. Build live simulation environment
    #    ⚠️ SAPIEN 3.0: render_mode="human" + obs_mode="rgb" conflict in GPU batched mode.
    #    GUI mode uses obs_mode="state" to avoid this.
    if gui:
        render_mode = "human"
        obs_mode    = "state"
    else:
        render_mode = "all"
        obs_mode    = "rgb"

    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode=obs_mode,
        control_mode="pd_joint_delta_pos",
        render_mode=render_mode,
        sim_backend="gpu",
        reward_mode="normalized_dense",
    )
    env = StateOnlyTensorToDictWrapper(env)

    # 3. Prepare video writer (fixed output path, no RecordEpisode)
    ep_name  = os.path.splitext(os.path.basename(npz_path))[0]   # e.g. train_data_0
    out_path = os.path.join(video_dir, f"{ep_name}.mp4")
    writer   = None
    if not gui:
        os.makedirs(video_dir, exist_ok=True)

    # 4. Reset then restore exact initial physics state from the stored state vector.
    #    Two reasons seed alone is unreliable:
    #    (a) ManiSkill seeds torch.rand with _episode_seed[0] (always the first env's seed
    #        in a batched env), so per-env positions diverge across batched/single-env modes.
    #    (b) evaluate() (called inside reset()) moves cups to z+HEIGHT_OFFSET=1000 on the
    #        CPU buffer only; the GPU physics state has cups at z~0.04.  The collection
    #        script calls _gpu_fetch_all() before get_state() to capture the true GPU
    #        physics state.  set_state() restores that state (z~0.04) into BOTH the GPU
    #        and CPU, giving exact physics reproducibility without free-fall artifacts.
    obs, info = env.reset(seed=seed)
    if 'init_state' in ep:
        _dev = env.unwrapped.device
        init_state = torch.tensor(ep['init_state'], dtype=torch.float32, device=_dev)
        env.unwrapped.set_state(init_state)

        # Restore Python-level task attributes that set_state() does not touch.
        # oracle_info is the authoritative task state (e.g. cup_with_ball_number for
        # ShellGame, color index for RememberColor, etc.).  We set both the generic
        # oracle_info attribute and any task-specific alias so reward / evaluate()
        # compute against the correct target.
        if 'init_oracle_info' in ep:
            # Keep shape (num_envs, ...) — do NOT squeeze to scalar or flatten_state_dict will error
            oracle_val = torch.tensor(ep['init_oracle_info'], dtype=torch.float32, device=_dev).reshape(1, -1).squeeze(-1)
            env.unwrapped.oracle_info = oracle_val
            # Sync task-specific aliases
            base = env.unwrapped
            if hasattr(base, 'cup_with_ball_number'):   # ShellGame family
                base.cup_with_ball_number = oracle_val.to(torch.uint8)
            if hasattr(base, 'target_color'):            # RememberColor family
                base.target_color = oracle_val
            if hasattr(base, 'target_shape'):            # RememberShape family
                base.target_shape = oracle_val

        print(f"  Env reset with seed={seed}  (physics + task state restored — exact replay)")
    else:
        print(f"  Env reset with seed={seed}  (no init_state in file — scene may differ)")
    if gui:
        print("  GUI window should now be open. Close it to stop.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5. Step through with stored actions, capturing frames manually
    stats = {"total_reward": 0.0, "success_steps": 0}
    for t in range(T):
        action = torch.from_numpy(actions_np[t]).unsqueeze(0).to(device)  # (1, 8)
        obs, reward, terminated, truncated, info = env.step(action)

        if gui:
            env.render()
            time.sleep(1.0 / fps)
        else:
            # env.render() returns the human-render-camera frame (RGB, uint8, HxWx3)
            frame_rgb = env.render()                         # torch tensor or np array
            if hasattr(frame_rgb, "cpu"):
                frame_rgb = frame_rgb.cpu().numpy()
            frame_rgb = np.asarray(frame_rgb, dtype=np.uint8)
            if frame_rgb.ndim == 4:                          # (1, H, W, 3) → (H, W, 3)
                frame_rgb = frame_rgb[0]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            h, w = frame_bgr.shape[:2]
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            writer.write(frame_bgr)

        r = float(reward.item())
        s = int(info.get("success", torch.zeros(1)).item())
        stats["total_reward"]  += r
        stats["success_steps"] += s
        print(f"  t={t:2d}  reward={r:.4f}  success={s}")

    stats["final_success"] = bool(
        info.get("success", torch.zeros(1)).item()
    )
    if writer is not None:
        writer.release()
        stats["video_path"] = out_path
    env.close()
    return stats


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MIKASA-Robo action replay — feeds stored actions into live sim"
    )
    p.add_argument("--env_id",   default=ENV_ID)
    p.add_argument("--data_dir", default=DATA_DIR)
    p.add_argument("--video_dir", default=VIDEO_DIR)
    p.add_argument("--episode",  type=int, default=0,
                   help="Episode index (selects train_data_<N>.npz)")
    p.add_argument("--seed",     type=int, default=0,
                   help="Seed for env.reset() (original seed is unknown)")
    p.add_argument("--fps",      type=int, default=15)
    p.add_argument("--gui",      action="store_true",
                   help="Open interactive SAPIEN viewer (requires $DISPLAY / desktop)")
    return p.parse_args()


def main():
    args = parse_args()

    npz_path = os.path.join(args.data_dir, f"train_data_{args.episode}.npz")
    if not os.path.exists(npz_path):
        print(f"ERROR: {npz_path} not found.")
        return

    print(f"\n=== Action Replay ===")
    print(f"  env    : {args.env_id}")
    print(f"  episode: {args.episode}  ({npz_path})")
    print(f"  seed   : {args.seed}")
    print(f"  output : {args.video_dir}\n")

    stats = run_action_replay(
        env_id=args.env_id,
        npz_path=npz_path,
        video_dir=args.video_dir,
        seed=args.seed,
        fps=args.fps,
        gui=args.gui,
    )

    print(f"\n=== Done ===")
    print(f"  total_reward  : {stats['total_reward']:.4f}")
    print(f"  success_steps : {stats['success_steps']}")
    print(f"  final_success : {stats['final_success']}")
    if "video_path" in stats:
        print(f"  video saved   : {stats['video_path']}")


if __name__ == "__main__":
    main()
