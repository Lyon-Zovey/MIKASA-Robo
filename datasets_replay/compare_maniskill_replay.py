"""
compare_maniskill_replay.py
============================
Side-by-side video demonstrating the difference between:

  LEFT:  env_states replay  — at each step, forcibly set the environment
                               back to the exact recorded state.
                               ✓ Scene is IDENTICAL to original trajectory.

  RIGHT: action-only replay — play the same actions from a DIFFERENT
                               initial seed (scene randomised differently).
                               ✗ Scene diverges from original trajectory.

Usage:
    conda activate mikasa-robo
    cd ~/Sim_Data/MIKASA-Robo
    python3 compare_maniskill_replay.py

Output:
    videos/compare_maniskill_replay/compare_traj<N>.mp4
"""

import argparse
import os
import json

import cv2
import h5py
import numpy as np
import torch
import gymnasium as gym

import mikasa_robo_suite                        # registers all envs
import mani_skill.trajectory.utils as traj_utils
from mani_skill.utils import common

# ── Config ────────────────────────────────────────────────────────────────────

H5_FILE   = "/home/CNF2026716696/Sim_Data/MIKASA-Robo/videos/RememberColor9-v0/20260420_180005.h5"
JSON_FILE = H5_FILE.replace(".h5", ".json")
VIDEO_DIR = "/home/CNF2026716696/Sim_Data/MIKASA-Robo/videos/compare_maniskill_replay"
ENV_ID    = "RememberColor9-v0"
FPS       = 10
WRONG_SEED = 9999          # deliberately different seed for action-only replay

FONT      = cv2.FONT_HERSHEY_SIMPLEX


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_env(seed: int, render_mode="rgb_array"):
    env = gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        render_mode=render_mode,
        sim_backend="gpu",
        reward_mode="normalized_dense",
    )
    env.reset(seed=seed)
    return env


def get_frame(env) -> np.ndarray:
    """Render → (H, W, 3) BGR uint8."""
    img = env.render()
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    img = np.asarray(img, dtype=np.uint8)
    if img.ndim == 4:
        img = img[0]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def label_frame(frame: np.ndarray, text: str, color) -> np.ndarray:
    frame = frame.copy()
    cv2.putText(frame, text, (6, 20), FONT, 0.55, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (6, 20), FONT, 0.55, color,   1, cv2.LINE_AA)
    return frame


def side_by_side(left: np.ndarray, right: np.ndarray,
                 t: int, reward_l: float, reward_r: float) -> np.ndarray:
    sep  = np.zeros((left.shape[0], 6, 3), dtype=np.uint8)
    row  = np.concatenate([left, sep, right], axis=1)

    # header bar
    hdr = np.full((28, row.shape[1], 3), 30, dtype=np.uint8)
    lbl = f"step {t:02d} | env_states(exact): r={reward_l:.3f}   |   action+wrong_seed: r={reward_r:.3f}"
    cv2.putText(hdr, lbl, (8, 18), FONT, 0.42, (200,200,200), 1, cv2.LINE_AA)
    return np.concatenate([hdr, row], axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def replay_traj(traj_id: str, episode_seed: int):
    print(f"\n=== Replaying {traj_id}  (correct seed={episode_seed}, wrong seed={WRONG_SEED}) ===")

    with h5py.File(H5_FILE, "r") as f:
        traj      = f[traj_id]
        actions   = np.array(traj["actions"])        # (T, 8)
        rewards_h5= np.array(traj["rewards"])        # (T,)
        success_h5= np.array(traj["success"])        # (T,)
        env_states = traj_utils.dict_to_list_of_dicts(traj["env_states"])
        T = len(actions)

    # ── env A: env_states replay (correct seed + force set state) ─────────────
    env_a = make_env(episode_seed)
    # ── env B: action-only replay (WRONG seed) ─────────────────────────────────
    env_b = make_env(WRONG_SEED)

    # Set env_a to the recorded t=0 state
    env_a.unwrapped.set_state_dict(env_states[0])

    frames      = []
    rewards_a   = []
    rewards_b   = []
    successes_a = []
    successes_b = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for t in range(T):
        act_tensor = torch.tensor(actions[t], dtype=torch.float32,
                                  device=device).unsqueeze(0)

        # ── Step both envs ──
        _, rew_a, _, _, info_a = env_a.step(act_tensor)
        _, rew_b, _, _, info_b = env_b.step(act_tensor)

        # ── For env_a: forcibly restore recorded state AFTER step ──
        # (This is exactly what ManiSkill's --use-env-states flag does.)
        # We set state[t+1] so the observation matches the recorded trajectory.
        if t + 1 < len(env_states):
            env_a.unwrapped.set_state_dict(env_states[t + 1])

        ra = float(rew_a.item())
        rb = float(rew_b.item())
        sa = int(info_a.get("success", torch.zeros(1)).item())
        sb = int(info_b.get("success", torch.zeros(1)).item())

        rewards_a.append(ra);   rewards_b.append(rb)
        successes_a.append(sa); successes_b.append(sb)

        frame_a = get_frame(env_a)
        frame_b = get_frame(env_b)

        frame_a = label_frame(frame_a, f"env_states  r={ra:.3f} {'SUCCESS' if sa else ''}",
                               (80, 220, 80) if sa else (200, 200, 200))
        frame_b = label_frame(frame_b, f"action+wrong_seed  r={rb:.3f} {'SUCCESS' if sb else ''}",
                               (80, 220, 80) if sb else (80, 80, 220))

        frames.append(side_by_side(frame_a, frame_b, t, ra, rb))

        print(f"  t={t:2d} | env_states r={ra:.4f} s={sa} | action_only r={rb:.4f} s={sb} "
              f"| reward_diff={abs(ra-rb):.4f}")

    env_a.close()
    env_b.close()

    # ── Write video ────────────────────────────────────────────────────────────
    os.makedirs(VIDEO_DIR, exist_ok=True)
    out_path = os.path.join(VIDEO_DIR, f"compare_{traj_id}.mp4")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for fr in frames:
        writer.write(fr)
    writer.release()

    # ── Summary ────────────────────────────────────────────────────────────────
    reward_diffs = [abs(a - b) for a, b in zip(rewards_a, rewards_b)]
    print(f"\n  ─── Summary for {traj_id} ───")
    print(f"  env_states   total_reward={sum(rewards_a):.3f}  final_success={bool(successes_a[-1])}")
    print(f"  action_only  total_reward={sum(rewards_b):.3f}  final_success={bool(successes_b[-1])}")
    print(f"  reward diff  mean={np.mean(reward_diffs):.4f}  max={np.max(reward_diffs):.4f}")
    print(f"  video saved: {out_path}")
    return out_path


def main():
    global WRONG_SEED
    p = argparse.ArgumentParser()
    p.add_argument("--traj", type=str, default="traj_0",
                   help="Which trajectory to replay (traj_0 … traj_3)")
    p.add_argument("--wrong-seed", type=int, default=WRONG_SEED)
    args = p.parse_args()

    WRONG_SEED = args.wrong_seed

    with open(JSON_FILE) as f:
        meta = json.load(f)

    # find episode seed for requested traj
    ep_id     = int(args.traj.split("_")[-1])
    episode   = next(e for e in meta["episodes"] if e["episode_id"] == ep_id)
    seed      = episode["episode_seed"]

    print(f"env_id : {meta['env_info']['env_id']}")
    print(f"traj   : {args.traj}  correct_seed={seed}  wrong_seed={WRONG_SEED}")

    video = replay_traj(args.traj, seed)

    print(f"\n=== Done! Video: {video} ===")


if __name__ == "__main__":
    main()
