"""
live_replay_compare.py
=======================
实时并排显示两种 replay 的差异（OpenCV 窗口，不存视频）：

  LEFT  : env_states replay — 每步强制 set_state_dict → 精确复现原轨迹
  RIGHT : action-only replay — 同样动作 + 错误 seed    → 场景不同，轨迹偏离

Usage:
    conda activate mikasa-robo
    cd ~/Sim_Data/MIKASA-Robo
    python3 live_replay_compare.py

    # 指定轨迹、错误 seed、播放速度
    python3 live_replay_compare.py --traj traj_1 --wrong-seed 12345 --delay 80
"""

import argparse
import json
import sys

import cv2
import h5py
import numpy as np
import torch
import gymnasium as gym

import mikasa_robo_suite                       # registers envs
import mani_skill.trajectory.utils as traj_utils

# ── Config ────────────────────────────────────────────────────────────────────
H5_FILE    = "/home/CNF2026716696/Sim_Data/MIKASA-Robo/videos/RememberColor9-v0/20260420_180005.h5"
JSON_FILE  = H5_FILE.replace(".h5", ".json")
ENV_ID     = "RememberColor9-v0"
DELAY_MS   = 120       # ms per frame (≈8 fps), press any key to skip
WRONG_SEED = 9999
FONT       = cv2.FONT_HERSHEY_SIMPLEX
WIN_NAME   = "ManiSkill Replay Compare  |  LEFT: env_states(exact)   RIGHT: action+wrong_seed"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_env(seed: int):
    env = gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        render_mode="all",
        sim_backend="gpu",
        reward_mode="normalized_dense",
    )
    env.reset(seed=seed)
    return env


def get_frame(env) -> np.ndarray:
    """env.render() → (H,W,3) BGR uint8"""
    img = env.render()
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    img = np.asarray(img, dtype=np.uint8)
    if img.ndim == 4:
        img = img[0]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def annotate(frame: np.ndarray, lines: list, ok: bool) -> np.ndarray:
    frame = frame.copy()
    color_ok  = (60, 220, 60)
    color_bad = (60, 60, 220)
    color_txt = (230, 230, 230)
    y = 18
    for i, line in enumerate(lines):
        col = (color_ok if ok else color_bad) if i == len(lines)-1 else color_txt
        cv2.putText(frame, line, (5, y + i*17), FONT, 0.45, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (5, y + i*17), FONT, 0.45, col,    1, cv2.LINE_AA)
    return frame


def compose(left, right, t, ra, rb, sa, sb) -> np.ndarray:
    h = left.shape[0]
    sep  = np.full((h, 8, 3), 50, dtype=np.uint8)
    row  = np.concatenate([left, sep, right], axis=1)

    hdr_h = 32
    hdr   = np.full((hdr_h, row.shape[1], 3), 25, dtype=np.uint8)
    diff  = abs(ra - rb)
    label = (f"  Step {t:02d}  |  "
             f"env_states r={ra:.3f} {'✓SUCCESS' if sa else ''}   "
             f"vs   action+wrong_seed r={rb:.3f} {'✓SUCCESS' if sb else ''}   "
             f"|  reward_diff={diff:.3f}")
    cv2.putText(hdr, label, (6, 22), FONT, 0.45, (180,180,180), 1, cv2.LINE_AA)

    # color the header border to highlight divergence
    border_col = (0, int(255*(1-min(diff*5, 1))), int(255*min(diff*5, 1)))
    cv2.line(hdr, (0, hdr_h-1), (hdr.shape[1], hdr_h-1), border_col, 2)

    return np.concatenate([hdr, row], axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj",       default="traj_0",
                   help="traj_0 … traj_3")
    p.add_argument("--wrong-seed", type=int, default=WRONG_SEED)
    p.add_argument("--delay",      type=int, default=DELAY_MS,
                   help="ms per frame (0 = wait for keypress)")
    args = p.parse_args()

    # ── Load trajectory data ──────────────────────────────────────────────────
    with open(JSON_FILE) as f:
        meta = json.load(f)
    ep_id   = int(args.traj.split("_")[-1])
    episode = next(e for e in meta["episodes"] if e["episode_id"] == ep_id)
    seed    = episode["episode_seed"]

    print(f"env      : {ENV_ID}")
    print(f"traj     : {args.traj}")
    print(f"correct seed = {seed}   wrong seed = {args.wrong_seed}")

    with h5py.File(H5_FILE, "r") as f:
        traj       = f[args.traj]
        actions    = np.array(traj["actions"])     # (T, 8)
        env_states = traj_utils.dict_to_list_of_dicts(traj["env_states"])
        T          = len(actions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build two environments ────────────────────────────────────────────────
    print("Building env_A (correct seed + env_states replay) …")
    env_a = make_env(seed)
    env_a.unwrapped.set_state_dict(env_states[0])   # set to recorded t=0

    print(f"Building env_B (wrong seed={args.wrong_seed}, action-only replay) …")
    env_b = make_env(args.wrong_seed)

    # ── OpenCV window ─────────────────────────────────────────────────────────
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1100, 520)

    reward_diffs = []

    print(f"\nRunning {T} steps  (press Q or ESC to quit early) …\n")
    for t in range(T):
        act = torch.tensor(actions[t], dtype=torch.float32, device=device).unsqueeze(0)

        _, ra, _, _, info_a = env_a.step(act)
        _, rb, _, _, info_b = env_b.step(act)

        # env_states replay: force recorded state
        if t + 1 < len(env_states):
            env_a.unwrapped.set_state_dict(env_states[t + 1])

        ra = float(ra.item())
        rb = float(rb.item())
        sa = int(info_a.get("success", torch.zeros(1)).item())
        sb = int(info_b.get("success", torch.zeros(1)).item())
        reward_diffs.append(abs(ra - rb))

        frame_a = get_frame(env_a)
        frame_b = get_frame(env_b)

        frame_a = annotate(frame_a,
                           ["LEFT: env_states replay",
                            f"seed={seed}  (correct)",
                            f"r={ra:.4f}",
                            "SUCCESS" if sa else ""],
                           ok=True)
        frame_b = annotate(frame_b,
                           ["RIGHT: action-only replay",
                            f"seed={args.wrong_seed}  (WRONG)",
                            f"r={rb:.4f}",
                            "SUCCESS" if sb else ""],
                           ok=False)

        combined = compose(frame_a, frame_b, t, ra, rb, sa, sb)
        cv2.imshow(WIN_NAME, combined)

        print(f"  t={t:2d} | exact r={ra:.4f} s={sa} | action r={rb:.4f} s={sb} | diff={abs(ra-rb):.4f}")

        key = cv2.waitKey(args.delay) & 0xFF
        if key in (ord('q'), ord('Q'), 27):   # Q or ESC
            print("\n[用户中止]")
            break

    env_a.close()
    env_b.close()
    cv2.destroyAllWindows()

    if reward_diffs:
        print(f"\n=== 汇总 ===")
        print(f"  reward_diff  mean={np.mean(reward_diffs):.4f}  max={np.max(reward_diffs):.4f}")
        print("  → 数值差距越大，说明 action-only replay 和原轨迹偏离越严重")


if __name__ == "__main__":
    main()
