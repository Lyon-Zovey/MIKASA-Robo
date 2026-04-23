#!/usr/bin/env bash
# ============================================================================
# Step 3 of the MIKASA-Robo two-stage data pipeline (visualization / replay).
#
# Opens a SAPIEN viewer that re-runs the oracle policy with the same seed as
# a previously collected episode, while overlaying the recorded scene-point
# flow on top of the live 3D scene. Designed to inspect data produced by:
#   Step 1: collect_gpu_demos.py   (state-only GPU rollout, success-filtered)
#   Step 2: run_mikasa_demos.sh    (CPU rbs_replay -> RGB+Depth+Seg+Flow)
#
# What it does:
#   1) Reads <demos_dir>/trajectory.<obs>.<ctrl>.<backend>.json to look up the
#      episode_seed corresponding to the chosen TRAJ_ID.
#   2) Decompresses (only if needed):
#        seg.b2nd                                  -> seg.npy
#        scene_point_flow_ref<ANCHOR>.anchor.npy
#        + scene_point_flow_ref<ANCHOR>_*.mp4      -> scene_point_flow_ref<ANCHOR>.npy
#      because get_mikasa_robo_datasets.py --visualize loads the raw .npy form.
#   3) Launches the SAPIEN viewer via:
#        python -m mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets \
#               --visualize --pointflow-npy=<...>
#
# Usage:
#   bash run_scripts/replay_one.sh
#
# Override any tunable on the command line:
#   ENV_ID=ShellGamePush-v0 TRAJ_ID=traj_12 ANCHOR=00045 \
#       bash run_scripts/replay_one.sh
# ============================================================================
set -euo pipefail

# ─── 你只需要改这 3 个 ────────────────────────────────────────────────────────
ENV_ID="${ENV_ID:-ShellGameTouch-v0}"   # 任务名（需在 mikasa_robo_suite.memory_envs 注册）
TRAJ_ID="${TRAJ_ID:-traj_0}"            # camera_data/ 下的目录名
ANCHOR="${ANCHOR:-00022}"               # pointflow 参考帧（Step 2 默认 5 个：00000/00022/00045/00068/00090）

# ─── 一般不用动（除非你换了数据落盘根目录或仓库位置）──────────────────────────
DATA_ROOT="${DATA_ROOT:-/home/CNF2026716696/Sim_Data/MIKASA_Data}"
MIKASA_ROOT="${MIKASA_ROOT:-/home/CNF2026716696/Sim_Data/MIKASA-Robo}"
MANISKILL_ROOT="${MANISKILL_ROOT:-/home/CNF2026716696/Sim_Data/ManiSkill}"
CKPT_DIR="${CKPT_DIR:-${MIKASA_ROOT}}"  # contains oracle_checkpoints/...

# ─── pointflow 视觉调味料 ────────────────────────────────────────────────────
PF_STRIDE="${PF_STRIDE:-2}"             # 1=最密 ~23k点/帧, 2=约5.7k(推荐), 4=约1.4k
PF_COLOR="${PF_COLOR:-id}"              # id / z / rgb
PF_EXCLUDE=("${PF_EXCLUDE[@]:-actor:table scene-builder actor:ground ground-plane}")
VIEWER_CAMERA="${VIEWER_CAMERA:-render_camera}"   # render_camera / base_camera / hand_camera

# ─── 让动画看得到 + 让点云贴齐物体（关键！）──────────────────────────────────
VIS_STEP_DELAY="${VIS_STEP_DELAY:-0.10}"        # 秒/步。0=全速(5秒就结束)，0.10≈10fps
VIS_LOOP_POINTFLOW="${VIS_LOOP_POINTFLOW:--1}"  # episode 跑完后循环回放 pointflow，-1=一直循环到关窗
USE_RECORDED_STATES="${USE_RECORDED_STATES:-1}" # 1=强烈推荐：viewer 每步 set_state_dict() 到 H5 录的状态
                                                #   ⇒ 现场物理 == pointflow 生成时的物理 ⇒ 点严格贴在物体上
                                                # 0=只用 seed 跑 fresh policy（pointflow 会和现场漂开）

# ─── 内部计算 ────────────────────────────────────────────────────────────────
DEMO_DIR="${DATA_ROOT}/MIKASA-Robo/demos/${ENV_ID}"
TRAJ_DIR="${DEMO_DIR}/camera_data/${TRAJ_ID}"

if [ ! -d "${TRAJ_DIR}" ]; then
    echo "[ERROR] traj dir not found: ${TRAJ_DIR}" >&2
    echo "[hint]  ls ${DEMO_DIR}/camera_data/ | head" >&2
    exit 1
fi

H5_PATH=$(ls "${DEMO_DIR}"/trajectory.*.h5 2>/dev/null | grep -v physx_cuda | head -1)
if [ -z "${H5_PATH}" ]; then
    H5_PATH=$(ls "${DEMO_DIR}"/trajectory.*.h5 | head -1)
fi
JSON_PATH="${H5_PATH%.h5}.json"
echo "[INFO] env_id    : ${ENV_ID}"
echo "[INFO] traj_id   : ${TRAJ_ID}"
echo "[INFO] anchor    : ${ANCHOR}"
echo "[INFO] traj_dir  : ${TRAJ_DIR}"
echo "[INFO] json      : ${JSON_PATH}"

source /home/CNF2026716696/miniconda3/etc/profile.d/conda.sh
conda activate mikasa-robo
export PYTHONPATH="${MIKASA_ROOT}:${MANISKILL_ROOT}:${PYTHONPATH:-}"
export DISPLAY="${DISPLAY:-:1}"

# (1) 查 episode_seed —— 必须和 traj 对齐，否则策略跑到完全不同的初始场景
TRAJ_NUM="${TRAJ_ID#traj_}"             # traj_5 -> 5    或 traj_5_p123_1_abc -> 5_p123_1_abc
TRAJ_NUM="${TRAJ_NUM%%_*}"              # 5_p123_1_abc -> 5
SEED=$(python - "${JSON_PATH}" "${TRAJ_NUM}" <<'PYEOF'
import json, sys
js, traj_num = sys.argv[1], int(sys.argv[2])
data = json.load(open(js))
ep = next((e for e in data["episodes"] if int(e["episode_id"]) == traj_num), None)
if ep is None:
    raise SystemExit(f"episode_id={traj_num} not found in {js}")
print(ep["episode_seed"])
PYEOF
)
echo "[INFO] episode_seed = ${SEED}  (looked up from ${JSON_PATH})"

# (2) seg.b2nd -> seg.npy （已存在则跳过）
if [ -f "${TRAJ_DIR}/seg.npy" ]; then
    echo "[SKIP] seg.npy already exists"
else
    python - "${TRAJ_DIR}" <<'PYEOF'
import sys
from pathlib import Path
from mikasa_robo_suite.dataset_collectors.rbs_record.seg_compress import decompress_seg
td = Path(sys.argv[1])
b2nd = td / "seg.b2nd"
out = td / "seg.npy"
if not b2nd.exists():
    raise SystemExit(f"missing {b2nd}; was Step 2 run with --postprocess-camera-data?")
seg = decompress_seg(b2nd, out)
print(f"[OK] seg.npy  shape={seg.shape}  dtype={seg.dtype}  size={out.stat().st_size/1024**2:.1f} MB")
PYEOF
fi

# (3) anchor + mp4 -> scene_point_flow_ref<ANCHOR>.npy （已存在则跳过）
PF_NPY="${TRAJ_DIR}/scene_point_flow_ref${ANCHOR}.npy"
if [ -f "${PF_NPY}" ]; then
    echo "[SKIP] ${PF_NPY##*/} already exists"
else
    python - "${TRAJ_DIR}" "${ANCHOR}" <<'PYEOF'
import sys, numpy as np
from pathlib import Path
from mikasa_robo_suite.dataset_collectors.rbs_record.flow_compress import decompress_one_flow
td = Path(sys.argv[1])
a  = sys.argv[2]
anchor_path = td / f"scene_point_flow_ref{a}.anchor.npy"
mp4s = sorted(td.glob(f"scene_point_flow_ref{a}_*.mp4"))
if not anchor_path.exists() or not mp4s:
    raise SystemExit(f"missing anchor/mp4 for ref{a} in {td}")
anchor = np.load(anchor_path)
flow = decompress_one_flow(mp4s[0], anchor)
out = td / f"scene_point_flow_ref{a}.npy"
np.save(out, flow.astype(np.float32))
print(f"[OK] {out.name}  shape={flow.shape}  dtype=float32  size={out.stat().st_size/1024**2:.1f} MB")
PYEOF
fi

# (4) 启动 SAPIEN viewer + pointflow overlay
EXTRA_ARGS=()
if [ "${USE_RECORDED_STATES}" = "1" ]; then
    EXTRA_ARGS+=(--replay-h5 "${H5_PATH}" --replay-traj-id "${TRAJ_ID}")
    echo "[INFO] tight-tracking mode ON: viewer will set_state_dict() each step from H5"
else
    echo "[INFO] tight-tracking mode OFF: viewer runs fresh policy with seed=${SEED}"
fi

echo
echo "[INFO] launching SAPIEN viewer (DISPLAY=${DISPLAY})  -- close the window to exit (loop is ON by default)"
cd "${MIKASA_ROOT}"
python -u -m mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets \
    --env_id="${ENV_ID}" \
    --ckpt-dir "${CKPT_DIR}" \
    --visualize \
    --no-postprocess-camera-data \
    --pointflow-npy="${PF_NPY}" \
    --pointflow-exclude-names ${PF_EXCLUDE[@]} \
    --pointflow-stride="${PF_STRIDE}" \
    --pointflow-color-mode="${PF_COLOR}" \
    --viewer-camera "${VIEWER_CAMERA}" \
    --seed "${SEED}" \
    --count 1 \
    --path-to-save-data "${DATA_ROOT}" \
    --vis-step-delay "${VIS_STEP_DELAY}" \
    --vis-loop-pointflow "${VIS_LOOP_POINTFLOW}" \
    "${EXTRA_ARGS[@]}"

echo
echo "[DONE] vis output dir: ${DATA_ROOT}/MIKASA-Robo/vis/${ENV_ID}/"
