#!/usr/bin/env bash
# ============================================================================
# Step-1 + Step-2 fused GPU batched collector for MIKASA-Robo
# ============================================================================
# Drives `get_mikasa_robo_datasets.py --gpu-batched`. Replaces the old
#   Step-1 (collect_gpu_demos.py)  +  Step-2 (run_mikasa_demos.sh)
# pipeline with a single GPU process that:
#   - rolls out the oracle policy on `num_envs` parallel GPU envs,
#   - records rgb / depth / seg / cam_poses / id_poses via RBSRecordEpisode,
#   - runs the postprocess chain (scene_point_flow_ref*.npy + .b2nd + ...),
#   - skips the SAPIEN viewer entirely (no display required).
#
# Multi-GPU: this script auto-detects `CUDA_VISIBLE_DEVICES` (or queries
# `nvidia-smi -L` if unset) and launches ONE Python subprocess per GPU,
# splitting `NUM_EPISODES` evenly and giving each subprocess a different
# `--seed` so the seed ranges don't overlap. Each subprocess writes into
# the same `save_dir`; trajectory .h5 / camera_data/ collisions are handled
# by RBSRecordEpisode (per-pid filename suffix + `_p<pid>_<retry>_<hash>`
# camera_data dirnames).
#
# Multi-process per single GPU?  (PROCS_PER_GPU > 1)
#   YES, useful in practice. RBSRecord blocks every step on GPU->CPU sync
#   for obs/seg/depth/pose buffers, so a single process leaves the SMs idle
#   ~80-90% of the time. Stacking K processes on the same device makes their
#   render bursts overlap each other's CPU work and saturates the GPU.
#   Trade-off: each process loads its own PhysX-GPU + renderer + weights
#   (~2-3GB VRAM at NUM_ENVS=8). Pick K so K*VRAM ≤ total - 1GB(desktop).
#
# Usage:
#   bash run_scripts/collect_gpu_batched.sh
#
# Override any tunable on the command line:
#   ENV_ID=ShellGamePush-v0 NUM_EPISODES=200 NUM_ENVS=4 \
#       bash run_scripts/collect_gpu_batched.sh
#
# Force single-GPU even with multi-GPU machine:
#   CUDA_VISIBLE_DEVICES=0 bash run_scripts/collect_gpu_batched.sh
#
# Saturate a single 8GB GPU with 2 parallel pipelines:
#   PROCS_PER_GPU=2 NUM_ENVS=8 NUM_EPISODES=64 \
#       bash run_scripts/collect_gpu_batched.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ─── tweakables ──────────────────────────────────────────────────────────────
ENV_ID="${ENV_ID:-ShellGameTouch-v0}"
NUM_EPISODES="${NUM_EPISODES:-64}"          # total across all GPUs
NUM_ENVS="${NUM_ENVS:-16}"                  # GPU batch size per process
SEED_START="${SEED_START:-0}"
SENSOR_W="${SENSOR_W:-832}"
SENSOR_H="${SENSOR_H:-480}"
FPS="${FPS:-16}"
RECORD_ID_POSES="${RECORD_ID_POSES:-1}"     # 1=on, 0=off
SAVE_VIDEO="${SAVE_VIDEO:-1}"               # 1=on, 0=off
POSTPROCESS="${POSTPROCESS:-1}"             # 1=on, 0=off
DELETE_NPY="${DELETE_NPY:-1}"               # 1=delete raw .npy after compress
# Postprocess thread pool size: how many trajs can be compressed in parallel.
# Default: min(num_envs, nproc/3). Each running flow_compress.py spawns ffmpeg
# with ~4 threads, so we cap at nproc/3 to avoid CPU oversubscription.
_NPROC=$(nproc 2>/dev/null || echo 8)
POSTPROCESS_WORKERS="${POSTPROCESS_WORKERS:-$(( NUM_ENVS < (_NPROC / 3) ? NUM_ENVS : (_NPROC / 3) ))}"
[[ "${POSTPROCESS_WORKERS}" -lt 1 ]] && POSTPROCESS_WORKERS=1
# Per-traj subprocess parallelism: run convert/flow/point/seg in parallel
# within a single traj's postprocess slot (they touch different files).
PER_TRAJ_PARALLEL="${PER_TRAJ_PARALLEL:-1}"  # 1=on, 0=off
# Skip flow_compress.py (H.265 mp4 encoding for sceneflow). This is the
# slowest post-processing step. With SKIP_FLOW_MP4=1 (default), each traj
# directory keeps only rgb.mp4 (no scene_point_flow_*_v3_10b_h265_crf0.mp4),
# the raw scene_point_flow_ref*.npy is deleted (when DELETE_NPY=1), and the
# small .anchor.npy files are preserved. Set =0 if you actually need the
# tracked flow as mp4.
SKIP_FLOW_MP4="${SKIP_FLOW_MP4:-1}"  # 1=skip (faster, less disk), 0=encode
# Multiple Python processes per GPU. RBSRecord blocks on GPU->CPU sync every
# step (obs/seg/depth/poses), so a single process leaves the GPU idle ~90% of
# the time. Stacking K independent processes on the same device lets their
# render bursts overlap each other's CPU work and hammers the SMs harder.
# Each process gets disjoint seed range so trajectories don't collide.
# Heuristic: pick K so K * (per-process VRAM) < total VRAM - 1GB(desktop).
# 8GB card: 2 procs × NUM_ENVS=8 ≈ 4-5GB total, comfortable.
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MIKASA_ROOT="${MIKASA_ROOT:-${REPO_ROOT}}"
MANISKILL_ROOT="${MANISKILL_ROOT:-}"
CONDA_ENV="${CONDA_ENV:-mikasa-robo}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_BASE="${CONDA_BASE:-}"

# ─── env setup ───────────────────────────────────────────────────────────────
if [[ -z "${CONDA_BASE}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "[ERROR] 'conda' not found in PATH. Set CONDA_BASE or initialize Conda first." >&2
        exit 1
    fi
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
fi

if [[ -z "${CONDA_BASE}" || ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "[ERROR] Could not locate conda.sh under CONDA_BASE='${CONDA_BASE}'." >&2
    echo "        Override with: CONDA_BASE=/path/to/conda bash run_scripts/collect_gpu_batched.sh" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHONPATH_ENTRIES=("${MIKASA_ROOT}")
if [[ -n "${MANISKILL_ROOT}" ]]; then
    PYTHONPATH_ENTRIES+=("${MANISKILL_ROOT}")
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
    PYTHONPATH_ENTRIES+=("${PYTHONPATH}")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_ENTRIES[*]}")"
unset DISPLAY                                # no X server, no viewer

mkdir -p "${DATA_ROOT}/_logs"

# ─── figure out which GPUs to use ────────────────────────────────────────────
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
else
    if command -v nvidia-smi >/dev/null 2>&1; then
        mapfile -t GPU_LIST < <(nvidia-smi -L | awk -F'GPU ' '/^GPU/{split($2,a,":"); print a[1]}')
    else
        GPU_LIST=(0)
    fi
fi
N_GPUS=${#GPU_LIST[@]}
TOTAL_PROCS=$(( N_GPUS * PROCS_PER_GPU ))

# Each subprocess (one per (gpu_id, slot)) gets a (rounded-up) slice of NUM_EPISODES.
PER_PROC=$(( (NUM_EPISODES + TOTAL_PROCS - 1) / TOTAL_PROCS ))
# Backward-compat alias (older messages used PER_GPU naming).
PER_GPU="${PER_PROC}"

echo "============================================================="
echo "[gpu-batched] env=${ENV_ID}  total_episodes=${NUM_EPISODES}"
echo "  num_envs/proc      = ${NUM_ENVS}    sensor = ${SENSOR_W}x${SENSOR_H}"
echo "  postprocess workers= ${POSTPROCESS_WORKERS}  (cores=${_NPROC})"
echo "  per-traj parallel  = ${PER_TRAJ_PARALLEL}  (convert+flow+point+seg)"
echo "  delete raw npy     = ${DELETE_NPY}"
echo "  skip flow mp4      = ${SKIP_FLOW_MP4}  (1=no scene_point_flow_*.mp4, only rgb.mp4)"
echo "  GPUs               = ${GPU_LIST[*]}  (${N_GPUS} GPU(s))"
echo "  procs per GPU      = ${PROCS_PER_GPU}  (total subprocs=${TOTAL_PROCS})"
echo "  episodes/proc      = ${PER_PROC}"
echo "  data root          = ${DATA_ROOT}"
echo "============================================================="
echo "[hint] num_envs=8 used ~4.5GB VRAM; if you OOM at num_envs=${NUM_ENVS},"
echo "       rerun with: NUM_ENVS=8 bash run_scripts/collect_gpu_batched.sh"
echo "=============================================================" 

# ─── build the optional flag list ────────────────────────────────────────────
FLAGS=()
[[ "${RECORD_ID_POSES}" == "1" ]] && FLAGS+=(--record-id-poses) || FLAGS+=(--no-record-id-poses)
[[ "${SAVE_VIDEO}"      == "1" ]] && FLAGS+=(--save-video)      || FLAGS+=(--no-save-video)
[[ "${POSTPROCESS}"     == "1" ]] && FLAGS+=(--postprocess-camera-data) \
                                  || FLAGS+=(--no-postprocess-camera-data)
[[ "${DELETE_NPY}"      == "1" ]] && FLAGS+=(--postprocess-delete-npy) \
                                  || FLAGS+=(--no-postprocess-delete-npy)

# ─── launch ──────────────────────────────────────────────────────────────────
# Single-GPU → run in the FOREGROUND so tqdm progress bars and postprocess
# logs stream straight to your terminal. No log file unless something fails
# (the python subprocess still writes its own internal logs; we just don't
# redirect stdout).
#
# Multi-GPU → keep the per-GPU log redirection, because N tqdm progress bars
# all fighting over the same TTY would interleave into garbage. We tell you
# exactly how to follow each subprocess with `tail -f`.

run_one_subprocess() {
    local gpu_id="$1"
    local seed="$2"
    local n_eps="$3"
    cd "${MIKASA_ROOT}"
    # Cap ffmpeg threads inside flow_compress so 6 trajs × N threads doesn't
    # oversubscribe the CPU. Total target ≈ POSTPROCESS_WORKERS × FFMPEG_THREADS
    # ≤ nproc.
    local ffmpeg_threads=$(( _NPROC / (POSTPROCESS_WORKERS > 0 ? POSTPROCESS_WORKERS : 1) ))
    [[ "${ffmpeg_threads}" -lt 1 ]] && ffmpeg_threads=1
    [[ "${ffmpeg_threads}" -gt 4 ]] && ffmpeg_threads=4
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    RBS_PER_TRAJ_PARALLEL="${PER_TRAJ_PARALLEL}" \
    RBS_SKIP_FLOW_COMPRESS="${SKIP_FLOW_MP4}" \
    RBS_FFMPEG_THREADS="${ffmpeg_threads}" \
    "${PYTHON_BIN}" -u -m mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets \
        --env_id="${ENV_ID}" \
        --ckpt-dir "${MIKASA_ROOT}" \
        --gpu-batched \
        --num-envs "${NUM_ENVS}" \
        --num-episodes "${n_eps}" \
        --seed "${seed}" \
        --sensor-width "${SENSOR_W}" \
        --sensor-height "${SENSOR_H}" \
        --fps "${FPS}" \
        --postprocess-workers "${POSTPROCESS_WORKERS}" \
        --path-to-save-data "${DATA_ROOT}" \
        "${FLAGS[@]}"
}

if [[ "${TOTAL_PROCS}" == "1" ]]; then
    GPU_ID="${GPU_LIST[0]}"
    SEED=$(( SEED_START ))
    echo "[launch] gpu=${GPU_ID}  seed=${SEED}  episodes=${PER_PROC}  (foreground, live output)"
    echo
    if run_one_subprocess "${GPU_ID}" "${SEED}" "${PER_PROC}"; then
        echo
        echo "[DONE] subprocess finished successfully"
        echo "  collected output → ${DATA_ROOT}/MIKASA-Robo/gpu_batched/${ENV_ID}-${NUM_EPISODES}/"
    else
        echo "[ERROR] subprocess failed (see error output above)"
        exit 1
    fi
else
    PIDS=()
    LOGS=()
    proc_idx=0
    for gpu_id in "${GPU_LIST[@]}"; do
        for slot in $(seq 0 $(( PROCS_PER_GPU - 1 ))); do
            SEED=$(( SEED_START + proc_idx * PER_PROC ))
            LOG="${DATA_ROOT}/_logs/${ENV_ID}_gpu${gpu_id}_slot${slot}_seed${SEED}.log"
            echo "[launch] gpu=${gpu_id} slot=${slot}  seed=${SEED}  episodes=${PER_PROC}  log=${LOG}"
            ( run_one_subprocess "${gpu_id}" "${SEED}" "${PER_PROC}" ) >"${LOG}" 2>&1 &
            PIDS+=($!)
            LOGS+=("${LOG}")
            proc_idx=$(( proc_idx + 1 ))
        done
    done

    echo
    echo "[hint] to follow live progress in another terminal, e.g.:"
    for log in "${LOGS[@]}"; do
        echo "       tail -f ${log}"
    done
    echo "[hint] to watch GPU saturation:"
    echo "       nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1"
    echo
    echo "[wait] waiting for all ${TOTAL_PROCS} subprocess(es) to finish ..."

    fail=0
    for pid in "${PIDS[@]}"; do
        if ! wait "${pid}"; then
            echo "[ERROR] subprocess pid=${pid} failed; check the corresponding log in ${DATA_ROOT}/_logs/"
            fail=1
        fi
    done

    if [[ "${fail}" == "0" ]]; then
        echo "[DONE] all ${TOTAL_PROCS} subprocess(es) finished successfully"
        echo "  collected output → ${DATA_ROOT}/MIKASA-Robo/gpu_batched/${ENV_ID}-${NUM_EPISODES}/"
    else
        exit 1
    fi
fi
