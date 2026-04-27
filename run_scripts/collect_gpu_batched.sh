#!/usr/bin/env bash
# ============================================================================
# MIKASA-Robo 一键数据采集 + 点云可视化
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        可 调 参 数  ← 直接改这里                         ║
# ╠══════════════════════════════════════════════════════════════════════════╣

TASK_ID="ShellGameTouch-v0"
# 任务 ID，脚本会自动加载对应的 oracle policy
# 可选: InterceptGrabSlow-v0 / ShellGamePush-v0 / ShellGameTouch-v0 / ...

NUM_EPISODES=128
# 总共采集多少条轨迹（跨所有进程累计）

NUM_PROCS=4
# 并行进程数（跑在同一 GPU 上，或自动分配到多 GPU）
# 每进程约占 2-3 GB VRAM；单 8 GB GPU 建议 ≤ 2，24 GB GPU 可用 4-6

SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/data}"
# 落盘根路径，实际写入: $SAVE_DIR/MIKASA-Robo/gpu_batched/$TASK_ID/

# ╚══════════════════════════════════════════════════════════════════════════╝

# ─── 高级参数（通常不需要动）────────────────────────────────────────────────
NUM_ENVS=2           # 每进程的 GPU 并行 env 数
SEED_START=0
SENSOR_W=832
SENSOR_H=480
FPS=16
SAVE_VIDEO=1
POSTPROCESS=1
DELETE_NPY=1
SKIP_FLOW_MP4=0
PER_TRAJ_PARALLEL=1
_NPROC=$(nproc 2>/dev/null || echo 8)
POSTPROCESS_WORKERS="${POSTPROCESS_WORKERS:-$(( NUM_ENVS < (_NPROC / 3) ? NUM_ENVS : (_NPROC / 3) ))}"
[[ "${POSTPROCESS_WORKERS}" -lt 1 ]] && POSTPROCESS_WORKERS=1

MIKASA_ROOT="${MIKASA_ROOT:-${REPO_ROOT}}"
MANISKILL_ROOT="${MANISKILL_ROOT:-}"
CONDA_ENV="${CONDA_ENV:-mikasa-robo}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_BASE="${CONDA_BASE:-}"

# ─── 环境初始化 ───────────────────────────────────────────────────────────
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
unset DISPLAY

mkdir -p "${SAVE_DIR}/_logs"

# ─── GPU 检测 ─────────────────────────────────────────────────────────────
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

# 把 NUM_PROCS 个进程尽量均匀分配到各 GPU
TOTAL_PROCS="${NUM_PROCS}"
PER_PROC=$(( (NUM_EPISODES + TOTAL_PROCS - 1) / TOTAL_PROCS ))

# ─── 启动信息 ─────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
printf  "║  TASK     : %-47s ║\n" "${TASK_ID}"
printf  "║  episodes : %-6s   procs : %-6s   envs/proc : %-8s ║\n" \
        "${NUM_EPISODES}" "${TOTAL_PROCS}" "${NUM_ENVS}"
printf  "║  save_dir : %-47s ║\n" "${SAVE_DIR}"
printf  "║  GPUs     : %-47s ║\n" "${GPU_LIST[*]}  (${N_GPUS} GPU(s))"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# ─── 采集函数 ─────────────────────────────────────────────────────────────
FLAGS=()
[[ "${SAVE_VIDEO}"   == "1" ]] && FLAGS+=(--save-video)               || FLAGS+=(--no-save-video)
[[ "${POSTPROCESS}"  == "1" ]] && FLAGS+=(--postprocess-camera-data)  || FLAGS+=(--no-postprocess-camera-data)
[[ "${DELETE_NPY}"   == "1" ]] && FLAGS+=(--postprocess-delete-npy)   || FLAGS+=(--no-postprocess-delete-npy)
FLAGS+=(--record-id-poses)

_run_collect() {
    local gpu_id="$1" seed="$2" n_eps="$3"
    local ffmpeg_threads=$(( _NPROC / (POSTPROCESS_WORKERS > 0 ? POSTPROCESS_WORKERS : 1) ))
    [[ "${ffmpeg_threads}" -lt 1 ]] && ffmpeg_threads=1
    [[ "${ffmpeg_threads}" -gt 4 ]] && ffmpeg_threads=4
    cd "${MIKASA_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    RBS_PER_TRAJ_PARALLEL="${PER_TRAJ_PARALLEL}" \
    RBS_SKIP_FLOW_COMPRESS="${SKIP_FLOW_MP4}" \
    RBS_FFMPEG_THREADS="${ffmpeg_threads}" \
    "${PYTHON_BIN}" -u -m mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets \
        --env_id="${TASK_ID}" \
        --ckpt-dir "${MIKASA_ROOT}" \
        --gpu-batched \
        --num-envs "${NUM_ENVS}" \
        --num-episodes "${n_eps}" \
        --seed "${seed}" \
        --sensor-width "${SENSOR_W}" \
        --sensor-height "${SENSOR_H}" \
        --fps "${FPS}" \
        --postprocess-workers "${POSTPROCESS_WORKERS}" \
        --path-to-save-data "${SAVE_DIR}" \
        "${FLAGS[@]}"
}

# ─── 采集：单进程（前台，实时输出）/ 多进程（后台，日志文件）────────────────
OUTPUT_DIR="${SAVE_DIR}/MIKASA-Robo/gpu_batched/${TASK_ID}"

if [[ "${TOTAL_PROCS}" == "1" ]]; then
    GPU_ID="${GPU_LIST[0]}"
    echo "[collect] gpu=${GPU_ID}  seed=${SEED_START}  episodes=${PER_PROC}"
    echo
    _run_collect "${GPU_ID}" "${SEED_START}" "${PER_PROC}"
    echo
    echo "[collect] 完成"
else
    PIDS=(); LOGS=()
    for proc_idx in $(seq 0 $(( TOTAL_PROCS - 1 ))); do
        gpu_id="${GPU_LIST[$(( proc_idx % N_GPUS ))]}"
        seed=$(( SEED_START + proc_idx * PER_PROC ))
        log="${SAVE_DIR}/_logs/${TASK_ID}_proc${proc_idx}_seed${seed}.log"
        echo "[collect] proc=${proc_idx}  gpu=${gpu_id}  seed=${seed}  episodes=${PER_PROC}  → ${log}"
        ( _run_collect "${gpu_id}" "${seed}" "${PER_PROC}" ) >"${log}" 2>&1 &
        PIDS+=($!)
        LOGS+=("${log}")
    done
    echo
    echo "[hint] 实时查看进度："
    for log in "${LOGS[@]}"; do echo "         tail -f ${log}"; done
    echo
    echo "[wait] 等待 ${TOTAL_PROCS} 个进程完成..."
    fail=0
    for pid in "${PIDS[@]}"; do
        wait "${pid}" || { echo "[ERROR] pid=${pid} 失败，见 ${SAVE_DIR}/_logs/"; fail=1; }
    done
    [[ "${fail}" == "1" ]] && exit 1
    echo "[collect] 全部完成"
fi

# ─── 点云可视化 ───────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════════"
echo "[inspect] 开始对每条 traj 绘制四关键帧点云图..."
echo "          输入: ${OUTPUT_DIR}"
echo "════════════════════════════════════════════════════════════════"
echo

# inspect_sceneflow_first_frame.py 期望 --root 指向单次采集的 task 目录
# 它会自动遍历 <root>/camera_data/traj_* 并在各 traj 目录下写 _sceneflow_check_ref*.png
cd "${MIKASA_ROOT}"
"${PYTHON_BIN}" run_scripts/inspect_sceneflow_first_frame.py \
    --root "${OUTPUT_DIR}"

echo
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  全部完成！                                                   ║"
printf "║  数据路径: %-49s ║\n" "${OUTPUT_DIR}"
echo "║  点云图  : 每条 traj 目录下的 _sceneflow_check_ref*.png       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
