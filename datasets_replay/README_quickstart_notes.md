# MIKASA-Robo Quickstart Notes

This file summarizes the practical notes from the recent setup and first-run session for `quick_start.ipynb`.

## What `quick_start.ipynb` Is

`quick_start.ipynb` is a Jupyter Notebook demo for MIKASA-Robo. It is not a plain Python script that you run directly with `python xxx.py`. Instead, you open it in Jupyter and execute notebook cells.

Its purpose is:

- verify that the environments can be created and stepped
- render task rollouts
- save rollout videos under `./videos/`
- show how wrappers are used

The demos use random actions, so low rewards are expected.

## How To Run The Notebook

Activate the environment and start Jupyter from the project root:

```bash
conda activate mikasa-robo
cd ~/Sim_Data/MIKASA-Robo
jupyter notebook quick_start.ipynb
```

You can also use JupyterLab:

```bash
conda activate mikasa-robo
cd ~/Sim_Data/MIKASA-Robo
jupyter lab quick_start.ipynb
```

Then:

1. Open `quick_start.ipynb` in the browser.
2. Click a code cell.
3. Press `Shift+Enter` to run that cell.

Each code cell launches simulation rollouts, records video, and may display the recorded video inline in the notebook.

## What Each Example Does

### Example 1: Basic Usage

Environment:

- `RememberColor9-v0`
- `num_envs=4`
- `obs_mode="rgb"`

What it does:

- creates 4 parallel environments
- applies `StateOnlyTensorToDictWrapper` which is required for MIKASA-Robo
- records a video using `RecordEpisode`
- resets the environment
- runs for 60 steps with random actions
- closes the environment
- displays the saved video

Why it exists:

- fastest sanity check that the environment works
- simplest example with minimal wrapping

### Example 2: Recommended Usage With Predefined Wrappers

Environment:

- controlled by `env_name`, initially `RememberColor9-v0`

What it does:

- creates the environment
- calls `env_info(env_name)` to get the recommended wrappers and episode timeout
- applies the recommended wrapper stack automatically
- records video
- runs the environment with random actions

Why it exists:

- this is the preferred pattern for normal usage
- changing tasks is easier because the wrapper setup is task-aware

### Example 3: Selective Wrappers For Debugging

Environment:

- `ShellGameTouch-v0`

What it does:

- manually applies a set of wrappers:
  - `StateOnlyTensorToDictWrapper`
  - `InitialZeroActionWrapper`
  - `ShellGameRenderCupInfoWrapper`
  - `RenderStepInfoWrapper`
  - `RenderRewardInfoWrapper`
  - `DebugRewardWrapper`
- records video
- runs random actions for the full episode

Why it exists:

- useful for debugging and understanding task behavior
- overlays extra information such as step count, target cup, and reward

## Why Rewards Look Very Low

The notebook examples use:

```python
action = env.action_space.sample()
```

This means the robot acts randomly. It is not using a trained policy.

So rewards like `0.0x` are normal. These notebook demos are only for checking that the environment, rendering, and wrappers work.

To get meaningful performance, you need to train or load an agent.

## Issues Encountered And Their Meaning

### 1. `jupyter notebook` command not found

Cause:

- system `jupyter-core` alone is not enough
- you must use the `mikasa-robo` conda environment, which already has notebook packages installed

Correct usage:

```bash
conda activate mikasa-robo
cd ~/Sim_Data/MIKASA-Robo
jupyter notebook quick_start.ipynb
```

### 2. CUDA / kernel image error on first run

Observed symptom:

- notebook failed when creating the environment
- message referenced CUDA kernel image availability

Cause:

- initially looked like a GPU issue, but the machine does have a working NVIDIA GPU
- the real problem was PyTorch/CUDA compatibility with the hardware

After the environment was corrected, the notebook ran successfully.

Practical takeaway:

- if environment creation fails with CUDA-related errors, verify:
  - `nvidia-smi`
  - `torch.cuda.is_available()`
  - PyTorch version vs GPU architecture

### 3. `ShellGameTouch-v0` missing asset file

Observed symptom:

- `FileNotFoundError` for:
  - `~/.maniskill/data/assets/mani_skill2_ycb/info_pick_v0.json`

Cause:

- `ShellGameTouch-v0` needs ManiSkill YCB assets
- those assets are not automatically present

Important detail:

- the correct asset ID for ManiSkill's downloader is `ycb`
- `ShellGameTouch-v0` itself is not a valid download ID

### 4. Asset downloader hash mismatch

Observed symptom:

- `Downloaded file's SHA-256 hash does not match record`

Cause:

- the download was incomplete or interrupted

Workable solution:

- use manual download with resumable tools like `wget -c`
- then unzip into `~/.maniskill/data/assets/`

Once the YCB assets were present, `ShellGameTouch-v0` ran successfully.

## Asset Setup For `ShellGameTouch-v0`

If `ShellGameTouch-v0` complains about missing YCB assets, use one of the following.

Downloader route:

```bash
conda activate mikasa-robo
python -m mani_skill.utils.download_asset ycb
```

If that fails because the network is unstable, manual download is more reliable:

```bash
wget -c "https://huggingface.co/datasets/haosulab/ManiSkill2/resolve/main/data/mani_skill2_ycb.zip" \
  -O /tmp/mani_skill2_ycb.zip

mkdir -p ~/.maniskill/data/assets/
unzip /tmp/mani_skill2_ycb.zip -d ~/.maniskill/data/assets/
```

Success signs:

- `~/.maniskill/data/assets/mani_skill2_ycb/models/` exists
- `~/.maniskill/data/assets/mani_skill2_ycb/info_pick_v0.json` exists

## What You See When It Works

When a notebook cell runs successfully, you should see things like:

- a completed progress bar such as `60/60` or `90/90`
- printed observation keys like:
  - `dict_keys([...])`
- a rendered grid of rollouts for `num_envs=4`
- an inline video player or saved video under `./videos/...`

For `ShellGameTouch-v0`, the debug wrapper example may show overlays such as:

- `Step`
- `Target`
- `Reward`

## What To Do To Start The Benchmark Properly

There are two main directions.

### A. Online RL: Train An Agent In The Simulator

The repository already provides PPO baselines.

State-based PPO:

```bash
python3 baselines/ppo/ppo_memtasks.py \
    --env_id=RememberColor9-v0 \
    --exp-name=remember-color-9-v0 \
    --num-steps=60 \
    --num_eval_steps=180 \
    --include-state
```

RGB + joints PPO:

```bash
python3 baselines/ppo/ppo_memtasks.py \
    --env_id=RememberColor9-v0 \
    --exp-name=remember-color-9-v0 \
    --num-steps=60 \
    --num_eval_steps=180 \
    --include-rgb \
    --include-joints
```

LSTM PPO:

```bash
python3 baselines/ppo/ppo_memtasks_lstm.py \
    --env_id=RememberColor9-v0 \
    --exp-name=remember-color-9-v0 \
    --num-steps=60 \
    --num_eval_steps=180 \
    --include-rgb \
    --include-joints
```

Meaning of the modes:

- `--include-state`: gives rich state/oracle information, so memory is not really required
- `--include-rgb --include-joints`: gives visual and proprioceptive input, so the agent must solve the memory problem more directly

Recommended first experiment:

- start with `RememberColor3-v0` or `RememberColor9-v0`
- confirm training runs end-to-end
- then move to harder tasks such as `ShellGameTouch-v0`

### B. Offline RL: Train From Datasets

The project provides downloadable datasets for all tasks.

Example:

```bash
wget https://huggingface.co/datasets/avanturist/mikasa-robo/resolve/main/ShellGameTouch-v0.zip
unzip ShellGameTouch-v0.zip
```

Typical dataset fields:

- `rgb`
- `joints`
- `action`
- `reward`
- `success`
- `done`

Use this route if you want to benchmark offline RL algorithms without running environment interaction during training.

## Suggested First Practical Path

If you are just starting:

1. Ensure `quick_start.ipynb` basic examples run.
2. Ensure YCB assets are installed if you want `ShellGameTouch-v0`.
3. Run a small PPO experiment on an easier task.
4. Move to visual-memory settings with `--include-rgb --include-joints`.
5. Compare performance across tasks and observation modes.

## Important Reminder About Wrappers

For MIKASA-Robo environments, always use:

```python
env = StateOnlyTensorToDictWrapper(env)
```

unless a helper already applies it for you through the predefined wrapper list.

Without the correct wrapper setup, observation keys may not match what the benchmark expects.
