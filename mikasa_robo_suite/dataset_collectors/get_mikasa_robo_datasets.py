import os
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
from pathlib import Path
import shutil
import tyro
from dataclasses import dataclass
from typing import Optional

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from baselines.ppo.ppo_memtasks import AgentStateOnly, FlattenRGBDObservationWrapper
from mikasa_robo_suite.memory_envs import *
from mikasa_robo_suite.utils.wrappers import *


# ─────────────────────────────────────────────────────────────────────────────
# SAPIEN viewer hook (required for render_mode="human" with GPU sim backend)
# ─────────────────────────────────────────────────────────────────────────────

def _install_viewer_hook() -> None:
    """Monkey-patch BaseEnv._reconfigure to pre-initialize the SAPIEN viewer
    before the render_system_group is locked.

    ManiSkill creates render_system_group during the first get_obs() /
    update_render() call inside gym.make().  After that point no new entities
    can be added to the scene.  If the viewer is initialised later (on the
    first env.render()), viewer.set_scene() tries to add internal entities to
    an already-locked scene and crashes.  The fix is to create the viewer
    inside _reconfigure(), right after scene construction but before
    get_obs() seals it.

    NOTE: The SAPIEN viewer's F-key camera-switch is NOT supported when using
    the GPU sim backend.  Switching cameras at runtime modifies the scene's
    render state and triggers:
      RuntimeError: Modifying a scene … is not allowed after creating the
                    batched render system.
    Use the --viewer-camera CLI option to select the camera view at startup.
    """
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.viewer import create_viewer

    if getattr(BaseEnv, "_viewer_hook_installed", False):
        return

    _orig_reconfigure = BaseEnv._reconfigure

    def _hooked(env_self, options=dict()):  # noqa: B006
        _orig_reconfigure(env_self, options)
        if (getattr(env_self, "render_mode", None) == "human"
                and getattr(env_self, "_viewer", None) is None):
            env_self._viewer = create_viewer(env_self._viewer_camera_config)
            env_self._setup_viewer()

    BaseEnv._reconfigure = _hooked
    BaseEnv._viewer_hook_installed = True


import contextlib

@contextlib.contextmanager
def _override_human_render_camera(env_id: str, viewer_camera: str):
    """Context manager: temporarily override _default_human_render_camera_configs
    on the registered env class so that gym.make() builds the SAPIEN viewer
    with the requested camera perspective from the start.

    This is the correct way to change the initial viewpoint – it avoids the
    runtime scene-modification crash caused by pressing F in the viewer.

    Supported viewer_camera values
    --------------------------------
    'render_camera'  – default render camera defined by the env (no-op)
    '<sensor_name>'  – any fixed (un-mounted) sensor camera, e.g. 'base_camera'
                       Mounted cameras (e.g. 'hand_camera') are NOT supported
                       because their pose is relative to a moving robot link.
    """
    if viewer_camera == "render_camera":
        yield
        return

    # ── locate the env class via the gymnasium registry ───────────────────────
    try:
        import functools, importlib
        spec = gym.spec(env_id)
        entry = spec.entry_point
        if isinstance(entry, functools.partial):
            # gymnasium sometimes wraps the class in functools.partial; unwrap it
            env_class = entry.func
        elif isinstance(entry, type):
            env_class = entry
        elif callable(entry):
            env_class = entry
        elif isinstance(entry, str) and ":" in entry:
            mod_path, cls_name = entry.rsplit(":", 1)
            env_class = getattr(importlib.import_module(mod_path), cls_name)
        else:
            raise ValueError(f"Cannot parse entry_point: {entry!r}")
        if not isinstance(env_class, type):
            raise ValueError(f"Resolved env_class is not a type: {env_class!r}")
    except Exception as exc:
        print(f"[warn] --viewer-camera: could not locate env class ({exc}); "
              "falling back to render_camera.")
        yield
        return

    # ── read the sensor config for the requested camera ───────────────────────
    # Use object.__new__ to avoid running __init__ (we only need the property
    # value, which for most ManiSkill envs only references fixed constants).
    try:
        bare = object.__new__(env_class)
        sensor_cfgs = bare._default_sensor_configs
        if not isinstance(sensor_cfgs, list):
            sensor_cfgs = [sensor_cfgs]
        cam_cfg = next((c for c in sensor_cfgs if c.uid == viewer_camera), None)
    except Exception as exc:
        print(f"[warn] --viewer-camera: could not read sensor configs ({exc}); "
              "falling back to render_camera.")
        yield
        return

    if cam_cfg is None:
        available = [c.uid for c in sensor_cfgs]
        print(f"[warn] --viewer-camera: sensor '{viewer_camera}' not found in "
              f"{env_id}. Available sensor cameras: {available}. "
              "Falling back to render_camera.")
        yield
        return

    if getattr(cam_cfg, "mount", None) is not None:
        print(f"[warn] --viewer-camera: '{viewer_camera}' is a mounted camera "
              "(it follows a robot link and has no fixed world pose). "
              "Only fixed cameras are supported. Falling back to render_camera.")
        yield
        return

    # ── build a high-res render camera at the same pose / FOV ─────────────────
    from mani_skill.sensors.camera import CameraConfig
    render_cam = CameraConfig(
        uid="render_camera",
        pose=cam_cfg.pose,
        width=512,
        height=512,
        fov=cam_cfg.fov,
        near=cam_cfg.near,
        far=cam_cfg.far,
    )

    # ── temporarily override the property on the class ────────────────────────
    orig = env_class.__dict__.get("_default_human_render_camera_configs")
    env_class._default_human_render_camera_configs = property(lambda self: render_cam)
    try:
        yield
    finally:
        if orig is None:
            try:
                delattr(env_class, "_default_human_render_camera_configs")
            except AttributeError:
                pass
        else:
            env_class._default_human_render_camera_configs = orig


def env_info(env_id):
    noop_steps = 1
    if env_id in ['ShellGamePush-v0', 'ShellGamePick-v0', 'ShellGameTouch-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (ShellGameRenderCupInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = 'cup_with_ball_number'
        prompt_info = None
        EPISODE_TIMEOUT = 90
    elif env_id in ['InterceptSlow-v0', 'InterceptMedium-v0', 'InterceptFast-v0', 
                    'InterceptGrabSlow-v0', 'InterceptGrabMedium-v0', 'InterceptGrabFast-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 90
    elif env_id in ['RotateLenientPos-v0', 'RotateLenientPosNeg-v0',
                    'RotateStrictPos-v0', 'RotateStrictPosNeg-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (RotateRenderAngleInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = 'angle_diff'
        prompt_info = 'target_angle'
        EPISODE_TIMEOUT = 90
    elif env_id in ['CameraShutdownPush-v0', 'CameraShutdownPick-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (CameraShutdownWrapper, {"n_initial_steps": 19}), # camera works only for t ~ [0, 19]
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 90
    elif env_id in ['TakeItBack-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 180
    elif env_id in ['RememberColor3-v0', 'RememberColor5-v0', 'RememberColor9-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RememberColorInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 60
    elif env_id in ['RememberShape3-v0', 'RememberShape5-v0', 'RememberShape9-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RememberShapeInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 60
    elif env_id in ['RememberShapeAndColor3x2-v0', 'RememberShapeAndColor3x3-v0', 'RememberShapeAndColor5x3-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RememberShapeAndColorInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 60
    elif env_id in ['BunchOfColors3-v0', 'BunchOfColors5-v0', 'BunchOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 120
    elif env_id in ['SeqOfColors3-v0', 'SeqOfColors5-v0', 'SeqOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 120
    elif env_id in ['ChainOfColors3-v0', 'ChainOfColors5-v0', 'ChainOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 120
    else:
        raise ValueError(f"Unknown environment: {env_id}")
    
    wrappers_list.insert(0, (StateOnlyTensorToDictWrapper, {}))

    return wrappers_list, EPISODE_TIMEOUT


def collect_batched_data_from_ckpt(
    env_id="ShellGameTouch-v0", 
    checkpoint_path=None, 
    path_to_save_data="data",
    num_train_data=1000
):    
    """
    Collect batched data, consequent unbatching required!!!
    """
    # env_id = "ShellGameTouch-v0"
    
    NUMBER_OF_TRAIN_DATA = num_train_data
    batch_size = 250
    NUMBER_OF_BATCHES = NUMBER_OF_TRAIN_DATA // batch_size
    # render = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_kwargs_state = dict(
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="all",
        sim_backend="gpu",
        reward_mode="normalized_dense"
    )

    env_kwargs_rgb = dict(
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        render_mode="all",
        sim_backend="gpu",
        reward_mode="normalized_dense"
    )

    env_state = gym.make(env_id, num_envs=batch_size, **env_kwargs_state)
    env_rgb = gym.make(env_id, num_envs=batch_size, **env_kwargs_rgb)

    state_wrappers_list, episode_timeout = env_info(env_id)

    for wrapper_class, wrapper_kwargs in state_wrappers_list:
        env_state = wrapper_class(env_state, **wrapper_kwargs)

    env_state = FlattenRGBDObservationWrapper(
        env_state, 
        rgb=False,
        depth=False,
        state=True,
        oracle=False,
        joints=False
    )

    rgb_wrappers_list, _ = env_info(env_id)

    for wrapper_class, wrapper_kwargs in rgb_wrappers_list:
        env_rgb = wrapper_class(env_rgb, **wrapper_kwargs)

    env_rgb = FlattenRGBDObservationWrapper(
        env_rgb, 
        rgb=True,
        depth=False,
        state=False,
        oracle=False,
        joints=True
    )

    if isinstance(env_state.action_space, gym.spaces.Dict):
        env_state = FlattenActionSpaceWrapper(env_state)
    if isinstance(env_rgb.action_space, gym.spaces.Dict):
        env_rgb = FlattenActionSpaceWrapper(env_rgb)

    # env_state = RecordEpisode(
    #     env_state,
    #     output_dir="dataset_collection_videos/state",
    #     save_trajectory=True,
    #     video_fps=30
    # )

    # env_rgb = RecordEpisode(
    #     env_rgb,
    #     output_dir="dataset_collection_videos/rgb",
    #     save_trajectory=True,
    #     video_fps=30
    # )

    env_state = ManiSkillVectorEnv(env_state, batch_size, ignore_terminations=True, record_metrics=True)
    env_rgb = ManiSkillVectorEnv(env_rgb, batch_size, ignore_terminations=True, record_metrics=True)

    agent = AgentStateOnly(env_state).to(device)
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()


    # save_dir = f'data/MIKASA-Robo/batched/{env_id}'
    save_dir = f'{path_to_save_data}/MIKASA-Robo/batched/{env_id}'
    os.makedirs(save_dir, exist_ok=True)

    # Dataset collection
    print(f"Generating {NUMBER_OF_TRAIN_DATA} episodes in {NUMBER_OF_BATCHES} batches (batched with batch size {batch_size})")
    for episode in tqdm(range(NUMBER_OF_BATCHES)):
        rgbList, jointsList,actList, rewList, succList, doneList = [], [], [], [], [], []
        
        # Reset of both environments with the same seed for synchronization
        # seed = np.random.randint(0, 10000)
        seed = episode
        obs_state, _ = env_state.reset(seed=seed)
        obs_rgb, _ = env_rgb.reset(seed=seed)
        
        done = False
        for t in range(episode_timeout):
            # Get action from agent based on state
            rgbList.append(obs_rgb['rgb'].cpu().numpy())
            jointsList.append(obs_rgb['joints'].cpu().numpy())
            with torch.no_grad():
                for key, value in obs_state.items():
                    obs_state[key] = value.to(device)
                action = agent.get_action(obs_state, deterministic=True)
            
            # Make a step in both environments with the same action
            obs_state, reward_state, term_state, trunc_state, info_state = env_state.step(action)
            obs_rgb, reward_rgb, term_rgb, trunc_rgb, info_rgb = env_rgb.step(action)

            rewList.append(reward_rgb.cpu().numpy())
            succList.append(info_rgb['success'].cpu().numpy().astype(int))
            actList.append(action.cpu().numpy())
            done = torch.logical_or(term_rgb, trunc_rgb)
            doneList.append(done.cpu().numpy().astype(int))
            
            # Check synchronization of environments
            # assert np.allclose(reward_state.cpu().numpy(), reward_rgb.cpu().numpy()), "Environments desynchronized!"

        DATA = {'rgb': np.array(rgbList), # (15, 6, 128, 128)
                'joints': np.array(jointsList), # (15, 25)
                'action': np.array(actList), # (15, 8)
                'reward': np.array(rewList), # (15,)
                'success': np.array(succList), # (15,)
                'done': np.array(doneList)} # (15,)
        
        file_path = f'{save_dir}/train_data_{episode}.npz'
        np.savez(file_path, **DATA)
        
        # print(f"Episode completed")
        # if "final_info" in info_rgb:
        #     for k, v in info_rgb["final_info"]["episode"].items():
        #         print(f"{k}: {v.item()}")


    env_state.close()
    env_rgb.close()

    print(f"\nDataset saved to {save_dir}")


def collect_unbatched_data_from_batched(env_id="ShellGameTouch-v0", path_to_save_data="data"):
    dir_with_batched_data = f'{path_to_save_data}/MIKASA-Robo/batched/{env_id}'
    NUMBER_OF_BATCHES = len(list(Path(dir_with_batched_data).glob('*')))
    print(f"Unbatching {dir_with_batched_data}, {NUMBER_OF_BATCHES} batches")

    traj_cnt = 0
    save_dir_unbatched = f'{path_to_save_data}/MIKASA-Robo/unbatched/{env_id}'
    os.makedirs(save_dir_unbatched, exist_ok=True)

    for episode in tqdm(range(NUMBER_OF_BATCHES)):
        episode = np.load(f'{dir_with_batched_data}/train_data_{episode}.npz')
        episode = {key: episode[key] for key in episode.keys()}
        for trajectory_num in range(episode['reward'].shape[1]):
            unbatched_rgb = episode['rgb'][:, trajectory_num, :, :, :]
            unbatched_joints = episode['joints'][:, trajectory_num, :]
            unbatched_action = episode['action'][:, trajectory_num, :]
            unbatched_reward = episode['reward'][:, trajectory_num]
            unbatched_success = episode['success'][:, trajectory_num]
            unbatched_done = episode['done'][:, trajectory_num]

            DATA = {'rgb': unbatched_rgb,
                    'joints': unbatched_joints,
                    'action': unbatched_action,
                    'reward': unbatched_reward,
                    'success': unbatched_success,
                    'done': unbatched_done}

            file_path = f'{save_dir_unbatched}/train_data_{traj_cnt}.npz'
            np.savez(file_path, **DATA)

            traj_cnt += 1


def collect_single_episode_with_visualization(
    env_id: str = "ShellGameTouch-v0",
    checkpoint_path: str = None,
    seed: int = 0,
    save_dir: Optional[str] = None,
    viewer_camera: str = "render_camera",
) -> None:
    """Collect **one** episode using the oracle RL agent and display it live in
    the SAPIEN interactive 3-D viewer.

    Two environments are run in lock-step (same seed, same actions):
      * env_state  – state observations only, used for agent inference
      * env_vis    – rgb observations + render_mode="human", used for display

    Parameters
    ----------
    env_id          : Gym environment ID registered in MIKASA-Robo.
    checkpoint_path : Path to the oracle agent checkpoint (.pt file).
    seed            : Random seed passed to env.reset().
    save_dir        : If given, saves the episode as
                      <save_dir>/traj_seed<seed>.npz  (rgb, joints, action,
                      reward, success, done).  Pass None to skip saving.
    viewer_camera   : Which camera perspective to use in the SAPIEN viewer.
                      'render_camera' (default) uses the env's built-in render
                      camera.  Pass a sensor camera name (e.g. 'base_camera')
                      to start from that perspective.
                      WARNING: pressing F in the SAPIEN viewer to switch
                      cameras at runtime is NOT supported with the GPU sim
                      backend and will crash.  Use this parameter instead.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wl_state, episode_timeout = env_info(env_id)
    wl_vis, _                 = env_info(env_id)

    # Install the SAPIEN viewer hook BEFORE any gym.make() call.
    _install_viewer_hook()

    # ── state env (agent inference only, no viewer) ───────────────────────────
    print("[1/3] Building state env...")
    env_state = gym.make(
        env_id, num_envs=1,
        obs_mode="state",
        render_mode="sensors",
        sim_backend="gpu",
        reward_mode="normalized_dense",
    )
    for WC, kw in wl_state:
        env_state = WC(env_state, **kw)
    env_state = FlattenRGBDObservationWrapper(
        env_state, rgb=False, depth=False, state=True, oracle=False, joints=False)
    if isinstance(env_state.action_space, gym.spaces.Dict):
        env_state = FlattenActionSpaceWrapper(env_state)
    env_state = ManiSkillVectorEnv(env_state, 1, ignore_terminations=True)

    # ── visual env (SAPIEN viewer + rgb obs for data) ─────────────────────────
    print(f"[2/3] Building visual env (viewer_camera='{viewer_camera}')...")
    with _override_human_render_camera(env_id, viewer_camera):
        env_vis = gym.make(
            env_id, num_envs=1,
            obs_mode="rgb",
            render_mode="human",       # ← opens the SAPIEN interactive viewer
            sim_backend="gpu",
            reward_mode="normalized_dense",
        )
    for WC, kw in wl_vis:
        env_vis = WC(env_vis, **kw)
    env_vis = FlattenRGBDObservationWrapper(
        env_vis, rgb=True, depth=False, state=False, oracle=False, joints=True)
    if isinstance(env_vis.action_space, gym.spaces.Dict):
        env_vis = FlattenActionSpaceWrapper(env_vis)
    env_vis = ManiSkillVectorEnv(env_vis, 1, ignore_terminations=True)

    # ── load oracle agent ─────────────────────────────────────────────────────
    print(f"[3/3] Loading checkpoint: {checkpoint_path}")
    agent = AgentStateOnly(env_state).to(device)
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()

    # ── reset both envs with the same seed ───────────────────────────────────
    obs_state, _ = env_state.reset(seed=seed)
    obs_vis,   _ = env_vis.reset(seed=seed)

    rgbList, jointsList, actList, rewList, succList, doneList = [], [], [], [], [], []

    print(f"\nRunning episode  env={env_id}  seed={seed}  timeout={episode_timeout} steps")
    print("SAPIEN viewer controls: right-drag=rotate  scroll=zoom  mid-drag=pan")
    print("⚠  Do NOT press F to switch cameras – GPU sim does not support runtime")
    print("   camera switching.  Use --viewer-camera at startup instead.\n")

    for t in tqdm(range(episode_timeout), desc="steps"):
        rgbList.append(obs_vis["rgb"].cpu().numpy())
        jointsList.append(obs_vis["joints"].cpu().numpy())

        with torch.no_grad():
            for k in obs_state:
                obs_state[k] = obs_state[k].to(device)
            action = agent.get_action(obs_state, deterministic=True)

        obs_state, _, _, _, _ = env_state.step(action)

        try:
            obs_vis, reward, term, trunc, info = env_vis.step(action)
        except RuntimeError as exc:
            if "Modifying a scene" in str(exc):
                tqdm.write(
                    "\n[ERROR] SAPIEN scene was modified at runtime (most likely "
                    "you pressed F to switch cameras).\n"
                    "        GPU sim does not allow camera changes after the render "
                    "system is created.\n"
                    "        Stopping early – data collected so far will be saved."
                )
                break
            raise

        rewList.append(reward.cpu().numpy())
        succList.append(info["success"].cpu().numpy().astype(int))
        actList.append(action.cpu().numpy())
        doneList.append(torch.logical_or(term, trunc).cpu().numpy().astype(int))

        try:
            env_vis.render()   # refresh the SAPIEN viewer each step
        except RuntimeError as exc:
            if "Modifying a scene" in str(exc):
                tqdm.write("\n[ERROR] Viewer camera switch detected – stopping early.")
                break
            raise

        if info["success"].any():
            tqdm.write(f"  ✓ success at step {t}")

    # ── optional: save episode ────────────────────────────────────────────────
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        DATA = {
            "rgb":     np.array(rgbList),     # (T, 1, H, W, 3)
            "joints":  np.array(jointsList),  # (T, 1, D_joints)
            "action":  np.array(actList),     # (T, 1, D_action)
            "reward":  np.array(rewList),     # (T, 1)
            "success": np.array(succList),    # (T, 1)
            "done":    np.array(doneList),    # (T, 1)
        }
        out_path = os.path.join(save_dir, f"traj_seed{seed}.npz")
        np.savez_compressed(out_path, **DATA)
        print(f"\nEpisode saved → {out_path}")
        print(f"  rgb    : {DATA['rgb'].shape}  uint8")
        print(f"  joints : {DATA['joints'].shape}")
        print(f"  action : {DATA['action'].shape}")

    env_state.close()
    env_vis.close()


def get_list_of_all_checkpoints_available(ckpt_dir="."):
    oracle_checkpoints_dir = os.path.join(ckpt_dir, "oracle_checkpoints")

    # 1) Check if oracle_checkpoints_dir exists
    if not os.path.exists(oracle_checkpoints_dir):
        raise FileNotFoundError(f"Directory {oracle_checkpoints_dir} does not exist.")
    
    checkpoint_paths = []
    
    # 2) Iterate over all directories in oracle_checkpoints/ppo_memtasks/state/normalized_dense/
    normalized_dense_dir = os.path.join(oracle_checkpoints_dir, "ppo_memtasks", "state", "normalized_dense")
    for env_dir in os.listdir(normalized_dense_dir):
        env_path = os.path.join(normalized_dense_dir, env_dir)
        if os.path.isdir(env_path):
            # 3) Create list of checkpoint paths
            for root, _, files in os.walk(env_path):
                for file in files:
                    if file == "final_success_ckpt.pt":
                        checkpoint_paths.append([env_dir, os.path.join(root, file)])
    
    return checkpoint_paths


@dataclass
class Args:
    env_id: Optional[str] = "ShellGameTouch-v0"
    """Target environment ID (e.g. ShellGameTouch-v0)."""
    path_to_save_data: str = "data"
    """Root directory for batch dataset output."""
    ckpt_dir: str = "."
    """Directory containing oracle_checkpoints/."""
    num_train_data: int = 1000
    """Total number of training episodes to collect (batch mode only)."""
    # ── single-episode visualization mode ────────────────────────────────────
    visualize: bool = False
    """If True, collect a single episode and display it live in SAPIEN instead
    of running the full batch collection pipeline."""
    seed: int = 0
    """Random seed for the visualized episode (visualize mode only)."""
    save_vis_episode: bool = False
    """If True and --visualize is set, also save the episode to
    <path_to_save_data>/MIKASA-Robo/vis/<env_id>/traj_seed<seed>.npz."""
    viewer_camera: str = "render_camera"
    """Camera perspective for the SAPIEN viewer.
    'render_camera' (default) uses the env's built-in render camera (wide,
    high-resolution, fixed diagonal view).  Pass a sensor camera name such as
    'base_camera' to start the viewer from that perspective instead.
    NOTE: pressing F inside the viewer to switch cameras at runtime is NOT
    supported with the GPU sim backend – use this option at startup instead."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    path_to_save_data = args.path_to_save_data
    ckpt_dir = args.ckpt_dir
    ENV_ID = args.env_id

    # ── visualize mode: single episode + SAPIEN viewer ───────────────────────
    if args.visualize:
        checkpoints = get_list_of_all_checkpoints_available(ckpt_dir=ckpt_dir)
        ckpt_path = None
        for env_id, checkpoint in checkpoints:
            if env_id == ENV_ID:
                ckpt_path = checkpoint
                break
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint found for '{ENV_ID}' under {ckpt_dir}/oracle_checkpoints/")
        save_dir = (
            os.path.join(path_to_save_data, "MIKASA-Robo", "vis", ENV_ID)
            if args.save_vis_episode else None
        )
        collect_single_episode_with_visualization(
            env_id=ENV_ID,
            checkpoint_path=ckpt_path,
            seed=args.seed,
            save_dir=save_dir,
            viewer_camera=args.viewer_camera,
        )

    # ── batch mode: full dataset collection ──────────────────────────────────
    else:
        batch_size = 250
        if args.num_train_data % batch_size != 0:
            raise ValueError(
                f"num_train_data ({args.num_train_data}) must be divisible by "
                f"batch_size ({batch_size})"
            )

        checkpoints = get_list_of_all_checkpoints_available(ckpt_dir=ckpt_dir)

        for env_id, checkpoint in checkpoints:
            if env_id == ENV_ID:
                print(f"Collecting data for {env_id} from {checkpoint}")

                # 1. Collect batched data from ckpt
                collect_batched_data_from_ckpt(
                    env_id=env_id,
                    checkpoint_path=checkpoint,
                    path_to_save_data=path_to_save_data,
                    num_train_data=args.num_train_data,
                )

                # 2. Unbatch batched data
                collect_unbatched_data_from_batched(
                    env_id=env_id,
                    path_to_save_data=path_to_save_data,
                )

                # 3. Remove batched data
                dir_with_batched_data = (
                    f"{path_to_save_data}/MIKASA-Robo/batched/{env_id}"
                )
                shutil.rmtree(dir_with_batched_data)
                print(f"Deleted batched data for {env_id} at {dir_with_batched_data}")


# ── usage examples ────────────────────────────────────────────────────────────
# Default render camera (wide diagonal view):
#   python3 mikasa_robo_suite/dataset_collectors/get_mikasa_robo_datasets.py \
#       --env-id ShellGameTouch-v0 --ckpt-dir . --visualize
#
# Start viewer from base_camera perspective (same as the robot's sensor view):
#   python3 mikasa_robo_suite/dataset_collectors/get_mikasa_robo_datasets.py \
#       --env-id ShellGameTouch-v0 --ckpt-dir . --visualize --viewer-camera base_camera
#
# Visualize and save the episode:
#   python3 mikasa_robo_suite/dataset_collectors/get_mikasa_robo_datasets.py \
#       --env-id ShellGameTouch-v0 --ckpt-dir . --visualize --save-vis-episode --seed 3
#
# Full batch collection (original behaviour):
#   python3 mikasa_robo_suite/dataset_collectors/get_mikasa_robo_datasets.py \
#       --env-id ShellGameTouch-v0 --path-to-save-data data --ckpt-dir . --num-train-data 1000