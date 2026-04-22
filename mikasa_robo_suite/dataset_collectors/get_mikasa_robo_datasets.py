import os
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
from pathlib import Path
import shutil
import tyro
from dataclasses import dataclass
from typing import List, Optional

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from baselines.ppo.ppo_memtasks import AgentStateOnly, FlattenRGBDObservationWrapper
from mikasa_robo_suite.memory_envs import *
from mikasa_robo_suite.utils.wrappers import *

# Vendored from ManiSkill dev_wjj branch (see rbs_record/__init__.py).
# Produces the exact same on-disk layout / fields as dev_wjj's RBSRecordEpisode,
# so that downstream scene_point_flow / object_flow pipelines work unchanged.
from mikasa_robo_suite.dataset_collectors.rbs_record import RBSRecordEpisode


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
    try:
        from mani_skill.viewer import create_viewer
    except ImportError:
        from mani_skill.utils.sapien_utils import create_viewer

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


class SensorDataCollectWrapper(gym.ObservationWrapper):
    """Extracts per-camera modalities directly from raw sensor_data.

    Produced observation keys
    -------------------------
    rgb      : (B, H, W, 3)  uint8   – base_camera RGB  (H×W = sensor resolution)
    hand_rgb : (B, h, w, 3)  uint8   – hand_camera RGB  (may differ from base resolution)
    depth    : (B, H, W, 1)  float32 – base_camera depth in metres
    seg      : (B, H, W, 1)  int32   – base_camera segmentation IDs
    joints   : (B, D)        float32 – flattened agent + extra state vector
    """

    def __init__(self, env) -> None:
        from mani_skill.envs.sapien_env import BaseEnv
        self._base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        sample_obs, _ = env.reset()
        new_obs = self.observation(sample_obs)
        self._base_env.update_obs_space(new_obs)

    def observation(self, observation: dict) -> dict:
        from mani_skill.utils import common

        sensor_data  = observation.pop("sensor_data",  {})
        sensor_param = observation.pop("sensor_param", {}) or {}

        base       = sensor_data.get("base_camera",  {})
        hand       = sensor_data.get("hand_camera",  {})
        base_param = sensor_param.get("base_camera", {})

        ret = {}

        # ── base_camera: RGB, depth, segmentation ─────────────────────────────
        base_rgb = base.get("rgb")
        if base_rgb is not None:
            ret["rgb"] = base_rgb           # (B, H, W, 3)

        base_depth = base.get("depth")
        if base_depth is not None:
            ret["depth"] = base_depth       # (B, H, W, 1) float32, metres

        base_seg = base.get("segmentation")
        if base_seg is not None:
            ret["seg"] = base_seg           # (B, H, W, 1) int32, object IDs

        # ── base_camera: intrinsic K (3×3) and cam-to-world (4×4) ────────────
        # These are constant for a fixed camera; stored per step so callers can
        # grab frame-0 values for point-cloud reprojection.
        intrinsic_cv = base_param.get("intrinsic_cv")
        cam2world_gl = base_param.get("cam2world_gl")
        if intrinsic_cv is not None:
            ret["cam_intrinsic"] = intrinsic_cv   # (B, 3, 3)
        if cam2world_gl is not None:
            ret["cam2world"]     = cam2world_gl   # (B, 4, 4)

        # ── hand_camera: RGB only (stored separately; may differ in resolution) ─
        hand_rgb = hand.get("rgb")
        if hand_rgb is not None:
            ret["hand_rgb"] = hand_rgb      # (B, h, w, 3)

        # ── Joints: flatten agent + extra ─────────────────────────────────────
        extra_agent = {}
        for key in ["extra", "agent"]:
            if key in observation:
                extra_agent[key] = observation.pop(key)
        ret["joints"] = common.flatten_state_dict(
            extra_agent, use_torch=True, device=self._base_env.device
        )

        return ret


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
    num_train_data=1000,
    sensor_width=832,
    sensor_height=480,
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
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_delta_pos",
        render_mode="all",
        sim_backend="gpu",
        reward_mode="normalized_dense",
        sensor_configs={"base_camera": {"width": sensor_width, "height": sensor_height}},
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

    env_rgb = SensorDataCollectWrapper(env_rgb)

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
        rgbList, hand_rgbList, depthList, segList, jointsList, actList, rewList, succList, doneList = [], [], [], [], [], [], [], [], []

        # Reset of both environments with the same seed for synchronization
        seed = episode
        obs_state, _ = env_state.reset(seed=seed)
        obs_rgb, _ = env_rgb.reset(seed=seed)

        for t in range(episode_timeout):
            rgbList.append(obs_rgb['rgb'].cpu().numpy())
            hand_rgbList.append(obs_rgb['hand_rgb'].cpu().numpy())
            depthList.append(obs_rgb['depth'].cpu().numpy())
            segList.append(obs_rgb['seg'].cpu().numpy())
            jointsList.append(obs_rgb['joints'].cpu().numpy())

            with torch.no_grad():
                for key, value in obs_state.items():
                    obs_state[key] = value.to(device)
                action = agent.get_action(obs_state, deterministic=True)

            obs_state, reward_state, term_state, trunc_state, info_state = env_state.step(action)
            obs_rgb, reward_rgb, term_rgb, trunc_rgb, info_rgb = env_rgb.step(action)

            rewList.append(reward_rgb.cpu().numpy())
            succList.append(info_rgb['success'].cpu().numpy().astype(int))
            actList.append(action.cpu().numpy())
            done = torch.logical_or(term_rgb, trunc_rgb)
            doneList.append(done.cpu().numpy().astype(int))

        # rgb:      (T, B, 480, 832, 3) uint8   – base_camera RGB
        # hand_rgb: (T, B, 128, 128, 3) uint8   – hand_camera RGB
        # depth:    (T, B, 480, 832, 1) float32 – base_camera depth in metres
        # seg:      (T, B, 480, 832, 1) int32   – base_camera segmentation IDs
        DATA = {'rgb':      np.array(rgbList),
                'hand_rgb': np.array(hand_rgbList),
                'depth':    np.array(depthList),
                'seg':      np.array(segList),
                'joints':   np.array(jointsList),
                'action':   np.array(actList),
                'reward':   np.array(rewList),
                'success':  np.array(succList),
                'done':     np.array(doneList)}

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
            DATA = {
                'rgb':     episode['rgb'][:,     trajectory_num, :, :, :],
                'joints':  episode['joints'][:,  trajectory_num, :],
                'action':  episode['action'][:,  trajectory_num, :],
                'reward':  episode['reward'][:,  trajectory_num],
                'success': episode['success'][:, trajectory_num],
                'done':    episode['done'][:,     trajectory_num],
            }
            for optional_key in ('hand_rgb', 'depth', 'seg'):
                if optional_key in episode:
                    DATA[optional_key] = episode[optional_key][:, trajectory_num, :, :, :]

            file_path = f'{save_dir_unbatched}/train_data_{traj_cnt}.npz'
            np.savez(file_path, **DATA)

            traj_cnt += 1


def _unproject_rgbd_to_world(rgb: np.ndarray,
                             depth: np.ndarray,
                             K: np.ndarray,
                             cam2world_gl: np.ndarray,
                             far_clip: float = 5.0,
                             stride: int = 1):
    """Convert a single RGB-D frame into (points_world, colors) for viewer overlay.

    Inputs
    ------
    rgb           : (H, W, 3) uint8
    depth         : (H, W)    float32, **metres**
    K             : (3, 3)    OpenCV intrinsic
    cam2world_gl  : (4, 4)    OpenGL-convention camera-to-world transform
                              (cam_x=+right, cam_y=+up, cam_z=+back), which is
                              what ManiSkill exposes as `sensor_param.cam2world_gl`.
    far_clip      : drop points with depth > far_clip (metres)
    stride        : pixel stride for sub-sampling (1 keeps every pixel)

    Returns
    -------
    pts_world : (N, 3) float32
    colors    : (N, 3) float32 in [0, 1]
    """
    if stride > 1:
        rgb   = rgb[::stride, ::stride]
        depth = depth[::stride, ::stride]
    H, W = depth.shape

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    z = depth.astype(np.float32)
    valid = (z > 1e-4) & (z < far_clip) & np.isfinite(z)

    x_cam = (uu - cx) * z / fx
    y_cam = (vv - cy) * z / fy
    z_cam = z

    # OpenCV → OpenGL camera frame: flip Y and Z
    x_gl =  x_cam
    y_gl = -y_cam
    z_gl = -z_cam

    pts_cam = np.stack([x_gl, y_gl, z_gl], axis=-1).reshape(-1, 3)
    mask = valid.reshape(-1)
    pts_cam = pts_cam[mask]
    colors = (rgb.reshape(-1, 3)[mask].astype(np.float32) / 255.0)

    homo = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)], axis=1)
    pts_world = (cam2world_gl.astype(np.float32) @ homo.T).T[:, :3]
    return pts_world.astype(np.float32), colors.astype(np.float32)


def _load_id_to_name_map(traj_h5_path):
    """Read `id_poses.attrs` (seg_id → actor/link name) from a trajectory
    `.h5` produced by RBSRecordEpisode. Returns `{int: str}` or `{}`.
    """
    import h5py
    mapping: dict = {}
    try:
        with h5py.File(traj_h5_path, "r") as f:
            for key in f.keys():
                g = f[key]
                if isinstance(g, h5py.Group) and "id_poses" in g.keys():
                    idp = g["id_poses"]
                    for attr, val in idp.attrs.items():
                        try:
                            mapping[int(attr)] = str(val)
                        except Exception:
                            pass
                    if mapping:
                        return mapping
    except Exception:
        pass
    return mapping


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple:
    """Minimal HSV→RGB conversion (no scipy dependency). h,s,v in [0,1]."""
    import colorsys
    return colorsys.hsv_to_rgb(h % 1.0, s, v)


def _color_for_sid(sid: int) -> np.ndarray:
    """Deterministic per-sid colour via golden-ratio hue rotation. Gives
    visually distinct hues for any number of sids (hashes colour-collision
    free for ≤24 ids in practice)."""
    GOLDEN = 0.6180339887498949
    # Start hue slightly offset so sid=1 isn't pure red (robot body) and
    # manipulable objects (typically sid>=16) get warm distinctive hues.
    h = (0.07 + sid * GOLDEN) % 1.0
    r, g, b = _hsv_to_rgb(h, 0.78, 0.95)
    return np.asarray([r, g, b], dtype=np.float32)


def _install_pointflow_color_override(env_vis, per_point_colors: np.ndarray) -> None:
    """Monkey-patch `update_pointflow_visualization` so that, instead of
    dev_wjj's hard-coded z-gradient, it draws each point with a fixed
    pre-computed colour.

    This is VIEWER-ONLY; the on-disk data layout is unchanged.
    """
    import types

    assert per_point_colors.ndim == 2 and per_point_colors.shape[1] == 3
    env_vis._pointflow_per_point_colors = per_point_colors.astype(np.float32)

    def _patched(self):
        if not self.visualize_pointflow or self._pointflow_frames is None:
            return
        if self.num_envs != 1:
            return
        try:
            viewer = getattr(self.base_env, "_viewer", None)
            if viewer is None:
                return
            frame_idx = min(self._pointflow_frame_idx,
                            self._pointflow_frames.shape[0] - 1)
            pts = self._pointflow_frames[frame_idx].astype(np.float32)
            colors = self._pointflow_per_point_colors
            if colors.shape[0] != pts.shape[0]:
                # fall back to dev_wjj's z-gradient if sizes ever disagree
                z = pts[:, 2]
                zn = (z - self._pointflow_zmin) / self._pointflow_zrange
                colors = np.stack(
                    [zn, 1.0 - zn, np.zeros_like(zn)], axis=1
                ).astype(np.float32)

            if (
                self._pointflow_point_list_id is not None
                and hasattr(viewer, "remove_3d_point_list")
            ):
                try:
                    viewer.remove_3d_point_list(self._pointflow_point_list_id)
                except Exception:
                    self._pointflow_point_list_id = None
            elif hasattr(viewer, "clear_3d_point_lists"):
                try:
                    viewer.clear_3d_point_lists()
                except Exception:
                    pass

            try:
                self._pointflow_point_list_id = viewer.add_3d_point_list(pts, colors)
            except Exception:
                viewer.add_3d_point_list(pts, colors)
                self._pointflow_point_list_id = None
        except Exception:
            return

    env_vis.update_pointflow_visualization = types.MethodType(_patched, env_vis)


def _shrink_pointflow_to_match_devwjj_look(
    env_vis,
    pointflow_npy_path: str,
    exclude_name_patterns: list,
    keep_name_patterns: list,
    stride: int,
    cam_radius_clip: float,
    color_mode: str = "z",
) -> None:
    """Filter RBSRecordEpisode._pointflow_frames so the overlay matches
    dev_wjj's visual density on MIKASA-Robo envs — by dropping pixels
    that belong to scene-infrastructure actors (ground, table, etc.),
    which otherwise form a 33 m-wide "sheet" in `far=100m` cameras.

    Why `seg != 0` is NOT "foreground-only" in MIKASA-Robo
    ------------------------------------------------------
    In ManiSkill/MIKASA-Robo every actor and link gets a per-scene seg id,
    including `actor:table-workspace` and `actor:ground`. Empirically on
    ShellGameTouch-v0 these two account for **~95 %** of the 832×480
    pixels; the robot + mugs only occupy a few percent. `seg == 0` is
    empty. So the classical "mask out background" trick has to look up
    the actor name via `id_poses.attrs` and drop infrastructure by name.

    Visual impact
    -------------
    On ShellGameTouch-v0 the default pattern list turns a "red-green
    curtain" into a ~20 k-point cloud concentrated on the robot arm,
    the shell cups, and the red ball — matching the visual density of
    dev_wjj's canonical demo envs (PickCube etc.).

    This is VIEWER-ONLY filtering. The on-disk `.npy` / `.b2nd` files
    are NOT modified — downstream pipelines still see the exact same
    dev_wjj payload.

    Filters (applied in order)
    --------------------------
    1. Name-based include/exclude via `id_poses.attrs` in the sibling
       trajectory `.h5`. Pixels whose anchor-frame seg id maps to a name
       matching any `exclude_name_patterns` substring are dropped. If
       `keep_name_patterns` is non-empty, only pixels matching those are
       kept (takes priority over exclude).
    2. `stride`           — keep every Nth pixel on both axes.
    3. `cam_radius_clip`  — drop points farther than this many metres
       from the frame-0 camera origin (disabled at 0).
    """
    frames = getattr(env_vis, "_pointflow_frames", None)
    if frames is None:
        return

    from pathlib import Path
    import re
    p = Path(pointflow_npy_path)
    T, N, _ = frames.shape

    try:
        raw = np.load(p, allow_pickle=False, mmap_mode="r")
    except Exception as exc:
        print(f"  [pointflow filter] cannot mmap {p}: {exc}; skipping filter")
        return
    if raw.ndim != 4 or raw.shape[-1] != 3 or raw.shape[0] != T:
        print(f"  [pointflow filter] unexpected .npy shape {raw.shape}; skipping filter")
        return
    _, H, W, _ = raw.shape
    if H * W != N:
        print(f"  [pointflow filter] H*W ({H*W}) != N ({N}); skipping filter")
        return

    keep = np.ones(N, dtype=bool)

    # Anchor frame idx from filename (scene_point_flow_refXXXXX.npy).
    # IMPORTANT: our vendored `convert_camera_depths.track_anchor_file_exact`
    # uses `seg[anchor_idx]` as the pixel→sid mapping for the resulting
    # (T,H,W,3) tensor (dev_wjj used a hard-coded seg[0], which silently
    # drops any actor that wasn't visible at t=0, e.g. ShellGameTouch's
    # mugs). So the viewer-side filter MUST use seg[anchor_idx] too;
    # otherwise a mug pixel that was 'background' at frame 0 gets
    # classified by its frame-0 sid (==16/17 = table/ground) and wrongly
    # dropped, even though the tracker put the mug's tracked motion
    # there. Matching the tracker's reference frame keeps data and
    # overlay consistent.
    m = re.search(r"scene_point_flow_ref(\d+)", p.stem)
    anchor_idx = int(m.group(1)) if m else 0

    # 1) Name-pattern based filter via id_poses.attrs
    do_name_filter = bool(exclude_name_patterns) or bool(keep_name_patterns)
    if do_name_filter:
        seg_path = p.parent / "seg.npy"
        traj_h5_candidates = sorted(p.parent.glob("traj_*.h5"))
        traj_h5 = traj_h5_candidates[0] if traj_h5_candidates else None
        if seg_path.exists() and traj_h5 is not None:
            seg = np.load(seg_path, allow_pickle=False)  # (T,H,W)
            a = min(anchor_idx, seg.shape[0] - 1)
            if seg.ndim >= 3 and seg.shape[-2:] == (H, W):
                seg_flat = seg[a].reshape(-1).astype(np.int64)
                id_to_name = _load_id_to_name_map(traj_h5)

                def _match(name: str, patterns: list) -> bool:
                    return any(pat.lower() in name.lower() for pat in patterns)

                kept_ids, dropped_ids = [], []
                if keep_name_patterns:
                    pass_mask = np.zeros(N, dtype=bool)
                    for sid, name in id_to_name.items():
                        if _match(name, keep_name_patterns):
                            pass_mask |= (seg_flat == sid)
                            kept_ids.append((sid, name))
                        else:
                            dropped_ids.append((sid, name))
                else:
                    pass_mask = np.ones(N, dtype=bool)
                    for sid, name in id_to_name.items():
                        if _match(name, exclude_name_patterns):
                            pass_mask &= (seg_flat != sid)
                            dropped_ids.append((sid, name))
                        else:
                            kept_ids.append((sid, name))

                before = int(keep.sum())
                keep &= pass_mask
                print(f"  [pointflow filter] name-filter on anchor frame {a}:")
                print(f"    kept ({len(kept_ids)} ids): "
                      + ", ".join(f"{sid}:{nm.split('/')[-1]}" for sid, nm in kept_ids[:12])
                      + (" ..." if len(kept_ids) > 12 else ""))
                print(f"    dropped ({len(dropped_ids)} ids): "
                      + ", ".join(f"{sid}:{nm.split('/')[-1]}" for sid, nm in dropped_ids[:12])
                      + (" ..." if len(dropped_ids) > 12 else ""))
                print(f"    result: {int(keep.sum())}/{before} pixels")
            else:
                print(f"  [pointflow filter] seg.npy shape {seg.shape} "
                      f"not aligned to (T,{H},{W}); skipping name filter")
        else:
            miss = []
            if not seg_path.exists(): miss.append("seg.npy")
            if traj_h5 is None: miss.append("traj_*.h5")
            print(f"  [pointflow filter] missing {'/'.join(miss)} in {p.parent}; "
                  f"skipping name filter")

    # 2) pixel-grid stride downsample
    if stride > 1:
        grid_mask = np.zeros((H, W), dtype=bool)
        grid_mask[::stride, ::stride] = True
        before = int(keep.sum())
        keep &= grid_mask.reshape(-1)
        print(f"  [pointflow filter] stride={stride}: "
              f"{int(keep.sum())}/{before} pixels after grid subsample")

    # 3) optional radius clip around frame-0 camera origin
    if cam_radius_clip > 0.0:
        cam_poses_path = p.parent / "cam_poses.npy"
        if cam_poses_path.exists():
            cp = np.load(cam_poses_path, allow_pickle=False)
            if cp.ndim == 3 and cp.shape[1:] == (4, 4):
                cam_origin = cp[0, :3, 3].astype(np.float32)
                d2 = np.linalg.norm(
                    frames[0] - cam_origin[None, :], axis=1
                )
                rad_mask = d2 <= float(cam_radius_clip)
                before = int(keep.sum())
                keep &= rad_mask
                print(f"  [pointflow filter] cam_radius_clip={cam_radius_clip}m: "
                      f"{int(keep.sum())}/{before} pixels within radius of "
                      f"cam origin ({cam_origin.tolist()})")

    if keep.sum() == N:
        return

    if keep.sum() == 0:
        print("  [pointflow filter] all points filtered out; disabling overlay")
        env_vis.visualize_pointflow = False
        env_vis._pointflow_frames = None
        return

    filtered = frames[:, keep, :].astype(np.float32)
    env_vis._pointflow_frames = filtered
    z_all = filtered[:, :, 2]
    env_vis._pointflow_zmin = float(z_all.min())
    zmax = float(z_all.max())
    env_vis._pointflow_zrange = max(zmax - env_vis._pointflow_zmin, 1e-8)
    print(f"  [pointflow filter] active points: {filtered.shape[1]} "
          f"(was {N}); world-z=[{env_vis._pointflow_zmin:.3f},"
          f"{env_vis._pointflow_zmin + env_vis._pointflow_zrange:.3f}]")

    # ── per-point colour assignment ───────────────────────────────────────
    mode = (color_mode or "z").lower()
    if mode == "z":
        return  # keep dev_wjj's built-in z-gradient; no override installed

    n_pts = int(keep.sum())
    per_pt = None

    if mode == "id":
        # Colour every point by the seg id it had at the anchor frame.
        seg_path = p.parent / "seg.npy"
        traj_h5_candidates = sorted(p.parent.glob("traj_*.h5"))
        traj_h5 = traj_h5_candidates[0] if traj_h5_candidates else None
        if seg_path.exists() and traj_h5 is not None:
            seg = np.load(seg_path, allow_pickle=False)
            a = min(anchor_idx, seg.shape[0] - 1)
            seg_flat = seg[a].reshape(-1).astype(np.int64)
            seg_kept = seg_flat[keep]
            unique = sorted({int(s) for s in np.unique(seg_kept)})
            sid_to_color = {sid: _color_for_sid(sid) for sid in unique}
            per_pt = np.zeros((n_pts, 3), dtype=np.float32)
            for sid, col in sid_to_color.items():
                per_pt[seg_kept == sid] = col
            id2name = _load_id_to_name_map(traj_h5)
            print(f"  [pointflow colour] mode=id  palette:")
            for sid in unique:
                nm = id2name.get(sid, f"sid={sid}").split('/')[-1]
                c = sid_to_color[sid]
                print(f"    sid={sid:3d} {nm:30s}  rgb=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})")
        else:
            print(f"  [pointflow colour] mode=id but seg.npy/traj_*.h5 missing; "
                  f"falling back to z-gradient")
            return

    elif mode == "rgb":
        # Colour every point by the rgb pixel it was unprojected from, at
        # the anchor frame. Decoded from rgb.mp4 via OpenCV.
        try:
            import cv2
        except Exception as exc:
            print(f"  [pointflow colour] mode=rgb needs opencv ({exc}); "
                  f"falling back to z-gradient")
            return
        rgb_path = p.parent / "rgb.mp4"
        if not rgb_path.exists():
            print(f"  [pointflow colour] mode=rgb needs rgb.mp4; falling back")
            return
        cap = cv2.VideoCapture(str(rgb_path))
        frame_bgr = None
        idx = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if idx == anchor_idx:
                frame_bgr = fr
                break
            idx += 1
        cap.release()
        if frame_bgr is None:
            print(f"  [pointflow colour] couldn't read anchor frame {anchor_idx} "
                  f"from rgb.mp4; falling back to z-gradient")
            return
        if frame_bgr.shape[:2] != (H, W):
            frame_bgr = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_NEAREST)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        per_pt = frame_rgb.reshape(-1, 3)[keep].astype(np.float32)
        print(f"  [pointflow colour] mode=rgb  source={rgb_path.name} frame={anchor_idx}")

    else:
        print(f"  [pointflow colour] unknown color_mode={color_mode!r}; "
              f"using z-gradient")
        return

    if per_pt is not None:
        _install_pointflow_color_override(env_vis, per_pt)


class _LivePCDViewerOverlay:
    """Manages a single dynamic 3D-point-list on the SAPIEN viewer.

    Each call to :meth:`update` removes the previous point list (if any) and
    adds a fresh one. Mirrors the bookkeeping used by dev_wjj's
    `update_pointflow_visualization`, but driven by a live reprojected
    RGB-D frame instead of a pre-computed `pointflow.npy`.
    """

    def __init__(self, viewer):
        self.viewer = viewer
        self.prev_id = None

    def update(self, pts_world: np.ndarray, colors: np.ndarray):
        if self.viewer is None or pts_world.size == 0:
            return
        try:
            if self.prev_id is not None and hasattr(self.viewer, "remove_3d_point_list"):
                try:
                    self.viewer.remove_3d_point_list(self.prev_id)
                except Exception:
                    self.prev_id = None
            elif hasattr(self.viewer, "clear_3d_point_lists"):
                try:
                    self.viewer.clear_3d_point_lists()
                except Exception:
                    pass
            try:
                self.prev_id = self.viewer.add_3d_point_list(pts_world, colors)
            except Exception:
                self.viewer.add_3d_point_list(pts_world, colors)
                self.prev_id = None
        except Exception:
            self.prev_id = None


def collect_single_episode_with_visualization(
    env_id: str = "ShellGameTouch-v0",
    checkpoint_path: str = None,
    seed: int = 0,
    count: int = 1,
    save_dir: Optional[str] = None,
    viewer_camera: str = "render_camera",
    fps: int = 16,
    sensor_width: int = 832,
    sensor_height: int = 480,
    pointflow_npy: Optional[str] = None,
    pointflow_exclude_names: Optional[list] = None,
    pointflow_keep_names: Optional[list] = None,
    pointflow_stride: int = 1,
    pointflow_cam_radius_clip: float = 0.0,
    pointflow_color_mode: str = "z",
    live_pcd: bool = False,
    live_pcd_stride: int = 2,
    live_pcd_far_clip: float = 5.0,
    postprocess_camera_data: bool = True,
    postprocess_workers: int = 8,
    postprocess_delete_npy: bool = False,
) -> None:
    """Collect `count` episodes with oracle RL rollout, display them live in the
    SAPIEN interactive viewer, and save trajectories in the **exact same layout
    as dev_wjj's `RBSRecordEpisode`**.

    Backend policy
    --------------
    This matches dev_wjj's `rbs_replay_trajectory.py` behaviour:

        if args.vis and args.sim_backend not in CPU_SIM_BACKENDS:
            ...
            args.sim_backend = "physx_cpu"

    → env_vis is **forced** to `sim_backend="physx_cpu", num_envs=1`, because
    SAPIEN's batched-render system on `physx_gpu` does not allow dynamic
    scene modifications (add_3d_point_list etc.) required for pointflow
    / live-PCD overlay.

    Visualization modes
    -------------------
    • `--pointflow-npy <path>` → replay a pre-computed pointflow array in the
      viewer (identical to dev_wjj behaviour). Typical workflow: run once
      without this flag to get `scene_point_flow_ref00000.npy`, then re-run
      with this flag pointing at it. Note: `postprocess_delete_npy` defaults
      to False precisely so Pass 1's .npy survives to be loaded by Pass 2.
    • `--live-pcd`            → unproject the current base_camera RGB-D frame
      to world coordinates every step and overlay it in the viewer. No
      pre-computed pointflow required.
    • (neither)                → plain SAPIEN viewer, no point overlay.

    Output layout (dev_wjj RBSRecordEpisode-compatible)
    ---------------------------------------------------
        <save_dir>/
          <timestamp>.h5               # main trajectory h5, one `traj_i` per episode
          <timestamp>.json             # env_info + episodes[*]
          camera_data/
            traj_<i>/
              rgb.mp4                  # T frames, H×W
              depth_video.npy          # (T, H, W) float16, metres, far-plane sentinel-handled
              seg.npy                  # (T, H, W) int
              cam_poses.npy            # (T+1, 4, 4)
              cam_intrinsics.npy       # (T+1, 3, 3)
              traj_<i>.h5              # per-episode copy with id_poses / id_geometry_meta
        # (if --postprocess-camera-data) additionally:
              scene_point_flow_ref*.b2nd / depth.b2nd / seg.b2nd
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wl_state, episode_timeout = env_info(env_id)
    wl_vis, _                 = env_info(env_id)

    # Defensive: the hook is a no-op under physx_cpu (no batched render system
    # to race against), but installing it is harmless.
    _install_viewer_hook()

    # ── state env (agent inference only, no viewer) ───────────────────────────
    #   kept on GPU sim to match how the checkpoint was trained.
    print("[1/3] Building state env (GPU, for agent inference)...")
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

    # ── visual env (SAPIEN viewer + raw sensor_data for RBSRecordEpisode) ────
    # RBSRecordEpisode MUST sit directly on top of the ManiSkill env (with at
    # most FlattenActionSpaceWrapper in between) — it reads the raw
    # `observation["sensor_data"]["base_camera"]` / `sensor_param` structure
    # to extract rgb/depth/seg and camera poses for the camera_data/ export.
    # Therefore we deliberately do NOT apply SensorDataCollectWrapper or
    # ManiSkillVectorEnv on env_vis: those would flatten / rebatch the obs
    # and break the export path.
    # NOTE (backend string): dev_wjj uses `"physx_cpu"`; the public ManiSkill
    # release only understands `"cpu"` / `"cuda"` / `"gpu"` / `"auto"`. We pass
    # the portable `"cpu"` here so the same file works against either install.
    # `"cpu"` resolves to the exact same physx-cpu backend internally.
    #
    # NOTE (render_mode): we use "rgb_array" (NOT "human"), matching dev_wjj's
    # `rbs_replay_trajectory.py` default. `RBSRecordEpisode.capture_image()`
    # calls `env.render()` every step to grab a frame for `rgb.mp4`; with
    # render_mode="human" that returns a sapien.utils.Viewer object and
    # crashes the recorder. The interactive SAPIEN GUI is opened separately
    # by calling `env.base_env.render_human()` in the step loop below, which
    # lazily creates the viewer and refreshes it — independent of render_mode.
    print(f"[2/3] Building visual env (sim_backend=cpu, render_mode=rgb_array) "
          f"viewer_camera='{viewer_camera}' ...")
    with _override_human_render_camera(env_id, viewer_camera):
        env_vis = gym.make(
            env_id, num_envs=1,
            obs_mode="rgb+depth+segmentation",
            render_mode="rgb_array",   # for RBSRecordEpisode.capture_image()
            sim_backend="cpu",         # forced for GUI + point overlay
            reward_mode="normalized_dense",
            sensor_configs={"base_camera": {"width": sensor_width, "height": sensor_height}},
        )
    # Filter out MIKASA's text-overlay wrappers on the vis path. They
    # override `render()` with a `cv2.putText(...)` call that is very picky
    # about the dtype / memory-layout of the frame returned by
    # `env.render()` and crashes on many mani_skill / sapien / opencv
    # version combinations. They do NOT affect the underlying simulation,
    # observations, or reward — only the text annotation on rendered frames,
    # which we deliberately do NOT want on the rgb.mp4 captured by
    # `RBSRecordEpisode.capture_image()` anyway. The agent runs on env_state
    # (un-filtered) so any info-tracking done by these wrappers is preserved
    # where it actually matters.
    def _is_render_overlay_wrapper(cls) -> bool:
        name = cls.__name__
        return (
            "Render" in name
            or "Debug" in name
            or name.endswith("InfoWrapper")
        )
    wl_vis_filtered = [
        (W, kw) for (W, kw) in wl_vis if not _is_render_overlay_wrapper(W)
    ]
    _skipped = [W.__name__ for (W, _) in wl_vis if _is_render_overlay_wrapper(W)]
    if _skipped:
        print(f"  skipping vis-path render-overlay wrappers: {_skipped}")
    for WC, kw in wl_vis_filtered:
        env_vis = WC(env_vis, **kw)
    if isinstance(env_vis.action_space, gym.spaces.Dict):
        env_vis = FlattenActionSpaceWrapper(env_vis)

    # Default output directory (dev_wjj-style).
    if save_dir is None:
        save_dir = os.path.join("data", "MIKASA-Robo", "vis", env_id)
    os.makedirs(save_dir, exist_ok=True)

    # Whether pointflow-replay overlay is requested and viable.
    # dev_wjj disables it internally if backend is GPU, but we already force
    # physx_cpu above, so it's always eligible here if the path is given.
    visualize_pointflow = pointflow_npy is not None

    # Trajectory name without backend suffix; dev_wjj writes
    # `<ori_traj_name>.<obs_mode>.<control_mode>.<sim_backend>.h5`. We match that.
    import time as _time
    _ts = _time.strftime("%Y%m%d_%H%M%S")
    _ori_traj_name = f"traj_seed{seed}_{_ts}"
    # dev_wjj uses env.unwrapped.backend.sim_backend (a dev_wjj-only attribute);
    # fall back to deriving from gpu_sim_enabled so the same code works against
    # the public ManiSkill release too.
    _base_unwrapped = env_vis.unwrapped
    _sim_backend_suffix = "physx_cpu" if not getattr(
        _base_unwrapped, "gpu_sim_enabled", False
    ) else "physx_gpu"
    _suffix = "{}.{}.{}".format(
        _base_unwrapped.obs_mode,
        _base_unwrapped.control_mode,
        _sim_backend_suffix,
    )
    _traj_name = f"{_ori_traj_name}.{_suffix}"

    env_vis = RBSRecordEpisode(
        env_vis,
        output_dir=save_dir,
        trajectory_name=_traj_name,
        save_trajectory=True,
        save_video=True,
        info_on_video=False,
        save_on_reset=True,          # flush previous traj at every reset
        clean_on_close=True,
        record_reward=True,
        record_env_state=True,
        record_id_poses=True,        # enables id_poses recording (per-seg-id pose)
        record_id_mesh_info=True,    # enables id_geometry_meta recording
        visualize_pointflow=visualize_pointflow,
        pointflow_npy_path=pointflow_npy,
        postprocess_camera_data=postprocess_camera_data,
        postprocess_workers=postprocess_workers,
        postprocess_delete_npy=postprocess_delete_npy,
        video_fps=fps,
        source_type="rl",
        source_desc=f"MIKASA-Robo RL rollout (oracle agent, env={env_id})",
    )

    # Default exclude list — drops `actor:ground` / `actor:table-workspace`
    # and similar scene-infrastructure actors so the overlay concentrates on
    # the robot + manipulable objects. See the function docstring for why
    # `seg != 0` is NOT a valid "foreground-only" filter in MIKASA-Robo.
    default_excl = [
        "actor:ground",
        "ground-plane",
        "actor:table",        # catches table-workspace, table-top, etc.
        "scene-builder",
    ]
    _excl = list(pointflow_exclude_names) if pointflow_exclude_names is not None else default_excl
    _keep = list(pointflow_keep_names) if pointflow_keep_names is not None else []
    _need_filter = bool(_excl) or bool(_keep) or pointflow_stride > 1 or pointflow_cam_radius_clip > 0
    _need_color  = (pointflow_color_mode or "z").lower() != "z"
    if visualize_pointflow and pointflow_npy is not None and (_need_filter or _need_color):
        _shrink_pointflow_to_match_devwjj_look(
            env_vis,
            pointflow_npy_path=pointflow_npy,
            exclude_name_patterns=_excl,
            keep_name_patterns=_keep,
            stride=pointflow_stride,
            cam_radius_clip=pointflow_cam_radius_clip,
            color_mode=pointflow_color_mode,
        )

    # ── load oracle agent ─────────────────────────────────────────────────────
    print(f"[3/3] Loading checkpoint: {checkpoint_path}")
    agent = AgentStateOnly(env_state).to(device)
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()

    # Viewer handle for live-PCD overlay; fetched lazily after first reset
    # because the SAPIEN viewer is only constructed during the first render().
    live_overlay: Optional[_LivePCDViewerOverlay] = None

    print(f"\nRunning {count} episode(s)  env={env_id}  seed_base={seed}  timeout={episode_timeout} steps/ep")
    print(f"  base_camera resolution : {sensor_width}×{sensor_height}  fps={fps}")
    print(f"  sim_backend(env_vis)   : physx_cpu  (forced for GUI / point overlay)")
    if visualize_pointflow:
        active = (
            env_vis._pointflow_frames.shape[1]
            if getattr(env_vis, "_pointflow_frames", None) is not None
            else 0
        )
        print(f"  point overlay mode     : pointflow-replay  ({pointflow_npy})")
        print(f"    active points/frame  : {active}  "
              f"(exclude={_excl or 'none'}, "
              f"keep={_keep or 'all-except-exclude'}, "
              f"stride={pointflow_stride}, "
              f"cam_radius_clip={pointflow_cam_radius_clip}, "
              f"color_mode={pointflow_color_mode})")
    elif live_pcd:
        print(f"  point overlay mode     : live RGB-D reprojection "
              f"(stride={live_pcd_stride}, far_clip={live_pcd_far_clip}m)")
    else:
        print("  point overlay mode     : off")
    print("SAPIEN viewer controls: right-drag=rotate  scroll=zoom  mid-drag=pan\n")

    for ep in range(count):
        ep_seed = seed + ep
        obs_state, _ = env_state.reset(seed=ep_seed)
        obs_vis, _   = env_vis.reset(seed=ep_seed)

        for t in tqdm(range(episode_timeout),
                      desc=f"ep {ep+1}/{count} (seed={ep_seed})", leave=False):
            with torch.no_grad():
                for k in obs_state:
                    obs_state[k] = obs_state[k].to(device)
                action = agent.get_action(obs_state, deterministic=True)

            obs_state, _, _, _, _ = env_state.step(action)
            action_np = action.detach().cpu().numpy()
            obs_vis, reward, term, trunc, info = env_vis.step(action_np)

            # Point-cloud overlay in the SAPIEN viewer.
            if visualize_pointflow:
                # RBSRecordEpisode already advances its internal frame index
                # in step(); we just ask it to push the current frame.
                if hasattr(env_vis, "update_pointflow_visualization"):
                    env_vis.update_pointflow_visualization()
            elif live_pcd:
                base_env = env_vis.unwrapped
                viewer = getattr(base_env, "_viewer", None)
                if viewer is not None:
                    if live_overlay is None:
                        live_overlay = _LivePCDViewerOverlay(viewer)
                    try:
                        sd = obs_vis["sensor_data"]["base_camera"]
                        sp = obs_vis["sensor_param"]["base_camera"]
                        import torch as _torch
                        def _np(x):
                            if isinstance(x, _torch.Tensor):
                                return x.detach().cpu().numpy()
                            return np.asarray(x)
                        rgb_f  = _np(sd["rgb"])[0]             # (H, W, 3) uint8
                        depth_f = _np(sd["depth"])[0, ..., 0]  # (H, W)    float32, metres
                        K_f    = _np(sp["intrinsic_cv"])[0]    # (3, 3)
                        c2w_f  = _np(sp["cam2world_gl"])[0]    # (4, 4)
                        pts_w, cols = _unproject_rgbd_to_world(
                            rgb_f.astype(np.uint8),
                            depth_f.astype(np.float32),
                            K_f.astype(np.float32),
                            c2w_f.astype(np.float32),
                            far_clip=live_pcd_far_clip,
                            stride=live_pcd_stride,
                        )
                        live_overlay.update(pts_w, cols)
                    except Exception as exc:
                        # Keep rollout alive; viewer overlay is cosmetic.
                        tqdm.write(f"[live_pcd] overlay update failed: {exc}")

            try:
                env_vis.base_env.render_human()   # refresh the SAPIEN viewer each step
            except RuntimeError as exc:
                if "Modifying a scene" in str(exc):
                    tqdm.write("\n[ERROR] Scene modification not allowed; stopping early.")
                    break
                raise

            # stop early if the episode actually ended (term or trunc)
            if bool(np.asarray(term).any()) or bool(np.asarray(trunc).any()):
                break

        # Success log (optional, after episode)
        try:
            if "success" in info:
                succ = np.asarray(info["success"] if not hasattr(info["success"], "cpu")
                                  else info["success"].cpu().numpy()).any()
                print(f"  ep {ep+1}/{count} seed={ep_seed}: success={bool(succ)}")
        except Exception:
            pass

    # close() triggers the final flush + postprocess chain
    env_state.close()
    env_vis.close()

    print(f"\nAll {count} episode(s) written to: {save_dir}")
    print(f"  trajectory h5 : {os.path.join(save_dir, _traj_name + '.h5')}")
    print(f"  trajectory json: {os.path.join(save_dir, _traj_name + '.json')}")
    print(f"  camera_data/   : {os.path.join(save_dir, 'camera_data')}")


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
    """If True, run the dev_wjj-compatible RBSRecordEpisode path: GUI preview
    in SAPIEN + HDF5/JSON/camera_data/ output, with physx_cpu + num_envs=1
    (forced, matches dev_wjj's `rbs_replay_trajectory.py` vis auto-switch)."""
    seed: int = 0
    """First random seed for the visualized episode (visualize mode only).
    Successive episodes use seed+1, seed+2, ..."""
    count: int = 1
    """Number of episodes to roll out and record in visualize mode."""
    save_vis_episode: bool = False
    """If True and --visualize is set, save the trajectory under
    <path_to_save_data>/MIKASA-Robo/vis/<env_id>/. If False it still writes
    under a default data/MIKASA-Robo/vis/<env_id>/ so the h5/json/camera_data
    pipeline has somewhere to land; this flag only controls whether the
    user-requested path_to_save_data prefix is used."""
    viewer_camera: str = "render_camera"
    """Camera perspective for the SAPIEN viewer.
    'render_camera' (default) uses the env's built-in render camera (wide,
    high-resolution, fixed diagonal view).  Pass a sensor camera name such as
    'base_camera' to start the viewer from that perspective instead."""
    fps: int = 16
    """Frame-rate for the exported RGB video (visualize mode only, default 16)."""
    sensor_width: int = 832
    """Width in pixels for the base_camera sensor (visualize + batch mode, default 832)."""
    sensor_height: int = 480
    """Height in pixels for the base_camera sensor (visualize + batch mode, default 480)."""
    # ── dev_wjj-style SAPIEN point-cloud overlay (visualize mode only) ───────
    pointflow_npy: Optional[str] = None
    """Path to a pre-computed scene-pointflow .npy (shape (T,H,W,3) or (T,N,3)).
    When provided, the SAPIEN viewer overlays this pointflow frame-by-frame,
    exactly like dev_wjj's `rbs_replay_trajectory.py --pointflow_npy`.
    Leave None on the first pass (no pointflow exists yet); after that first
    pass produces `camera_data/traj_*/scene_point_flow_ref00000.npy` you can
    re-run with this flag pointing at it to reproduce dev_wjj's replay look.
    Tip: combine Pass 2 with `--no-postprocess-camera-data` so it doesn't
    compress/delete the already-good Pass-1 .npy files."""
    pointflow_exclude_names: Optional[List[str]] = None
    """Substring patterns (case-insensitive) matched against every
    `id_poses.attrs` name (e.g. 'actor:table-workspace',
    'link:panda_wristcam/panda_link0'). Any seg id whose name contains ANY
    pattern is DROPPED from the viewer overlay. Default (when left None):
    ['actor:ground', 'ground-plane', 'actor:table', 'scene-builder'].

    This filter is essential on MIKASA-Robo envs. Unlike dev_wjj demo envs,
    MIKASA-Robo's `table-workspace` + `ground` actors cover ~95 % of the
    base_camera pixels (empirically verified on ShellGameTouch-v0); without
    dropping them the overlay is a 33 m-wide 'sky sheet' that occludes the
    tabletop. `seg != 0` is NOT equivalent: in MIKASA-Robo every pixel
    belongs to some actor (seg == 0 is empty).

    Pass `--pointflow-exclude-names ""` (empty list) to see the raw
    dev_wjj payload with all H×W pixels."""
    pointflow_keep_names: Optional[List[str]] = None
    """Substring patterns (case-insensitive). If provided (non-empty),
    ONLY pixels whose seg id name matches any pattern are kept — this
    takes priority over --pointflow-exclude-names. Example to visualise
    just the robot + red ball:
        --pointflow-keep-names 'link:panda_wristcam' 'actor:red_ball' """
    pointflow_stride: int = 1
    """Pixel-grid stride for additional downsampling of the pointflow overlay
    (1 = no downsample, 2 = 1/4 of points, 4 = 1/16 of points).
    Useful when foreground-only still leaves too many points to feel 3D."""
    pointflow_cam_radius_clip: float = 0.0
    """If >0, drop points whose Euclidean distance from the frame-0 camera
    origin exceeds this many metres. Disabled by default (0 = keep all)."""
    pointflow_color_mode: str = "z"
    """How to colour the SAPIEN overlay points. One of:
        'z'   — dev_wjj's built-in red→green gradient over world-z
                (faithful to `RBSRecordEpisode.update_pointflow_visualization`).
        'id'  — fixed palette colour per segmentation id at the anchor frame,
                so mug-L/mug-C/mug-R (and every panda link) get distinct
                colours. Great for telling the three shells apart.
        'rgb' — per-pixel colour sampled from the anchor frame of rgb.mp4;
                produces a photographic 3-D point cloud.
    Modes 'id' and 'rgb' install a one-shot monkey-patch on the wrapper's
    `update_pointflow_visualization` so the colours are stable across
    frames (dev_wjj recomputes z-gradient every frame). Viewer-only; no
    on-disk data changes."""
    live_pcd: bool = False
    """If True (and --pointflow-npy is not set), overlay a live per-step point
    cloud reconstructed by unprojecting base_camera RGB-D into world
    coordinates each step. Useful for the first pass before you have a
    pointflow .npy to replay."""
    live_pcd_stride: int = 2
    """Pixel stride when unprojecting depth → point cloud for the live overlay
    (default 2 = 1/4 of the pixels; smaller = denser & slower)."""
    live_pcd_far_clip: float = 5.0
    """Discard depth values (in metres) larger than this before unprojection."""
    postprocess_camera_data: bool = True
    """Whether to run dev_wjj's post-processing chain (convert_camera_depths +
    flow_compress + point_compress + seg_compress) on each freshly saved
    camera_data/traj_<i>/ directory. Produces
    `scene_point_flow_ref*.b2nd` / `depth.b2nd` / `seg.b2nd`."""
    postprocess_workers: int = 8
    """Workers passed to convert_camera_depths during per-traj postprocess."""
    postprocess_delete_npy: bool = False
    """Whether to delete intermediate .npy files after compression.
    Default False so that `scene_point_flow_ref*.npy` survives pass 1 and can
    be fed back via --pointflow-npy for a pass-2 replay overlay. Set True
    only if you're done visualizing and want to save disk space."""


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
        save_dir = os.path.join(path_to_save_data, "MIKASA-Robo", "vis", ENV_ID)
        collect_single_episode_with_visualization(
            env_id=ENV_ID,
            checkpoint_path=ckpt_path,
            seed=args.seed,
            count=args.count,
            save_dir=save_dir,
            viewer_camera=args.viewer_camera,
            fps=args.fps,
            sensor_width=args.sensor_width,
            sensor_height=args.sensor_height,
            pointflow_npy=args.pointflow_npy,
            pointflow_exclude_names=args.pointflow_exclude_names,
            pointflow_keep_names=args.pointflow_keep_names,
            pointflow_stride=args.pointflow_stride,
            pointflow_cam_radius_clip=args.pointflow_cam_radius_clip,
            pointflow_color_mode=args.pointflow_color_mode,
            live_pcd=args.live_pcd,
            live_pcd_stride=args.live_pcd_stride,
            live_pcd_far_clip=args.live_pcd_far_clip,
            postprocess_camera_data=args.postprocess_camera_data,
            postprocess_workers=args.postprocess_workers,
            postprocess_delete_npy=args.postprocess_delete_npy,
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
                    sensor_width=args.sensor_width,
                    sensor_height=args.sensor_height,
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
#
# (A) dev_wjj RBSRecordEpisode-compatible rollout + GUI (NO point overlay yet):
#   python3 mikasa_robo_suite/dataset_collectors/get_mikasa_robo_datasets.py \
#       --env-id ShellGameTouch-v0 --ckpt-dir . --visualize --count 5 --seed 0
#
# (B) Same as (A), but with a live RGB-D-reprojection point cloud overlaid in SAPIEN:
#   python3 mikasa_robo_suite/dataset_collectors/get_mikasa_robo_datasets.py \
#       --env-id ShellGameTouch-v0 --ckpt-dir . --visualize --live-pcd \
#       --live-pcd-stride 2 --live-pcd-far-clip 4.0
#
# (C) Replay a pre-computed pointflow overlay (dev_wjj style). Two-pass workflow.
#     Pointflow is loaded in RBSRecordEpisode.__init__ (BEFORE any rollout),
#     so you need a PREVIOUS run's output as input.
#
#     Pass 1 — collect data + run postprocess (produces scene_point_flow_ref*.npy):
#   rm -rf data/MIKASA-Robo/vis/ShellGameTouch-v0/camera_data
#   python3 mikasa_robo_suite/dataset_collectors/get_mikasa_robo_datasets.py \
#       --env-id ShellGameTouch-v0 --ckpt-dir . --visualize --count 1 --seed 0
#
#     The postprocess step produces several `scene_point_flow_ref<ANCHOR:05d>.npy`
#     files per episode (one per anchor frame, e.g. 00000, 00022, 00045, ...).
#     Because `--postprocess-delete-npy` defaults to False, the .npy files SURVIVE
#     postprocess (alongside the compressed .b2nd copies) and are ready for Pass 2.
#     Sanity check — this glob should list 5 files with NO `.anchor.` in the names:
#       ls data/MIKASA-Robo/vis/ShellGameTouch-v0/camera_data/traj_0/scene_point_flow_ref*.npy \
#          | grep -v anchor
#     If it prints nothing, you accidentally passed `--postprocess-delete-npy`
#     (which removes the .npy after compression); rerun Pass 1 without that flag.
#
#     Pass 2 — replay in GUI with overlay (DO NOT delete camera_data first!):
#     look for the startup log line
#       "INFO - Loaded pointflow for visualization: ...shape=(T, H, W, 3)"
#     which confirms the file was found. Without that line the overlay is off.
#     Pass 2 auto-skips postprocess so it won't touch Pass-1's .npy files.
#   python3 mikasa_robo_suite/dataset_collectors/get_mikasa_robo_datasets.py \
#       --env-id ShellGameTouch-v0 --ckpt-dir . --visualize --count 1 --seed 0 \
#       --no-postprocess-camera-data \
#       --pointflow-npy data/MIKASA-Robo/vis/ShellGameTouch-v0/camera_data/traj_0/scene_point_flow_ref00000.npy
#
#     (The pass-2 run will write its fresh output into traj_0_p<pid>_..._c<hash>/
#      because traj_0/ is already taken by pass 1. Harmless.)
#
# (D) Change viewer startup camera (e.g. to robot's base_camera perspective):
#   python3 mikasa_robo_suite/dataset_collectors/get_mikasa_robo_datasets.py \
#       --env-id ShellGameTouch-v0 --ckpt-dir . --visualize --viewer-camera base_camera
#
# (E) Disable the post-processing chain (no b2nd compression / scene_point_flow generation):
#   python3 mikasa_robo_suite/dataset_collectors/get_mikasa_robo_datasets.py \
#       --env-id ShellGameTouch-v0 --ckpt-dir . --visualize --no-postprocess-camera-data
#
# (F) Full batch collection (original behaviour, unchanged):
#   python3 mikasa_robo_suite/dataset_collectors/get_mikasa_robo_datasets.py \
#       --env-id ShellGameTouch-v0 --path-to-save-data data --ckpt-dir . --num-train-data 1000
#
# Output layout of --visualize mode (identical to dev_wjj's RBSRecordEpisode):
#   data/MIKASA-Robo/vis/<env_id>/
#     <traj_name>.h5                 # main trajectory h5
#     <traj_name>.json               # env_info + episode meta
#     camera_data/traj_<i>/
#       rgb.mp4, depth_video.npy, seg.npy,
#       cam_poses.npy, cam_intrinsics.npy, traj_<i>.h5
#     (if --postprocess-camera-data, additionally)
#       scene_point_flow_ref*.b2nd, depth.b2nd, seg.b2nd