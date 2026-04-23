import copy
import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import gymnasium as gym
import h5py
import numpy as np
import sapien.physx as physx
import sapien.render as sapien_render
import torch

from mani_skill import get_commit_info
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.geometry.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from mani_skill.utils import common, gym_utils, sapien_utils
from mani_skill.utils.io_utils import dump_json
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.structs.types import Array
from mani_skill.utils.visualization.misc import (
    images_to_video,
    put_info_on_image,
    tile_images,
)
from mani_skill.utils.wrappers import CPUGymWrapper

# NOTE (stao): The code for record.py is quite messy and perhaps confusing as it is trying to support both recording on CPU and GPU seamlessly
# and handle partial resets. It works but can be claned up a lot.


def parse_env_info(env: gym.Env):
    # spec can be None if not initialized from gymnasium.make
    env = env.unwrapped
    if env.spec is None:
        return None
    if hasattr(env.spec, "_kwargs"):
        # gym<=0.21
        env_kwargs = env.spec._kwargs
    else:
        # gym>=0.22
        env_kwargs = env.spec.kwargs
    return dict(
        env_id=env.spec.id,
        env_kwargs=env_kwargs,
    )


def temp_deep_print_shapes(x, prefix=""):
    if isinstance(x, dict):
        for k in x:
            temp_deep_print_shapes(x[k], prefix=prefix + "/" + k)
    else:
        print(prefix, x.shape)


def clean_trajectories(h5_file: h5py.File, json_dict: dict, prune_empty_action=True):
    """Clean trajectories by renaming and pruning trajectories in place.

    After cleanup, trajectory names are consecutive integers (traj_0, traj_1, ...),
    and trajectories with empty action are pruned.

    Args:
        h5_file: raw h5 file
        json_dict: raw JSON dict
        prune_empty_action: whether to prune trajectories with empty action
    """
    json_episodes = json_dict["episodes"]
    assert len(h5_file) == len(json_episodes)

    # Assumes each trajectory is named "traj_{i}"
    prefix_length = len("traj_")
    ep_ids = sorted([int(x[prefix_length:]) for x in h5_file.keys()])

    new_json_episodes = []
    new_ep_id = 0

    for i, ep_id in enumerate(ep_ids):
        traj_id = f"traj_{ep_id}"
        ep = json_episodes[i]
        assert ep["episode_id"] == ep_id
        new_traj_id = f"traj_{new_ep_id}"

        if prune_empty_action and ep["elapsed_steps"] == 0:
            del h5_file[traj_id]
            continue

        if new_traj_id != traj_id:
            ep["episode_id"] = new_ep_id
            h5_file[new_traj_id] = h5_file[traj_id]
            del h5_file[traj_id]

        new_json_episodes.append(ep)
        new_ep_id += 1

    json_dict["episodes"] = new_json_episodes


@dataclass
class Step:
    state: np.ndarray
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    done: np.ndarray
    env_episode_ptr: np.ndarray
    """points to index in above data arrays where current episode started (any data before should already be flushed)"""

    success: np.ndarray = None
    fail: np.ndarray = None


class RBSRecordEpisode(gym.Wrapper):
    """Record trajectories or videos for episodes. You generally should always apply this wrapper last, particularly if you include
    observation wrappers which modify the returned observations. The only wrappers that may go after this one is any of the vector env
    interface wrappers that map the maniskill env to a e.g. gym vector env interface.

    Trajectory data is saved with two files, the actual data in a .h5 file via H5py and metadata in a JSON file of the same basename.

    Each JSON file contains:

    - `env_info` (dict): task (also known as environment) information, which can be used to initialize the task
    - `env_id` (str): task id
    - `max_episode_steps` (int)
    - `env_kwargs` (dict): keyword arguments to initialize the task. **Essential to recreate the environment.**
    - `episodes` (list[dict]): episode information
    - `source_type` (Optional[str]): a simple category string describing what process generated the trajectory data. ManiSkill official datasets will usually write one of "human", "motionplanning", or "rl" at the moment.
    - `source_desc` (Optional[str]): a longer explanation of how the data was generated.

    The episode information (the element of `episodes`) includes:

    - `episode_id` (int): a unique id to index the episode
    - `reset_kwargs` (dict): keyword arguments to reset the task. **Essential to reproduce the trajectory.**
    - `control_mode` (str): control mode used for the episode.
    - `elapsed_steps` (int): trajectory length
    - `info` (dict): information at the end of the episode.

    With just the meta data, you can reproduce the task the same way it was created when the trajectories were collected as so:

    ```python
    env = gym.make(env_info["env_id"], **env_info["env_kwargs"])
    episode = env_info["episodes"][0] # picks the first
    env.reset(**episode["reset_kwargs"])
    ```

    Each HDF5 demonstration dataset consists of multiple trajectories. The key of each trajectory is `traj_{episode_id}`, e.g., `traj_0`.

    Each trajectory is an `h5py.Group`, which contains:

    - actions: [T, A], `np.float32`. `T` is the number of transitions.
    - terminated: [T], `np.bool_`. It indicates whether the task is terminated or not at each time step.
    - truncated: [T], `np.bool_`. It indicates whether the task is truncated or not at each time step.
    - env_states: [T+1, D], `np.float32`. Environment states. It can be used to set the environment to a certain state via `env.set_state_dict`. However, it may not be enough to reproduce the trajectory.
    - success (optional): [T], `np.bool_`. It indicates whether the task is successful at each time step. Included if task defines success.
    - fail (optional): [T], `np.bool_`. It indicates whether the task is in a failure state at each time step. Included if task defines failure.
    - obs (optional): [T+1, D] observations.

    Note that env_states is in a dictionary form (and observations may be as well depending on obs_mode), where it is formatted as a dictionary of lists. For example, a typical environment state looks like this:

    ```python
    env_state = env.get_state_dict()
    \"\"\"
    env_state = {
    "actors": {
        "actor_id": [...numpy_actor_state...],
        ...
    },
    "articulations": {
        "articulation_id": [...numpy_articulation_state...],
        ...
    }
    }
    \"\"\"
    ```
    In the trajectory file env_states will be the same structure but each value/leaf in the dictionary will be a sequence of states representing the state of that particular entity in the simulation over time.

    In practice it is may be more useful to use slices of the env_states data (or the observations data), which can be done with

    ```python
    import mani_skill.trajectory.utils as trajectory_utils
    env_states = trajectory_utils.dict_to_list_of_dicts(env_states)
    # now env_states[i] is the same as the data env.get_state_dict() returned at timestep i
    i = 10
    env_state_i = trajectory_utils.index_dict(env_states, i)
    # now env_state_i is the same as the data env.get_state_dict() returned at timestep i
    ```

    Args:
        env: the environment to record
        output_dir: output directory
        save_trajectory: whether to save trajectory
        trajectory_name: name of trajectory file (.h5). Use timestamp if not provided.
        save_video: whether to save video
        info_on_video: whether to write data about reward, action, and data in the info object to the video. The first video frame is generally the result
            of the first env.reset() (visualizing the first observation). Text is written on frames after that, showing the action taken to get to that
            environment state and reward.
        save_on_reset: whether to save the previous trajectory (and video of it if `save_video` is True) automatically when resetting.
            Not that for environments simulated on the GPU (to leverage fast parallel rendering) you must
            set `max_steps_per_video` to a fixed number so that every `max_steps_per_video` steps a video is saved. This is
            required as there may be partial environment resets which makes it ambiguous about how to save/cut videos.
        save_video_trigger: a function that takes the current number of elapsed environment steps and outputs a bool. If output is True, will start saving that timestep to the video.
        max_steps_per_video: how many steps can be recorded into a single video before flushing the video. If None this is not used. A internal step counter is maintained to do this.
            If the video is flushed at any point, the step counter is reset to 0.
        clean_on_close: whether to rename and prune trajectories when closed.
            See `clean_trajectories` for details.
        record_reward: whether to record the reward in the trajectory data
        record_env_state: whether to record the environment state in the trajectory data
        video_fps (int): The FPS of the video to generate if save_video is True
        render_substeps (bool): Whether to render substeps for video. This is captures an image of the environment after each physics step. This runs slower but generates more image frames
            per environment step which when coupled with a higher video FPS can yield a smoother video.
        avoid_overwriting_video (bool): If true, the wrapper will iterate over possible video names to avoid overwriting existing videos in the output directory. Useful for resuming training runs.
        source_type (Optional[str]): a word to describe the source of the actions used to record episodes (e.g. RL, motionplanning, teleoperation)
        source_desc (Optional[str]): A longer description describing how the demonstrations are collected
    """

    def __init__(
        self,
        env: BaseEnv,
        output_dir: str,
        save_trajectory: bool = True,
        trajectory_name: Optional[str] = None,
        save_video: bool = True,
        info_on_video: bool = False,
        save_on_reset: bool = True,
        save_video_trigger: Optional[Callable[[int], bool]] = None,
        max_steps_per_video: Optional[int] = None,
        clean_on_close: bool = True,
        record_reward: bool = True,
        record_env_state: bool = True,
        video_fps: int = 30,
        render_substeps: bool = False,
        avoid_overwriting_video: bool = False,
        source_type: Optional[str] = None,
        source_desc: Optional[str] = None,
        record_id_poses: bool = False,
        record_id_mesh_info: bool = False,
        visualize_pointflow: bool = False,
        pointflow_npy_path: Optional[str] = None,
        postprocess_camera_data: bool = False,
        postprocess_workers: int = 19,
        postprocess_delete_npy: bool = True,
    ) -> None:
        super().__init__(env)

        self.output_dir = Path(output_dir)
        if save_trajectory or save_video:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_fps = video_fps
        self._elapsed_record_steps = 0
        self._episode_id = -1
        self._video_id = -1
        self._video_steps = 0
        self._closed = False

        self.save_video_trigger = save_video_trigger

        self._trajectory_buffer: Step = None

        self.max_steps_per_video = max_steps_per_video
        self.max_episode_steps = gym_utils.find_max_episode_steps_value(env)

        self.save_on_reset = save_on_reset
        self.save_trajectory = save_trajectory
        if self.base_env.num_envs > 1 and save_video:
            assert (
                max_steps_per_video is not None
            ), "On GPU parallelized environments, \
                there must be a given max steps per video value in order to flush videos in order \
                to avoid issues caused by partial resets. If your environment does not do partial \
                resets you may set max_steps_per_video equal to the max_episode_steps"
        self.clean_on_close = clean_on_close
        self.record_reward = record_reward
        self.record_env_state = record_env_state
        self.record_id_poses = record_id_poses
        self.record_id_mesh_info = record_id_mesh_info
        self.id_to_name = dict()
        self._id_pose_buffer = dict()
        self.id_geometry_meta = dict()
        self._already_warned_about_id_pose = False
        if self.record_id_poses:
            self._build_id_mapping()
        if self.record_id_mesh_info:
            if len(self.id_to_name) == 0:
                self._build_id_mapping()
            self._build_id_geometry_metadata()
        if self.save_trajectory:
            if not trajectory_name:
                trajectory_name = time.strftime("%Y%m%d_%H%M%S")

            self._h5_file = h5py.File(self.output_dir / f"{trajectory_name}.h5", "w")

            # Use a separate json to store non-array data
            self._json_path = self._h5_file.filename.replace(".h5", ".json")
            self._json_data = dict(
                env_info=parse_env_info(self.env),
                commit_info=get_commit_info(),
                episodes=[],
            )
            if self._json_data["env_info"] is not None:
                self._json_data["env_info"][
                    "max_episode_steps"
                ] = self.max_episode_steps
            if source_type is not None:
                self._json_data["source_type"] = source_type
            if source_desc is not None:
                self._json_data["source_desc"] = source_desc
        self._save_video = save_video
        self.info_on_video = info_on_video
        self.render_images = []
        self.video_nrows = int(np.sqrt(self.unwrapped.num_envs))
        self._avoid_overwriting_video = avoid_overwriting_video

        self._already_warned_about_state_dict_inconsistency = False

        # pointflow visualization related
        self.visualize_pointflow = visualize_pointflow
        self.pointflow_npy_path = pointflow_npy_path
        self._pointflow_frames: Optional[np.ndarray] = None
        self._pointflow_frame_idx: int = 0
        self._pointflow_point_list_id = None
        self._pointflow_zmin = 0.0
        self._pointflow_zrange = 1.0
        self._pointflow_overlay_disabled_reason: Optional[str] = None
        self.postprocess_camera_data = postprocess_camera_data
        self.postprocess_workers = int(postprocess_workers)
        self.postprocess_delete_npy = postprocess_delete_npy
        self._postprocess_executor = (
            ThreadPoolExecutor(
                max_workers=max(1, self.postprocess_workers),
                thread_name_prefix="camera-postprocess",
            )
            if self.postprocess_camera_data
            else None
        )
        self._postprocess_futures = []
        if self.visualize_pointflow:
            if self.base_env.gpu_sim_enabled:
                self.visualize_pointflow = False
                self._pointflow_overlay_disabled_reason = (
                    "Pointflow overlay is disabled on GPU simulation backend because "
                    "batched render systems do not allow dynamic scene add/remove operations."
                )
                logger.warning(self._pointflow_overlay_disabled_reason)
            
            self._load_pointflow_for_visualization()

        # check if wrapped env is already wrapped by a CPU gym wrapper
        cur_env = self.env
        self.cpu_wrapped_env = False
        while cur_env is not None:
            if isinstance(cur_env, CPUGymWrapper):
                self.cpu_wrapped_env = True
                break
            if hasattr(cur_env, "env"):
                cur_env = cur_env.env
            else:
                break

        self.render_substeps = render_substeps
        if self.render_substeps:
            _original_after_simulation_step = self.base_env._after_simulation_step

            def wrapped_after_simulation_step():
                _original_after_simulation_step()
                if self.save_video:
                    if self.base_env.gpu_sim_enabled:
                        self.base_env.scene._gpu_fetch_all()
                    self.render_images.append(self.capture_image())

            self.base_env._after_simulation_step = wrapped_after_simulation_step

    def _postprocess_camera_dir(self, camera_dir: Path):
        if not self.postprocess_camera_data:
            return
        done_flag = camera_dir / ".postprocess_done"
        lock_file = camera_dir / ".postprocess_lock"
        if done_flag.exists():
            return
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            return

        try:
            scripts_root = Path(__file__).resolve().parent
            convert_script = scripts_root / "convert_camera_depths.py"
            flow_script = scripts_root / "flow_compress.py"
            point_script = scripts_root / "point_compress.py"
            seg_script = scripts_root / "seg_compress.py"

            convert_cmd = [
                sys.executable,
                str(convert_script),
                str(camera_dir),
                "--workers",
                "1",
            ]
            flow_cmd = [
                sys.executable,
                str(flow_script),
                "compress",
                "--out_dir",
                str(camera_dir),
            ]
            if self.postprocess_delete_npy:
                flow_cmd.append("--delete_npy")
            point_cmd = [
                sys.executable,
                str(point_script),
                "--mode",
                "compress",
                "--seg_dir",
                str(camera_dir),
                "--delete-existing",
            ]
            seg_cmd = [
                sys.executable,
                str(seg_script),
                "compress",
                "--seg-dir",
                str(camera_dir),
                "--method",
                "b2nd",
            ]
            if self.postprocess_delete_npy:
                seg_cmd.append("--delete-source")

            # Dependency graph (per-traj postprocess):
            #   convert_camera_depths  ──► writes  scene_point_flow_ref*.npy
            #                                       depth_video_int16mm_dt.b2nd
            #                                  │
            #                                  ▼
            #   flow_compress  ── reads scene_point_flow_ref*.npy → mp4 (+ optional delete .npy)
            #
            #   point_compress ── independent of convert/flow
            #   seg_compress   ── independent of convert/flow
            #
            # → Phase 1 (parallel): convert + point + seg
            # → Phase 2 (serial, after Phase 1 convert):  flow
            # Set RBS_PER_TRAJ_PARALLEL=0 to fall back to the legacy serial chain
            # (convert → flow → point → seg).
            phase1 = [
                ("convert_camera_depths", convert_cmd),
                ("point_compress",        point_cmd),
                ("seg_compress",          seg_cmd),
            ]
            phase2 = [
                ("flow_compress",         flow_cmd),
            ]

            if os.environ.get("RBS_PER_TRAJ_PARALLEL", "1") != "0":
                with ThreadPoolExecutor(
                    max_workers=len(phase1),
                    thread_name_prefix="rbs-per-traj",
                ) as pool:
                    futures = {
                        pool.submit(subprocess.run, c, check=True): name
                        for name, c in phase1
                    }
                    for fut in futures:
                        try:
                            fut.result()
                        except Exception as sub_exc:
                            logger.warning(
                                f"per-traj postprocess step {futures[fut]} "
                                f"failed for {camera_dir}: {sub_exc}"
                            )
                            raise
                # Phase 2: flow_compress depends on convert's *.npy outputs
                for name, c in phase2:
                    try:
                        subprocess.run(c, check=True)
                    except Exception as sub_exc:
                        logger.warning(
                            f"per-traj postprocess step {name} "
                            f"failed for {camera_dir}: {sub_exc}"
                        )
                        raise
            else:
                for _, c in phase1 + phase2:
                    subprocess.run(c, check=True)

            done_flag.write_text("ok\n")
        except Exception as e:
            logger.warning(f"camera_dir postprocess failed for {camera_dir}: {e}")
        finally:
            try:
                lock_file.unlink(missing_ok=True)
            except Exception:
                pass

    def _submit_postprocess_camera_dir(self, camera_dir: Path):
        if not self.postprocess_camera_data:
            return
        if self._postprocess_executor is None:
            self._postprocess_camera_dir(camera_dir)
            return
        fut = self._postprocess_executor.submit(self._postprocess_camera_dir, camera_dir)
        self._postprocess_futures.append(fut)
        # 清理已完成 future，避免列表无限增长
        self._postprocess_futures = [f for f in self._postprocess_futures if not f.done()]

    @property
    def num_envs(self):
        return self.base_env.num_envs

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    @property
    def save_video(self):
        if not self._save_video:
            return False
        if self.save_video_trigger is not None:
            return self.save_video_trigger(self._elapsed_record_steps)
        else:
            return self._save_video

    def capture_image(self, infos=None):
        img = self.env.render()
        img = common.to_numpy(img)
        if len(img.shape) == 3:
            img = img[None]
        if infos is not None:
            for i in range(len(img)):
                info_item = {
                    k: v if np.size(v) == 1 else v[i] for k, v in infos.items()
                }
                img[i] = put_info_on_image(img[i], info_item)
        if len(img.shape) > 3:
            if len(img) == 1:
                img = img[0]
            else:
                img = tile_images(img, nrows=self.video_nrows)
        return img

    def _load_pointflow_for_visualization(self):
        if self.pointflow_npy_path is None:
            logger.warning(
                "visualize_pointflow=True but pointflow_npy_path is None; disabling pointflow visualization."
            )
            self.visualize_pointflow = False
            return
        try:
            path = Path(self.pointflow_npy_path)
            arr = np.load(path, allow_pickle=False)
            if arr.ndim == 4 and arr.shape[-1] == 3:
                t, h, w, _ = arr.shape
                frames = arr.reshape((t, h * w, 3)).astype(np.float32)
            elif arr.ndim == 3 and arr.shape[-1] == 3:
                frames = arr.astype(np.float32)
            else:
                raise ValueError(
                    f"Unsupported pointflow shape {arr.shape}, expected (T,H,W,3) or (T,N,3)"
                )

            # optional camera->world transform if cam_poses.npy exists in same folder
            cam_poses_path = path.parent / "cam_poses.npy"
            if cam_poses_path.exists():
                cam_poses = np.load(cam_poses_path, allow_pickle=False)
                if cam_poses.ndim == 3 and cam_poses.shape[1:] == (4, 4):
                    max_t = min(frames.shape[0], cam_poses.shape[0])
                    for i in range(max_t):
                        pts = frames[i]
                        homo = np.concatenate(
                            [pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1
                        )
                        world = (cam_poses[i] @ homo.T).T[:, :3]
                        frames[i] = world.astype(np.float32)
                else:
                    logger.warning(
                        f"cam_poses.npy has unexpected shape {cam_poses.shape}, skipping cam2world transform."
                    )

            z_all = frames[:, :, 2]
            self._pointflow_zmin = float(z_all.min())
            zmax = float(z_all.max())
            self._pointflow_zrange = max(zmax - self._pointflow_zmin, 1e-8)
            self._pointflow_frames = frames
            self._pointflow_frame_idx = 0
            self._pointflow_point_list_id = None
            logger.info(
                f"Loaded pointflow for visualization: {path}, shape={frames.shape}"
            )
        except Exception as e:
            logger.warning(f"Failed to load pointflow npy {self.pointflow_npy_path}: {e}")
            self.visualize_pointflow = False
            self._pointflow_frames = None

    def update_pointflow_visualization(self):
        if not self.visualize_pointflow or self._pointflow_frames is None:
            return
        if self.num_envs != 1:
            return
        try:
            viewer = getattr(self.base_env, "_viewer", None)
            if viewer is None:
                return
            frame_idx = min(self._pointflow_frame_idx, self._pointflow_frames.shape[0] - 1)
            pts = self._pointflow_frames[frame_idx].astype(np.float32)
            z = pts[:, 2]
            z_norm = (z - self._pointflow_zmin) / self._pointflow_zrange
            colors = np.stack(
                [z_norm, 1 - z_norm, np.zeros_like(z_norm)], axis=1
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

    def _build_id_mapping(self):
        self.id_to_name = dict()
        env = self.base_env
        try:
            for actor_name, actor in env.scene.actors.items():
                try:
                    seg_id = common.to_numpy(actor.per_scene_id)
                    if np.size(seg_id) == 0:
                        continue
                    seg_ids = np.array(seg_id).reshape(-1)
                    for i, seg_id in enumerate(seg_ids):
                        seg_id = int(seg_id)
                        if seg_id in self.id_to_name:
                            continue
                        if self.num_envs > 1 and len(seg_ids) == self.num_envs:
                            self.id_to_name[seg_id] = f"actor:{actor_name}[env{i}]"
                        elif len(seg_ids) == 1:
                            self.id_to_name[seg_id] = f"actor:{actor_name}"
                        else:
                            self.id_to_name[seg_id] = f"actor:{actor_name}[{i}]"
                except Exception:
                    continue

            for art_name, articulation in env.scene.articulations.items():
                try:
                    for link in articulation.get_links():
                        try:
                            seg_ids = np.array(common.to_numpy(link.per_scene_id)).reshape(-1)
                            if seg_ids.size == 0:
                                continue
                            for i, seg_id in enumerate(seg_ids):
                                seg_id = int(seg_id)
                                if seg_id in self.id_to_name:
                                    continue
                                if seg_ids.size == 1:
                                    self.id_to_name[seg_id] = (
                                        f"link:{art_name}/{link.get_name()}"
                                    )
                                elif self.num_envs > 1 and seg_ids.size == self.num_envs:
                                    self.id_to_name[seg_id] = (
                                        f"link:{art_name}/{link.get_name()}[env{i}]"
                                    )
                                else:
                                    self.id_to_name[seg_id] = (
                                        f"link:{art_name}/{link.get_name()}[{i}]"
                                    )
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception:
            if not self._already_warned_about_id_pose:
                logger.warn(
                    "Failed to build segmentation id mapping for id pose recording."
                )
                self._already_warned_about_id_pose = True

    def _convert_to_serializable(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [self._convert_to_serializable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._convert_to_serializable(v) for k, v in value.items()}
        if isinstance(value, Path):
            return str(value)
        return value

    def _extract_path_candidates(self, obj) -> list[str]:
        candidates = []
        attr_names = [
            "filename",
            "file_name",
            "file_path",
            "filepath",
            "path",
            "uri",
            "asset_path",
            "mesh_file",
            "mesh_path",
            "source_file",
            "usd_path",
            "obj_path",
            "glb_path",
            "gltf_path",
        ]
        for name in attr_names:
            if not hasattr(obj, name):
                continue
            try:
                val = getattr(obj, name)
            except Exception:
                continue
            if isinstance(val, Path):
                candidates.append(str(val))
            elif isinstance(val, str) and len(val) > 0:
                candidates.append(val)
        # de-duplicate while preserving order
        seen = set()
        unique = []
        for p in candidates:
            if p in seen:
                continue
            seen.add(p)
            unique.append(p)
        return unique

    def _extract_collision_shapes_metadata(self, body) -> tuple[list[dict], list[str]]:
        shape_metas = []
        paths = []
        if body is None:
            return shape_metas, paths
        try:
            collision_shapes = body.get_collision_shapes()
        except Exception:
            return shape_metas, paths

        for geom in collision_shapes:
            meta = dict(shape_type=type(geom).__name__)
            if isinstance(geom, physx.PhysxCollisionShapeBox):
                meta["half_size"] = self._convert_to_serializable(
                    np.asarray(common.to_numpy(geom.half_size))
                )
            elif isinstance(geom, physx.PhysxCollisionShapeCapsule):
                meta["radius"] = float(geom.radius)
                meta["half_length"] = float(geom.half_length)
            elif isinstance(geom, physx.PhysxCollisionShapeCylinder):
                meta["radius"] = float(geom.radius)
                meta["half_length"] = float(geom.half_length)
            elif isinstance(geom, physx.PhysxCollisionShapeSphere):
                meta["radius"] = float(geom.radius)
            elif isinstance(
                geom,
                (
                    physx.PhysxCollisionShapeConvexMesh,
                    physx.PhysxCollisionShapeTriangleMesh,
                ),
            ):
                try:
                    meta["scale"] = self._convert_to_serializable(
                        np.asarray(common.to_numpy(geom.scale))
                    )
                except Exception:
                    pass
                try:
                    meta["num_vertices"] = int(len(geom.vertices))
                except Exception:
                    pass
                try:
                    meta["num_triangles"] = int(len(geom.get_triangles()))
                except Exception:
                    pass

            shape_metas.append(meta)
            paths.extend(self._extract_path_candidates(geom))
        return shape_metas, paths

    def _extract_render_shapes_metadata(self, entity) -> tuple[list[dict], list[str]]:
        shape_metas = []
        paths = []
        if entity is None:
            return shape_metas, paths
        try:
            rb_comp = entity.find_component_by_type(sapien_render.RenderBodyComponent)
        except Exception:
            return shape_metas, paths
        if rb_comp is None:
            return shape_metas, paths

        try:
            render_shapes = rb_comp.render_shapes
        except Exception:
            return shape_metas, paths

        for shape in render_shapes:
            meta = dict(shape_type=type(shape).__name__)
            try:
                if hasattr(shape, "half_size"):
                    meta["half_size"] = self._convert_to_serializable(
                        np.asarray(common.to_numpy(shape.half_size))
                    )
                if hasattr(shape, "radius"):
                    meta["radius"] = float(shape.radius)
                if hasattr(shape, "half_length"):
                    meta["half_length"] = float(shape.half_length)
                if hasattr(shape, "scale"):
                    meta["scale"] = self._convert_to_serializable(
                        np.asarray(common.to_numpy(shape.scale))
                    )
                if hasattr(shape, "parts"):
                    parts = shape.parts
                    meta["num_parts"] = int(len(parts))
                    tri_count = 0
                    vert_count = 0
                    for part in parts:
                        try:
                            tri_count += int(len(part.triangles))
                        except Exception:
                            pass
                        try:
                            vert_count += int(len(part.vertices))
                        except Exception:
                            pass
                        paths.extend(self._extract_path_candidates(part))
                    if tri_count > 0:
                        meta["num_triangles"] = tri_count
                    if vert_count > 0:
                        meta["num_vertices"] = vert_count
            except Exception:
                pass
            shape_metas.append(meta)
            paths.extend(self._extract_path_candidates(shape))
        return shape_metas, paths

    def _build_geometry_meta_for_obj(self, obj, name: str, obj_kind: str) -> dict:
        body = None
        entity = None
        try:
            if hasattr(obj, "_bodies") and len(obj._bodies) > 0:
                body = obj._bodies[0]
        except Exception:
            body = None
        try:
            if hasattr(obj, "_objs") and len(obj._objs) > 0:
                first_obj = obj._objs[0]
                if hasattr(first_obj, "entity"):
                    entity = first_obj.entity
                else:
                    entity = first_obj
        except Exception:
            entity = None

        coll_metas, coll_paths = self._extract_collision_shapes_metadata(body)
        render_metas, render_paths = self._extract_render_shapes_metadata(entity)
        all_paths = coll_paths + render_paths
        dedup_paths = []
        seen = set()
        for p in all_paths:
            if p in seen:
                continue
            seen.add(p)
            dedup_paths.append(p)

        return dict(
            name=name,
            object_kind=obj_kind,
            mesh_file_paths=dedup_paths,
            mesh_file_path=(dedup_paths[0] if len(dedup_paths) > 0 else None),
            reconstructable_params=dict(
                collision_shapes=coll_metas,
                render_shapes=render_metas,
            ),
        )

    def _build_id_geometry_metadata(self):
        self.id_geometry_meta = dict()
        env = self.base_env
        try:
            for actor_name, actor in env.scene.actors.items():
                try:
                    seg_ids = np.array(common.to_numpy(actor.per_scene_id)).reshape(-1)
                    if seg_ids.size == 0:
                        continue
                    if seg_ids.size == 1:
                        name = f"actor:{actor_name}"
                    else:
                        name = f"actor:{actor_name}"
                    meta = self._build_geometry_meta_for_obj(actor, name, "actor")
                    for seg_id in seg_ids:
                        self.id_geometry_meta[int(seg_id)] = copy.deepcopy(meta)
                except Exception:
                    continue

            for art_name, articulation in env.scene.articulations.items():
                try:
                    for link in articulation.get_links():
                        try:
                            seg_ids = np.array(common.to_numpy(link.per_scene_id)).reshape(
                                -1
                            )
                            if seg_ids.size == 0:
                                continue
                            link_name = link.get_name()
                            if seg_ids.size == 1:
                                name = f"link:{art_name}/{link_name}"
                            else:
                                name = f"link:{art_name}/{link_name}"
                            meta = self._build_geometry_meta_for_obj(link, name, "link")
                            for seg_id in seg_ids:
                                self.id_geometry_meta[int(seg_id)] = copy.deepcopy(meta)
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception:
            if not self._already_warned_about_id_pose:
                logger.warn(
                    "Failed to build id geometry metadata for mesh path/params recording."
                )
                self._already_warned_about_id_pose = True

    def _get_camera_world_pose(self):
        if not hasattr(self.base_env, "_sensors"):
            return None
        if "base_camera" not in self.base_env._sensors:
            return None
        camera = self.base_env._sensors["base_camera"]
        try:
            cam2world = common.to_numpy(camera.camera.get_model_matrix())
            cam2world = np.array(cam2world)
            if cam2world.ndim == 2:
                cam2world = np.repeat(cam2world[None], self.num_envs, axis=0)
            elif cam2world.ndim == 3 and cam2world.shape[0] == 1 and self.num_envs > 1:
                cam2world = np.repeat(cam2world, self.num_envs, axis=0)
            return cam2world
        except Exception:
            return None

    def _as_pose_batch(self, value: Array, width: int) -> np.ndarray:
        arr = np.array(common.to_numpy(value), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None]
        arr = arr.reshape(-1, width)
        if arr.shape[0] == self.num_envs:
            return arr
        if arr.shape[0] == 1:
            return np.repeat(arr, self.num_envs, axis=0)
        if arr.shape[0] > self.num_envs:
            return arr[: self.num_envs]
        return np.repeat(arr[:1], self.num_envs, axis=0)

    def _world_pose_to_camera_pose(
        self, pos_world: np.ndarray, quat_world: np.ndarray, world2cam: np.ndarray
    ):
        pos_cam, q_cam = self._world_pose_to_camera_pose_batch(
            pos_world[None], quat_world[None], world2cam[None]
        )
        return pos_cam[0], q_cam[0]

    def _world_pose_to_camera_pose_batch(
        self,
        pos_world_batch: np.ndarray,
        quat_world_batch: np.ndarray,
        world2cam_batch: np.ndarray,
    ):
        pos_world_batch = np.asarray(pos_world_batch, dtype=np.float32)
        quat_world_batch = np.asarray(quat_world_batch, dtype=np.float32)
        world2cam_batch = np.asarray(world2cam_batch, dtype=np.float32)

        if pos_world_batch.ndim == 1:
            pos_world_batch = pos_world_batch[None]
        if quat_world_batch.ndim == 1:
            quat_world_batch = quat_world_batch[None]
        if world2cam_batch.ndim == 2:
            world2cam_batch = world2cam_batch[None]

        rot = world2cam_batch[:, :3, :3]
        trans = world2cam_batch[:, :3, 3]
        pos_cam = np.einsum("bij,bj->bi", rot, pos_world_batch) + trans

        q_world = torch.from_numpy(quat_world_batch)
        r_world = quaternion_to_matrix(q_world)
        r_world2cam = torch.from_numpy(rot)
        r_cam = torch.matmul(r_world2cam, r_world)
        q_cam = matrix_to_quaternion(r_cam).cpu().numpy().astype(np.float32)
        return pos_cam.astype(np.float32), q_cam

    def _assign_entity_pose_rows(
        self,
        seg_ids: np.ndarray,
        pos_batch: np.ndarray,
        quat_batch: np.ndarray,
        rows: dict,
        world2cam_batch: Optional[np.ndarray],
    ):
        if seg_ids.size == 0:
            return
        if self.num_envs > 1 and seg_ids.size == self.num_envs:
            for env_i, seg_id in enumerate(seg_ids):
                seg_id = int(seg_id)
                if seg_id not in rows:
                    continue
                pos = pos_batch[min(env_i, pos_batch.shape[0] - 1)]
                quat = quat_batch[min(env_i, quat_batch.shape[0] - 1)]
                rows[seg_id]["position"][env_i] = pos
                rows[seg_id]["quaternion"][env_i] = quat
                if world2cam_batch is not None:
                    cam_pos, cam_quat = self._world_pose_to_camera_pose(
                        pos, quat, world2cam_batch[min(env_i, world2cam_batch.shape[0] - 1)]
                    )
                    rows[seg_id]["camera_position"][env_i] = cam_pos
                    rows[seg_id]["camera_quaternion"][env_i] = cam_quat
            return

        for raw_seg_id in seg_ids:
            seg_id = int(raw_seg_id)
            if seg_id not in rows:
                continue
            rows[seg_id]["position"][:] = pos_batch
            rows[seg_id]["quaternion"][:] = quat_batch
            if world2cam_batch is not None:
                cam_pos, cam_quat = self._world_pose_to_camera_pose_batch(
                    pos_batch, quat_batch, world2cam_batch
                )
                rows[seg_id]["camera_position"][:] = cam_pos
                rows[seg_id]["camera_quaternion"][:] = cam_quat

    def _record_id_poses_snapshot(self):
        if not self.record_id_poses:
            return
        if len(self.id_to_name) == 0:
            self._build_id_mapping()
            if len(self.id_to_name) == 0:
                return

        if len(self._id_pose_buffer) == 0:
            self._id_pose_buffer = {
                seg_id: dict(
                    position=[],
                    quaternion=[],
                    camera_position=[],
                    camera_quaternion=[],
                )
                for seg_id in self.id_to_name.keys()
            }

        rows = self._collect_id_pose_rows()

        for seg_id, row in rows.items():
            self._id_pose_buffer[seg_id]["position"].append(row["position"].copy())
            self._id_pose_buffer[seg_id]["quaternion"].append(
                row["quaternion"].copy()
            )
            self._id_pose_buffer[seg_id]["camera_position"].append(
                row["camera_position"].copy()
            )
            self._id_pose_buffer[seg_id]["camera_quaternion"].append(
                row["camera_quaternion"].copy()
            )

    def _collect_id_pose_rows(self):
        cam2world = self._get_camera_world_pose()
        world2cam_batch = None if cam2world is None else np.linalg.inv(cam2world)
        rows = {
            seg_id: dict(
                position=np.full((self.num_envs, 3), np.nan, dtype=np.float32),
                quaternion=np.full((self.num_envs, 4), np.nan, dtype=np.float32),
                camera_position=np.full((self.num_envs, 3), np.nan, dtype=np.float32),
                camera_quaternion=np.full((self.num_envs, 4), np.nan, dtype=np.float32),
            )
            for seg_id in self._id_pose_buffer.keys()
        }
        env = self.base_env

        for actor_name, actor in env.scene.actors.items():
            try:
                seg_id = common.to_numpy(actor.per_scene_id)
                if np.size(seg_id) == 0:
                    continue
                seg_ids = np.array(seg_id).reshape(-1)
                pose = actor.pose
                pos_batch = self._as_pose_batch(pose.p, width=3)
                quat_batch = self._as_pose_batch(pose.q, width=4)
                self._assign_entity_pose_rows(
                    seg_ids,
                    pos_batch,
                    quat_batch,
                    rows,
                    world2cam_batch,
                )
            except Exception:
                continue

        for art_name, articulation in env.scene.articulations.items():
            try:
                for link in articulation.get_links():
                    try:
                        seg_ids = np.array(common.to_numpy(link.per_scene_id)).reshape(-1)
                        if seg_ids.size == 0:
                            continue
                        pose = link.pose
                        pos_batch = self._as_pose_batch(pose.p, width=3)
                        quat_batch = self._as_pose_batch(pose.q, width=4)
                        self._assign_entity_pose_rows(
                            seg_ids,
                            pos_batch,
                            quat_batch,
                            rows,
                            world2cam_batch,
                        )
                    except Exception:
                        continue
            except Exception:
                continue
        return rows

    def overwrite_last_id_pose_snapshot(self):
        """Overwrite the latest id-pose snapshot with current simulator state.

        Useful when replay changes env state right after reset via `set_state_dict`.
        """
        if not self.record_id_poses:
            return
        if len(self._id_pose_buffer) == 0:
            self._record_id_poses_snapshot()
            return
        rows = self._collect_id_pose_rows()
        for seg_id, row in rows.items():
            if seg_id not in self._id_pose_buffer:
                continue
            if len(self._id_pose_buffer[seg_id]["position"]) == 0:
                self._id_pose_buffer[seg_id]["position"].append(row["position"].copy())
                self._id_pose_buffer[seg_id]["quaternion"].append(row["quaternion"].copy())
                self._id_pose_buffer[seg_id]["camera_position"].append(
                    row["camera_position"].copy()
                )
                self._id_pose_buffer[seg_id]["camera_quaternion"].append(
                    row["camera_quaternion"].copy()
                )
            else:
                self._id_pose_buffer[seg_id]["position"][-1] = row["position"].copy()
                self._id_pose_buffer[seg_id]["quaternion"][-1] = row["quaternion"].copy()
                self._id_pose_buffer[seg_id]["camera_position"][-1] = row[
                    "camera_position"
                ].copy()
                self._id_pose_buffer[seg_id]["camera_quaternion"][-1] = row[
                    "camera_quaternion"
                ].copy()

    def reset(
        self,
        *args,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
        save=True,
        **kwargs,
    ):
        if self.save_on_reset:
            if self.save_video and self.num_envs == 1:
                self.flush_video(save=save)
            # if doing a full reset then we flush all trajectories including incompleted ones
            if self._trajectory_buffer is not None:
                if options is None or "env_idx" not in options:
                    self.flush_trajectory(
                        env_idxs_to_flush=np.arange(self.num_envs), save=save
                    )
                else:
                    self.flush_trajectory(
                        env_idxs_to_flush=common.to_numpy(options["env_idx"]), save=save
                    )

        obs, info = super().reset(*args, seed=seed, options=options, **kwargs)
        if info["reconfigure"]:
            # if we reconfigure, there is the possibility that state dictionary looks different now
            # so trajectory buffer must be wiped
            self._trajectory_buffer = None
        if self.save_trajectory:
            state_dict = self.base_env.get_state_dict()
            action = common.batch(
                self.env.get_wrapper_attr("single_action_space").sample()
            )
            # check if state_dict is consistent
            if not sapien_utils.is_state_dict_consistent(state_dict):
                self.record_env_state = False
                if not self._already_warned_about_state_dict_inconsistency:
                    logger.warn(
                        f"State dictionary is not consistent, disabling recording of environment states for {self.env}"
                    )
                    self._already_warned_about_state_dict_inconsistency = True
            first_step = Step(
                state=None,
                observation=common.to_numpy(common.batch(obs)),
                # note first reward/action etc. are ignored when saving trajectories to disk
                action=common.to_numpy(common.batch(action.repeat(self.num_envs, 0))),
                reward=np.zeros(
                    (
                        1,
                        self.num_envs,
                    ),
                    dtype=float,
                ),
                # terminated and truncated are fixed to be True at the start to indicate the start of an episode.
                # an episode is done when one of these is True otherwise the trajectory is incomplete / a partial episode
                terminated=np.ones((1, self.num_envs), dtype=bool),
                truncated=np.ones((1, self.num_envs), dtype=bool),
                done=np.ones((1, self.num_envs), dtype=bool),
                success=np.zeros((1, self.num_envs), dtype=bool),
                fail=np.zeros((1, self.num_envs), dtype=bool),
                env_episode_ptr=np.zeros((self.num_envs,), dtype=int),
            )
            if self.record_env_state:
                first_step.state = common.to_numpy(common.batch(state_dict))
            env_idx = np.arange(self.num_envs)
            if options is not None and "env_idx" in options:
                env_idx = common.to_numpy(options["env_idx"])
            if self._trajectory_buffer is None:
                # Initialize trajectory buffer on the first episode based on given observation (which should be generated after all wrappers)
                self._trajectory_buffer = first_step
            else:

                def recursive_replace(x, y):
                    if isinstance(x, np.ndarray):
                        x[-1, env_idx] = y[-1, env_idx]
                    else:
                        for k in x.keys():
                            recursive_replace(x[k], y[k])

                # TODO (stao): how do we store states from GPU sim of tasks with objects not in every sub-scene?
                # Maybe we shouldn't?
                if self.record_env_state:
                    recursive_replace(self._trajectory_buffer.state, first_step.state)
                recursive_replace(
                    self._trajectory_buffer.observation, first_step.observation
                )
                recursive_replace(self._trajectory_buffer.action, first_step.action)
                if self.record_reward:
                    recursive_replace(self._trajectory_buffer.reward, first_step.reward)
                recursive_replace(
                    self._trajectory_buffer.terminated, first_step.terminated
                )
                recursive_replace(
                    self._trajectory_buffer.truncated, first_step.truncated
                )
                recursive_replace(self._trajectory_buffer.done, first_step.done)
                if self._trajectory_buffer.success is not None:
                    recursive_replace(
                        self._trajectory_buffer.success, first_step.success
                    )
                if self._trajectory_buffer.fail is not None:
                    recursive_replace(self._trajectory_buffer.fail, first_step.fail)
        if options is not None and "env_idx" in options:
            options["env_idx"] = common.to_numpy(options["env_idx"])
        self.last_reset_kwargs = copy.deepcopy(dict(options=options, **kwargs))
        if seed is not None:
            self.last_reset_kwargs.update(seed=seed)

        if self.record_id_poses:
            if info["reconfigure"]:
                self._build_id_mapping()
            self._id_pose_buffer = dict()
            self._record_id_poses_snapshot()
        if self.record_id_mesh_info and info["reconfigure"]:
            if len(self.id_to_name) == 0:
                self._build_id_mapping()
            self._build_id_geometry_metadata()

        if self.visualize_pointflow and self._pointflow_frames is not None:
            self._pointflow_frame_idx = 0
            self._pointflow_point_list_id = None
        return obs, info

    def step(self, action):
        if self.save_video and self._video_steps == 0:
            # save the first frame of the video here (s_0) instead of inside reset as user
            # may call env.reset(...) multiple times but we want to ignore empty trajectories
            self.render_images.append(self.capture_image())
        obs, rew, terminated, truncated, info = super().step(action)

        if self.save_trajectory:
            state_dict = self.base_env.get_state_dict()
            if self.record_env_state:
                self._trajectory_buffer.state = common.append_dict_array(
                    self._trajectory_buffer.state,
                    common.to_numpy(common.batch(state_dict)),
                )
            self._trajectory_buffer.observation = common.append_dict_array(
                self._trajectory_buffer.observation,
                common.to_numpy(common.batch(obs)),
            )

            self._trajectory_buffer.action = common.append_dict_array(
                self._trajectory_buffer.action,
                common.to_numpy(common.batch(action)),
            )
            if self.record_reward:
                self._trajectory_buffer.reward = common.append_dict_array(
                    self._trajectory_buffer.reward,
                    common.to_numpy(common.batch(rew)),
                )
            self._trajectory_buffer.terminated = common.append_dict_array(
                self._trajectory_buffer.terminated,
                common.to_numpy(common.batch(terminated)),
            )
            self._trajectory_buffer.truncated = common.append_dict_array(
                self._trajectory_buffer.truncated,
                common.to_numpy(common.batch(truncated)),
            )
            done = terminated | truncated
            self._trajectory_buffer.done = common.append_dict_array(
                self._trajectory_buffer.done,
                common.to_numpy(common.batch(done)),
            )
            if "success" in info:
                self._trajectory_buffer.success = common.append_dict_array(
                    self._trajectory_buffer.success,
                    common.to_numpy(common.batch(info["success"])),
                )
            else:
                self._trajectory_buffer.success = None
            if "fail" in info:
                self._trajectory_buffer.fail = common.append_dict_array(
                    self._trajectory_buffer.fail,
                    common.to_numpy(common.batch(info["fail"])),
                )
            else:
                self._trajectory_buffer.fail = None

        if self.save_video:
            self._video_steps += 1
            if self.info_on_video:
                scalar_info = gym_utils.extract_scalars_from_info(
                    common.to_numpy(info), batch_size=self.num_envs
                )
                scalar_info["reward"] = common.to_numpy(rew)
                if np.size(scalar_info["reward"]) > 1:
                    scalar_info["reward"] = [
                        float(rew) for rew in scalar_info["reward"]
                    ]
                else:
                    scalar_info["reward"] = float(scalar_info["reward"])
                image = self.capture_image(scalar_info)
            else:
                image = self.capture_image()

            self.render_images.append(image)
            if (
                self.max_steps_per_video is not None
                and self._video_steps >= self.max_steps_per_video
            ):
                self.flush_video()
        self._elapsed_record_steps += 1
        if self.record_id_poses:
            self._record_id_poses_snapshot()
        if self.visualize_pointflow and self._pointflow_frames is not None:
            self._pointflow_frame_idx = min(
                self._pointflow_frame_idx + 1, self._pointflow_frames.shape[0] - 1
            )
        return obs, rew, terminated, truncated, info

    def flush_trajectory(
        self,
        verbose=False,
        ignore_empty_transition=True,
        env_idxs_to_flush=None,
        save: bool = True,
    ):
        """
        Flushes a trajectory and by default saves it to disk

        Arguments:
            verbose (bool): whether to print out information about the flushed trajectory
            ignore_empty_transition (bool): whether to ignore trajectories that did not have any actions
            env_idxs_to_flush: which environments by id to flush. If None, all environments are flushed.
            save (bool): whether to save the trajectory to disk
        """
        flush_count = 0

        def _allocate_unique_camera_dir(traj_id: str) -> Path:
            root = self.output_dir / "camera_data"
            root.mkdir(parents=True, exist_ok=True)
            base_name = traj_id
            for retry in range(1024):
                if retry == 0:
                    name = base_name
                else:
                    name = f"{base_name}_p{os.getpid()}_{retry}_{uuid.uuid4().hex[:6]}"
                p = root / name
                try:
                    p.mkdir(parents=False, exist_ok=False)
                    return p
                except FileExistsError:
                    continue
            raise RuntimeError("Failed to allocate unique camera_data trajectory directory")

        if env_idxs_to_flush is None:
            env_idxs_to_flush = np.arange(0, self.num_envs)
        for env_idx in env_idxs_to_flush:
            start_ptr = self._trajectory_buffer.env_episode_ptr[env_idx]
            end_ptr = len(self._trajectory_buffer.done)
            if ignore_empty_transition and end_ptr - start_ptr <= 1:
                continue
            flush_count += 1
            if save:
                self._episode_id += 1
                traj_id = "traj_{}".format(self._episode_id)
                group = self._h5_file.create_group(traj_id, track_order=True)
                camera_dir = _allocate_unique_camera_dir(traj_id)

                def recursive_add_to_h5py(
                    group: h5py.Group,
                    data: Union[dict, Array],
                    key,
                    path_prefix: str = "",
                ):
                    """simple recursive data insertion for nested data structures into h5py, optimizing for visual data as well"""
                    cur_path = f"{path_prefix}/{key}" if path_prefix else str(key)
                    if key == "sensor_param" and isinstance(data, dict):
                        if len(data) == 0:
                            return
                        cam_name = (
                            "base_camera"
                            if "base_camera" in data
                            else sorted(list(data.keys()))[0]
                        )
                        cam_params = data.get(cam_name, None)
                        if not isinstance(cam_params, dict):
                            return

                        cam2world = cam_params.get("cam2world_gl", None)
                        intrinsic = cam_params.get("intrinsic_cv", None)
                        if cam2world is None or intrinsic is None:
                            return

                        cam2world_seq = np.asarray(cam2world)[start_ptr:end_ptr, env_idx]
                        intrinsic_seq = np.asarray(intrinsic)[start_ptr:end_ptr, env_idx]
                        if cam2world_seq.ndim != 3 or cam2world_seq.shape[-2:] != (4, 4):
                            return
                        if intrinsic_seq.ndim != 3 or intrinsic_seq.shape[-2:] != (3, 3):
                            return
                        if cam2world_seq.shape[0] == 0:
                            return

                        # Keep camera-to-world directly (T_wc) and all time steps
                        cam_poses = cam2world_seq.astype(np.float32)
                        cam_intrinsics = intrinsic_seq.astype(np.float32)

                        np.save(camera_dir / "cam_poses.npy", cam_poses)
                        np.save(camera_dir / "cam_intrinsics.npy", cam_intrinsics)
                        return

                    if isinstance(data, dict):
                        subgrp = group.create_group(key, track_order=True)
                        for k in data.keys():
                            recursive_add_to_h5py(subgrp, data[k], k, cur_path)
                    else:
                        # 仅导出 base_camera 的视觉数据，跳过腕部等其他相机
                        if key in ["rgb", "depth", "segmentation"] and "base_camera" not in cur_path:
                            return

                        if key == "rgb":
                            rgb_data = np.asarray(data)[start_ptr:end_ptr, env_idx]
                            # save as rgb.mp4 under camera_dir
                            images_to_video(
                                list(rgb_data),
                                str(camera_dir),
                                video_name="rgb",
                                fps=16,
                                quality=10,
                                verbose=False,
                            )
                        elif key == "depth":
                            depth_data = np.asarray(data)[start_ptr:end_ptr, env_idx]
                            # remove any singleton channel dim (works for (T,H,W,1) and (T,H,W))
                            depth_data = np.squeeze(depth_data)
                            # replace sentinel/zero (invalid) with camera far plane (convert m->mm)
                            cam = list(self.base_env._sensors.values())[0]
                            far_mm = int(round(getattr(cam.config, "far", 100) * 1000))
                            depth_data = np.where((depth_data == 0) | (depth_data == -32768), far_mm, depth_data)
                            depth_data = depth_data.astype(np.float32) / 1000.0
                            
                            # save as depth_video.npy under camera_dir (meters, float16)
                            np.save(camera_dir / "depth_video.npy", depth_data.astype(np.float16))
                        elif key == "segmentation":
                            seg_data = np.asarray(data)[start_ptr:end_ptr, env_idx]
                            # squeeze trailing channel dim if present (e.g. (T,H,W,1) -> (T,H,W))
                            if seg_data.ndim >= 3 and seg_data.shape[-1] == 1:
                                seg_data = np.squeeze(seg_data, axis=-1)
                            # save segmentation as seg.npy under camera_dir
                            np.save(camera_dir / "seg.npy", seg_data)
                        else:
                            group.create_dataset(
                                key,
                                data=data[start_ptr:end_ptr, env_idx],
                                dtype=data.dtype,
                            )

                # Observations need special processing
                if isinstance(self._trajectory_buffer.observation, dict):
                    recursive_add_to_h5py(
                        group, self._trajectory_buffer.observation, "obs"
                    )
                elif isinstance(self._trajectory_buffer.observation, np.ndarray):
                    if self.cpu_wrapped_env:
                        group.create_dataset(
                            "obs",
                            data=self._trajectory_buffer.observation[start_ptr:end_ptr],
                            dtype=self._trajectory_buffer.observation.dtype,
                        )
                    else:
                        group.create_dataset(
                            "obs",
                            data=self._trajectory_buffer.observation[
                                start_ptr:end_ptr, env_idx
                            ],
                            dtype=self._trajectory_buffer.observation.dtype,
                        )
                else:
                    raise NotImplementedError(
                        f"RecordEpisode wrapper does not know how to handle observation data of type {type(self._trajectory_buffer.observation)}"
                    )
                episode_info = dict(
                    episode_id=self._episode_id,
                    episode_seed=self.base_env._episode_seed[env_idx],
                    control_mode=self.base_env.control_mode,
                    elapsed_steps=end_ptr - start_ptr - 1,
                )
                if self.num_envs == 1:
                    episode_info.update(reset_kwargs=self.last_reset_kwargs)
                else:
                    # NOTE (stao): With multiple envs in GPU simulation, reset_kwargs do not make much sense
                    episode_info.update(reset_kwargs=dict())

                # slice some data to remove the first dummy frame.
                actions = common.index_dict_array(
                    self._trajectory_buffer.action,
                    (slice(start_ptr + 1, end_ptr), env_idx),
                )
                terminated = self._trajectory_buffer.terminated[
                    start_ptr + 1 : end_ptr, env_idx
                ]
                truncated = self._trajectory_buffer.truncated[
                    start_ptr + 1 : end_ptr, env_idx
                ]
                if isinstance(self._trajectory_buffer.action, dict):
                    recursive_add_to_h5py(group, actions, "actions")
                else:
                    group.create_dataset("actions", data=actions, dtype=np.float32)
                group.create_dataset("terminated", data=terminated, dtype=bool)
                group.create_dataset("truncated", data=truncated, dtype=bool)

                if self._trajectory_buffer.success is not None:
                    group.create_dataset(
                        "success",
                        data=self._trajectory_buffer.success[
                            start_ptr + 1 : end_ptr, env_idx
                        ],
                        dtype=bool,
                    )
                    episode_info.update(
                        success=self._trajectory_buffer.success[end_ptr - 1, env_idx]
                    )
                if self._trajectory_buffer.fail is not None:
                    group.create_dataset(
                        "fail",
                        data=self._trajectory_buffer.fail[
                            start_ptr + 1 : end_ptr, env_idx
                        ],
                        dtype=bool,
                    )
                    episode_info.update(
                        fail=self._trajectory_buffer.fail[end_ptr - 1, env_idx]
                    )
                if self.record_env_state:
                    recursive_add_to_h5py(
                        group, self._trajectory_buffer.state, "env_states"
                    )
                if self.record_reward:
                    group.create_dataset(
                        "rewards",
                        data=self._trajectory_buffer.reward[
                            start_ptr + 1 : end_ptr, env_idx
                        ],
                        dtype=np.float32,
                    )

                if self.record_id_poses and len(self._id_pose_buffer) > 0:
                    id_poses_group = group.create_group("id_poses", track_order=True)
                    for seg_id, name in self.id_to_name.items():
                        id_poses_group.attrs[str(seg_id)] = name
                    for seg_id, pose_data in self._id_pose_buffer.items():
                        if len(pose_data["position"]) == 0:
                            continue
                        pos_arr = np.asarray(pose_data["position"], dtype=np.float32)
                        quat_arr = np.asarray(pose_data["quaternion"], dtype=np.float32)
                        cpos_arr = np.asarray(
                            pose_data["camera_position"], dtype=np.float32
                        )
                        cquat_arr = np.asarray(
                            pose_data["camera_quaternion"], dtype=np.float32
                        )
                        if end_ptr > pos_arr.shape[0]:
                            continue
                        seg_group = id_poses_group.create_group(
                            f"{seg_id}", track_order=True
                        )
                        seg_group.attrs["name"] = self.id_to_name.get(
                            seg_id, f"unknown_{seg_id}"
                        )
                        seg_group.attrs["seg_id"] = seg_id
                        if self.record_id_mesh_info and seg_id in self.id_geometry_meta:
                            meta = self.id_geometry_meta[seg_id]
                            mesh_file_path = meta.get("mesh_file_path", None)
                            if mesh_file_path is not None:
                                seg_group.attrs["mesh_file_path"] = str(mesh_file_path)
                            mesh_file_paths = meta.get("mesh_file_paths", [])
                            seg_group.attrs["mesh_file_paths_json"] = json.dumps(
                                self._convert_to_serializable(mesh_file_paths),
                                ensure_ascii=False,
                            )
                            seg_group.attrs["geometry_params_json"] = json.dumps(
                                self._convert_to_serializable(
                                    meta.get("reconstructable_params", dict())
                                ),
                                ensure_ascii=False,
                            )
                        seg_group.create_dataset(
                            "position",
                            data=pos_arr[start_ptr:end_ptr, env_idx],
                            dtype=np.float32,
                        )
                        seg_group.create_dataset(
                            "quaternion",
                            data=quat_arr[start_ptr:end_ptr, env_idx],
                            dtype=np.float32,
                        )
                        seg_group.create_dataset(
                            "camera_position",
                            data=cpos_arr[start_ptr:end_ptr, env_idx],
                            dtype=np.float32,
                        )
                        seg_group.create_dataset(
                            "camera_quaternion",
                            data=cquat_arr[start_ptr:end_ptr, env_idx],
                            dtype=np.float32,
                        )

                self._json_data["episodes"].append(common.to_numpy(episode_info))
                dump_json(self._json_path, self._json_data, indent=2)
                # Also write a per-episode h5 file under camera_data for convenience.
                try:
                    per_h5_path = camera_dir / f"{traj_id}.h5"
                    # Copy the episode group from the main h5 file to a per-episode h5 file.
                    with h5py.File(str(per_h5_path), "w") as per_f:
                        try:
                            # Copy the group under its original name into the new file.
                            self._h5_file.copy(traj_id, per_f)
                        except Exception:
                            # Fall back to copying children if direct copy fails.
                            grp = self._h5_file[traj_id]
                            for k in grp.keys():
                                try:
                                    self._h5_file.copy(f"{traj_id}/{k}", per_f, name=k)
                                except Exception:
                                    continue
                except Exception:
                    logger.warn(f"Failed to write per-episode h5 for {traj_id}")

                self._submit_postprocess_camera_dir(camera_dir)

                if verbose:
                    if flush_count == 1:
                        print(f"Recorded episode {self._episode_id}")
                    else:
                        print(
                            f"Recorded episodes {self._episode_id - flush_count} to {self._episode_id}"
                        )

        # truncate self._trajectory_buffer down to save memory
        if flush_count > 0:
            self._trajectory_buffer.env_episode_ptr[env_idxs_to_flush] = (
                len(self._trajectory_buffer.done) - 1
            )
            min_env_ptr = self._trajectory_buffer.env_episode_ptr.min()
            N = len(self._trajectory_buffer.done)

            if self.record_env_state:
                self._trajectory_buffer.state = common.index_dict_array(
                    self._trajectory_buffer.state, slice(min_env_ptr, N)
                )
            self._trajectory_buffer.observation = common.index_dict_array(
                self._trajectory_buffer.observation, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.action = common.index_dict_array(
                self._trajectory_buffer.action, slice(min_env_ptr, N)
            )
            if self.record_reward:
                self._trajectory_buffer.reward = common.index_dict_array(
                    self._trajectory_buffer.reward, slice(min_env_ptr, N)
                )
            self._trajectory_buffer.terminated = common.index_dict_array(
                self._trajectory_buffer.terminated, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.truncated = common.index_dict_array(
                self._trajectory_buffer.truncated, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.done = common.index_dict_array(
                self._trajectory_buffer.done, slice(min_env_ptr, N)
            )
            if self._trajectory_buffer.success is not None:
                self._trajectory_buffer.success = common.index_dict_array(
                    self._trajectory_buffer.success, slice(min_env_ptr, N)
                )
            if self._trajectory_buffer.fail is not None:
                self._trajectory_buffer.fail = common.index_dict_array(
                    self._trajectory_buffer.fail, slice(min_env_ptr, N)
                )
            if self.record_id_poses and len(self._id_pose_buffer) > 0:
                for seg_id in self._id_pose_buffer.keys():
                    for key in self._id_pose_buffer[seg_id].keys():
                        arr = np.asarray(self._id_pose_buffer[seg_id][key])
                        if len(arr) == 0:
                            continue
                        self._id_pose_buffer[seg_id][key] = list(arr[min_env_ptr:N])
            self._trajectory_buffer.env_episode_ptr -= min_env_ptr

    def flush_video(
        self,
        name=None,
        suffix="",
        verbose=False,
        ignore_empty_transition=True,
        save: bool = True,
    ):
        """
        Flush a video of the recorded episode(s) anb by default saves it to disk

        Arguments:
            name (str): name of the video file. If None, it will be named with the episode id.
            suffix (str): suffix to add to the video file name
            verbose (bool): whether to print out information about the flushed video
            ignore_empty_transition (bool): whether to ignore trajectories that did not have any actions
            save (bool): whether to save the video to disk
        """
        if len(self.render_images) == 0:
            return
        if ignore_empty_transition and len(self.render_images) == 1:
            return
        if save:
            self._video_id += 1
            if name is None:
                video_name = "{}".format(self._video_id)
                if suffix:
                    video_name += "_" + suffix
                if self._avoid_overwriting_video:
                    while (
                        Path(self.output_dir)
                        / (video_name.replace(" ", "_").replace("\n", "_") + ".mp4")
                    ).exists():
                        self._video_id += 1
                        video_name = "{}".format(self._video_id)
                        if suffix:
                            video_name += "_" + suffix
            else:
                video_name = name
            images_to_video(
                self.render_images,
                str(self.output_dir),
                video_name=video_name,
                fps=self.video_fps,
                quality=10,
                verbose=verbose,
            )
        self._video_steps = 0
        self.render_images = []

    def close(self) -> None:
        if self._closed:
            # There is some strange bug when vector envs using record wrapper are closed/deleted, this code runs twice
            return
        self._closed = True
        if self.save_trajectory:
            # Handle the last episode only when `save_on_reset=True`
            if self.save_on_reset and self._trajectory_buffer is not None:
                self.flush_trajectory(
                    ignore_empty_transition=True,
                    env_idxs_to_flush=np.arange(self.num_envs),
                )
            if self.clean_on_close:
                clean_trajectories(self._h5_file, self._json_data)
                dump_json(self._json_path, self._json_data, indent=2)
            self._h5_file.close()
        if self.save_video:
            if self.save_on_reset:
                self.flush_video()

        if self._postprocess_executor is not None:
            self._postprocess_executor.shutdown(wait=True)

        return super().close()
