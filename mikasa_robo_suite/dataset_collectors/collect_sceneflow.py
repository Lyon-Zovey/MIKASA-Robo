"""
MIKASA-Robo Scene Flow Collector
=================================
对已有的 unbatched 轨迹 .npz 文件进行物理回放，每帧采集
  depth / segmentation / camera-params / object-poses
并计算 **5条** 双向点云 tracking（以轨迹 [0, 1/4, 2/4, 3/4, 4/4] 处的帧为参考），
最终以 (5, T, H, W, 3) float32 保存为 sceneflow_<N>.npy。

坐标系约定
  - 深度反投影 / 变换均在 OpenCV 相机坐标系（x-右 y-下 z-前）
  - delta 全部在 **参考帧的相机坐标系** 下表示:
      delta_t = p_cam_ref(pts_world_t) − p_cam_ref(pts_world_ref)

可视化（--visualize）
  为每条轨迹生成一张 PNG，包含：
    行 1 : RGB | 深度图 | 分割图（按 actor 染色）
    行 2-6 : 5 个参考帧的 scene-flow 热力图（flow 幅度）叠加到 RGB 上

用法示例
  python mikasa_robo_suite/dataset_collectors/collect_sceneflow.py \\
      --env_id ShellGameTouch-v0 \\
      --data_dir data_with_seed/MIKASA-Robo/unbatched/ShellGameTouch-v0 \\
      --save_dir sceneflow/MIKASA-Robo/ShellGameTouch-v0 \\
      --cam_name base_camera \\
      --max_trajectories 5 \\
      --visualize
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import tyro

from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mikasa_robo_suite.memory_envs import *  # registers all envs


# ──────────────────────────────────────────────────────────────────────────────
# 1. 坐标变换工具
# ──────────────────────────────────────────────────────────────────────────────

def depth_to_pts_cam(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    depth : (H, W) float32  z-depth [metres], 0 表示无效
    K     : (3, 3) float64  相机内参矩阵 (OpenCV convention)

    Returns
    -------
    (H, W, 3) float32  在 OpenCV 相机坐标系下的 3-D 坐标
    """
    H, W = depth.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = depth.astype(np.float32)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return np.stack([x, y, z], axis=-1)          # (H, W, 3)


def pts_cam_to_world(pts: np.ndarray, cam2world_gl: np.ndarray) -> np.ndarray:
    """
    OpenCV 相机坐标系 → 世界坐标系。

    Parameters
    ----------
    pts          : (..., 3) OpenCV 相机坐标系下的点（x-右 y-下 z-前）
    cam2world_gl : (4, 4) ManiSkill 提供的 cam-to-world 矩阵（OpenGL 约定）

    Notes
    -----
    OpenGL 相机约定：x-右 y-上 z-后。
    因此 OpenCV→OpenGL 需要 flip y 和 z：
      x_gl =  x_cv
      y_gl = -y_cv
      z_gl = -z_cv

    Returns
    -------
    (..., 3) 世界坐标 float32
    """
    shape = pts.shape
    flat = pts.reshape(-1, 3).astype(np.float64)
    # OpenCV → homogeneous OpenGL cam
    homo_gl = np.stack(
        [flat[:, 0], -flat[:, 1], -flat[:, 2], np.ones(len(flat), dtype=np.float64)],
        axis=1,
    )   # (N, 4)
    world = (cam2world_gl.astype(np.float64) @ homo_gl.T).T[:, :3]
    return world.astype(np.float32).reshape(shape)


def pts_world_to_cam(pts: np.ndarray, cam2world_gl: np.ndarray) -> np.ndarray:
    """
    世界坐标系 → OpenCV 相机坐标系（用于计算 delta）。

    Parameters
    ----------
    pts          : (..., 3) 世界坐标
    cam2world_gl : (4, 4) cam-to-world (OpenGL)

    Returns
    -------
    (..., 3) OpenCV 相机坐标 float32
    """
    shape = pts.shape
    flat = pts.reshape(-1, 3).astype(np.float64)
    homo = np.concatenate([flat, np.ones((len(flat), 1), dtype=np.float64)], axis=1)
    world2cam_gl = np.linalg.inv(cam2world_gl.astype(np.float64))
    pts_gl = (world2cam_gl @ homo.T).T[:, :3]           # OpenGL cam coords
    # OpenGL → OpenCV: flip y, z
    pts_cv = np.stack(
        [pts_gl[:, 0], -pts_gl[:, 1], -pts_gl[:, 2]], axis=1
    )
    return pts_cv.astype(np.float32).reshape(shape)


def pose7_to_Rt(pose7: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    SAPIEN raw_pose 格式: [tx, ty, tz, qw, qx, qy, qz]

    Returns
    -------
    R : (3, 3) float64 旋转矩阵
    t : (3,)   float64 平移向量
    """
    t = pose7[:3].astype(np.float64)
    qw, qx, qy, qz = float(pose7[3]), float(pose7[4]), float(pose7[5]), float(pose7[6])
    # scipy Rotation 使用 scalar-last [x, y, z, w]
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return R, t


def rigid_transform_pts(
    pts: np.ndarray,        # (N, 3) 参考帧世界坐标
    pose_ref: np.ndarray,   # (7,)   物体在参考帧的 raw_pose
    pose_t: np.ndarray,     # (7,)   物体在目标帧 t 的 raw_pose
) -> np.ndarray:
    """
    将 pts（固连在物体局部坐标系中）从参考帧世界坐标刚体变换到帧 t 的世界坐标。

    Returns
    -------
    (N, 3) float32  帧 t 下的世界坐标
    """
    R_ref, t_ref = pose7_to_Rt(pose_ref)
    R_t,   t_t   = pose7_to_Rt(pose_t)
    p = pts.astype(np.float64)
    p_local   = (p - t_ref) @ R_ref      # (N,3) 物体局部坐标系
    p_world_t = p_local @ R_t.T + t_t    # (N,3) 帧 t 世界坐标
    return p_world_t.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 2. 各 env 可追踪物体定义
# ──────────────────────────────────────────────────────────────────────────────

def get_trackable_actors(
    base_env,
    env_id: str,
) -> List[Tuple[str, object]]:
    """
    返回 [(actor_name, actor_obj), …]，只含场景中会运动的 task-level actor。
    新 env 类型在此处添加分支即可。

    注意：各 env 的 `evaluate()` 可能在 CPU-buffer 将物体移到 z=1000
    （"隐藏"效果），但 GPU 物理引擎中位置不变。调用本函数时需先
    执行 _gpu_fetch_all() 以获取真实物理位置。
    """
    eid = env_id.lower()

    if "shellgame" in eid:
        return [
            ("mug_left",   base_env.mug_left),
            ("mug_center", base_env.mug_center),
            ("mug_right",  base_env.mug_right),
            ("red_ball",   base_env.red_ball),
        ]
    if "takeitback" in eid:
        return [("cube", base_env.cube)]
    if "rotate" in eid:
        return [("peg", base_env.peg)]
    if "interceptgrab" in eid:
        return [("ball", base_env.ball)]
    if "intercept" in eid:
        return [("ball", base_env.ball), ("goal_region", base_env.goal_region)]
    if any(k in eid for k in ("remembercolor", "bunchofcolors", "seqofcolors", "chainofcolors")):
        return [(f"cube_{k}", cube) for k, cube in base_env.cubes.items()]
    if "remembershapeandcolor" in eid or "remembershape" in eid:
        return [(f"shape_{k}", sh) for k, sh in base_env.shapes.items()]

    raise ValueError(
        f"[SceneFlow] env_id {env_id!r} 尚无 actor 映射。"
        "请在 get_trackable_actors() 中添加对应分支。"
    )


def build_seg_id_map(
    trackable: List[Tuple[str, object]],
) -> Dict[int, str]:
    """
    构建 {segmentation_id (int): actor_name} 的查找表（num_envs=1）。
    """
    id_to_name: Dict[int, str] = {}
    for name, actor in trackable:
        ids_tensor = actor.per_scene_id
        if isinstance(ids_tensor, torch.Tensor):
            ids = ids_tensor.cpu().numpy().flatten().tolist()
        else:
            ids = [int(ids_tensor)]
        for sid in ids:
            id_to_name[int(sid)] = name
    return id_to_name


# ──────────────────────────────────────────────────────────────────────────────
# 3. 单条轨迹的场景流计算
# ──────────────────────────────────────────────────────────────────────────────

def _squeeze_sensor(t: torch.Tensor) -> np.ndarray:
    """将 (1, H, W, C) 或 (H, W, C) tensor 压成 (H, W, C) numpy。"""
    arr = t.cpu().numpy()
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _squeeze_param(t: torch.Tensor) -> np.ndarray:
    """将 (1, M, N) 或 (M, N) tensor 压成 (M, N) numpy。"""
    arr = t.cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def collect_sceneflow_one_traj(
    npz_path: str,
    env_vec,          # ManiSkillVectorEnv (num_envs=1, ignore_terminations=True)
    base_env,         # env_vec.base_env  (ManiSkill BaseEnv)
    env_id: str,
    cam_name: str,
    device: torch.device,
    return_frames: bool = False,
) -> Tuple:
    """
    回放一条轨迹，计算 5 条参考帧的双向点云 tracking。

    Returns
    -------
    scene_flow : (5, T, H, W, 3) float32
    extras     : dict  只在 return_frames=True 时包含:
        "frame_rgb"   : list[T] of (H,W,3) uint8
        "frame_depth" : list[T] of (H,W) float32
        "frame_seg"   : list[T] of (H,W) int32
        "valid_all"   : list[T] of (H,W) bool
        "id_to_name"  : Dict[int,str]
        "actor_names" : List[str]
    """
    data       = np.load(npz_path, allow_pickle=True)
    seed       = int(data["seed"][0])
    init_state = data["init_state"]          # (1, state_dim)
    actions    = data["action"]              # (T, action_dim)
    T = len(actions)

    # ── 3.1 重置环境并注入精确初始状态 ────────────────────────────────────────
    env_vec.reset(seed=[seed])
    if base_env.gpu_sim_enabled:
        base_env.scene._gpu_fetch_all()
    base_env.set_state(
        torch.tensor(init_state, dtype=torch.float32).to(device)
    )

    # ── 3.2 构建 actor 追踪表（reset 后 per_scene_id 才稳定） ──────────────────
    trackable  = get_trackable_actors(base_env, env_id)
    id_to_name = build_seg_id_map(trackable)
    actor_map: Dict[str, object] = {n: a for n, a in trackable}

    # ── 3.3 逐帧采集深度 / 分割 / 相机参数 / 物体位姿 ─────────────────────────
    frame_rgb:        List[np.ndarray] = []   # (H, W, 3) uint8  ← 用于可视化
    frame_depth:      List[np.ndarray] = []   # (H, W) float32  单位：米
    frame_seg:        List[np.ndarray] = []   # (H, W) int32
    frame_c2w_gl:     List[np.ndarray] = []   # (4, 4) cam2world OpenGL
    frame_obj_poses:  List[Dict[str, np.ndarray]] = []  # {name: (7,) float32}
    K_cv: Optional[np.ndarray] = None         # (3, 3) – fixed cameras 不变

    for t in range(T):
        # 安装版的 get_obs() 对视觉 obs_mode 直接返回未 flatten 的 dict；
        # 内部调用 evaluate()，更新 CPU-buffer 位姿（可能有 z=1000 隐藏效果），
        # 渲染（depth / seg）使用该 buffer。
        raw_obs = base_env.get_obs()

        sd = raw_obs["sensor_data"][cam_name]
        sp = raw_obs["sensor_param"][cam_name]

        # RGB（供可视化）
        rgb_arr = _squeeze_sensor(sd["rgb"])          # (H, W, 3) or (H, W, 4)
        rgb_arr = rgb_arr[..., :3].astype(np.uint8)  # (H, W, 3)
        frame_rgb.append(rgb_arr)

        # depth: int16, 单位毫米 → 转 float32 米
        depth_raw = _squeeze_sensor(sd["depth"])       # (H, W, 1) int16
        depth = depth_raw[..., 0].astype(np.float32) / 1000.0   # (H, W) 米

        seg = _squeeze_sensor(sd["segmentation"])     # (H, W, 1) int16
        if seg.ndim == 3:
            seg = seg[..., 0]
        seg = seg.astype(np.int32)                    # (H, W)

        # cam2world_gl (4,4) — GL convention，用于 cam↔world 变换
        c2w = _squeeze_param(sp["cam2world_gl"])       # (4, 4)

        if K_cv is None:
            K_cv = _squeeze_param(sp["intrinsic_cv"]).astype(np.float64)  # (3, 3)

        # GPU → CPU 同步，获取真实物理位置（绕过 evaluate() 的 z=1000 CPU trick）
        if base_env.gpu_sim_enabled:
            base_env.scene._gpu_fetch_all()

        poses_t: Dict[str, np.ndarray] = {}
        for name, actor in actor_map.items():
            raw = actor.pose.raw_pose
            if isinstance(raw, torch.Tensor):
                raw = raw.cpu().numpy()
            raw = raw.reshape(-1, 7)[0].astype(np.float32)  # (7,)
            poses_t[name] = raw

        frame_depth.append(depth)
        frame_seg.append(seg)
        frame_c2w_gl.append(c2w.astype(np.float64))
        frame_obj_poses.append(poses_t)

        # 用存储动作推进一步
        action_t = (
            torch.tensor(actions[t], dtype=torch.float32)
            .to(device)
            .unsqueeze(0)          # (1, action_dim)
        )
        env_vec.step(action_t)

    H, W = frame_depth[0].shape

    # ── 3.4 逐帧反投影得到世界坐标点云 ───────────────────────────────────────
    pts_world_all: List[np.ndarray] = []
    for t in range(T):
        pts_cam   = depth_to_pts_cam(frame_depth[t], K_cv)        # (H, W, 3)
        pts_world = pts_cam_to_world(pts_cam, frame_c2w_gl[t])    # (H, W, 3)
        pts_world_all.append(pts_world)

    # 有效深度 mask：排除无表面像素（深度=0 或超过远截面）
    DEPTH_MIN, DEPTH_MAX = 1e-3, 99.0
    valid_all = [
        (frame_depth[t] > DEPTH_MIN) & (frame_depth[t] < DEPTH_MAX)
        for t in range(T)
    ]   # list of (H, W) bool

    # ── 3.5 5 个参考帧的双向点云 tracking ────────────────────────────────────
    kf_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    scene_flow = np.zeros((5, T, H, W, 3), dtype=np.float32)

    for ki, ref_idx in enumerate(kf_indices):
        pts_world_ref = pts_world_all[ref_idx]    # (H, W, 3)
        seg_ref       = frame_seg[ref_idx]        # (H, W)
        c2w_ref       = frame_c2w_gl[ref_idx]     # (4, 4)
        poses_ref     = frame_obj_poses[ref_idx]  # {name: (7,)}

        # 参考帧点云投影到参考帧 OpenCV 相机坐标系
        pts_cam_ref = pts_world_to_cam(pts_world_ref, c2w_ref)      # (H, W, 3)

        # 预计算每个 actor 在参考帧里的像素 mask
        actor_masks: Dict[str, np.ndarray] = {}
        for name in actor_map:
            mask = np.zeros((H, W), dtype=bool)
            for seg_id, aname in id_to_name.items():
                if aname == name:
                    mask |= (seg_ref == seg_id)
            actor_masks[name] = mask

        for t in range(T):
            poses_t = frame_obj_poses[t]

            # 默认：所有点静止（背景 / 未标记区域）
            pts_world_t = pts_world_ref.copy()   # (H, W, 3)

            # 对每个 actor 做刚体变换
            for name in actor_map:
                if name not in poses_ref or name not in poses_t:
                    continue
                mask = actor_masks[name]
                if not mask.any():
                    continue
                pts_flat = pts_world_ref[mask]    # (N, 3)
                pts_new  = rigid_transform_pts(
                    pts_flat, poses_ref[name], poses_t[name]
                )
                pts_world_t[mask] = pts_new       # (N, 3)

            # 将追踪后的世界坐标投影回参考帧 OpenCV 相机坐标系
            pts_cam_t = pts_world_to_cam(pts_world_t, c2w_ref)      # (H, W, 3)

            # delta = 目标位置 − 参考位置（均在参考相机坐标系下）
            delta = pts_cam_t - pts_cam_ref                         # (H, W, 3)

            # 参考帧 或 目标帧 无有效深度的像素 → delta 置 0
            valid = valid_all[ref_idx] & valid_all[t]
            delta[~valid] = 0.0

            scene_flow[ki, t] = delta

    if return_frames:
        extras = {
            "frame_rgb":    frame_rgb,
            "frame_depth":  frame_depth,
            "frame_seg":    frame_seg,
            "frame_c2w_gl": frame_c2w_gl,
            "K_cv":         K_cv,
            "valid_all":    valid_all,
            "id_to_name":   id_to_name,
            "actor_names":  [n for n, _ in trackable],
        }
        return scene_flow, extras   # (5, T, H, W, 3), dict
    return scene_flow, {}           # (5, T, H, W, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 可视化
# ──────────────────────────────────────────────────────────────────────────────

# 每个 actor 的固定颜色 (RGB 0-1)
_ACTOR_PALETTE = [
    (0.9, 0.2, 0.2),   # red
    (0.2, 0.7, 0.2),   # green
    (0.2, 0.4, 0.9),   # blue
    (0.9, 0.7, 0.1),   # yellow
    (0.8, 0.2, 0.8),   # magenta
    (0.1, 0.8, 0.8),   # cyan
    (0.9, 0.5, 0.1),   # orange
    (0.5, 0.2, 0.9),   # violet
]


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    """depth (H,W) float → (H,W,3) uint8  (plasma colormap, invalid→black)."""
    valid = (depth > 1e-3) & (depth < 99.0)
    d_norm = np.zeros_like(depth)
    if valid.any():
        dmin, dmax = depth[valid].min(), depth[valid].max()
        d_norm[valid] = (depth[valid] - dmin) / max(dmax - dmin, 1e-6)
    rgba = cm.plasma(d_norm)                 # (H,W,4) float 0-1
    rgb  = (rgba[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def _colorize_seg(
    seg: np.ndarray,
    id_to_name: Dict[int, str],
    actor_names: List[str],
) -> np.ndarray:
    """
    seg (H,W) int → (H,W,3) uint8  (每个 actor 一种颜色，背景灰色).
    """
    H, W = seg.shape
    out = np.full((H, W, 3), 80, dtype=np.uint8)          # gray background
    name_to_color = {
        n: tuple(int(c * 255) for c in _ACTOR_PALETTE[i % len(_ACTOR_PALETTE)])
        for i, n in enumerate(actor_names)
    }
    for seg_id, name in id_to_name.items():
        color = name_to_color.get(name, (200, 200, 200))
        mask  = (seg == seg_id)
        out[mask] = color
    return out


def _flow_magnitude_overlay(
    rgb: np.ndarray,                   # (H,W,3) uint8
    flow_xy: np.ndarray,               # (H,W,2) float  delta_x, delta_y in cam
    valid_mask: np.ndarray,            # (H,W) bool
    alpha: float = 0.6,
    stride: int = 6,
) -> np.ndarray:
    """
    将 flow 幅度渲染成热力图叠加在 RGB 上，并用稀疏箭头显示方向。

    Parameters
    ----------
    flow_xy   : (H,W,2) X/Y 分量（相机坐标系，与图像轴一致）
    valid_mask: 有效深度 mask
    stride    : 箭头间隔（像素）
    """
    H, W = rgb.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=80)
    ax.set_axis_off()

    # 热力图（flow 幅度）
    mag = np.linalg.norm(flow_xy, axis=-1)              # (H,W)
    mag_show = np.where(valid_mask, mag, np.nan)
    vmax = np.nanpercentile(mag_show, 95) if valid_mask.any() else 1.0
    vmax = max(vmax, 1e-4)

    ax.imshow(rgb)
    im = ax.imshow(
        mag_show, cmap="hot", alpha=alpha,
        vmin=0, vmax=vmax, interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="‖Δ‖ cam [m]")

    # 稀疏箭头
    ys = np.arange(stride // 2, H, stride)
    xs = np.arange(stride // 2, W, stride)
    XX, YY = np.meshgrid(xs, ys)
    U = flow_xy[YY, XX, 0]        # delta_x (右)
    V = flow_xy[YY, XX, 1]        # delta_y (下) ← 图像 y 轴朝下
    vm = valid_mask[YY, XX]
    scale = vmax * max(H, W) * 0.08
    scale = max(scale, 1e-6)
    ax.quiver(
        XX[vm], YY[vm], U[vm], V[vm],
        color="white", scale=scale, scale_units="xy",
        width=0.004, headwidth=4, headlength=5, alpha=0.9,
    )
    ax.set_title(f"flow (vmax={vmax:.3f}m)", fontsize=8)

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    w_fig, h_fig = fig.canvas.get_width_height()
    # buffer_rgba returns RGBA; drop alpha channel
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(h_fig, w_fig, 4)[..., :3].copy()
    plt.close(fig)
    return img


def save_visualization(
    scene_flow:   np.ndarray,          # (5, T, H, W, 3)
    frame_rgb:    List[np.ndarray],    # list[T] of (H,W,3) uint8
    frame_depth:  List[np.ndarray],    # list[T] of (H,W) float
    frame_seg:    List[np.ndarray],    # list[T] of (H,W) int
    id_to_name:   Dict[int, str],
    actor_names:  List[str],
    valid_all:    List[np.ndarray],    # list[T] of (H,W) bool
    save_path:    str,
    traj_idx:     int,
) -> None:
    """
    生成并保存可视化 PNG：
      行 1: RGB | 深度图 | 分割图   (参考帧 0)
      行 2: 5 个参考帧各自在帧 T-1 的 flow 幅度叠加图
    """
    T = scene_flow.shape[1]
    kf_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]

    n_cols = 5
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 8))
    fig.suptitle(f"Traj {traj_idx}  |  T={T}", fontsize=12)

    # ── 行 1: 参考帧 0 的 RGB / depth / seg（后两格留给 flow legend 用） ──────
    rgb0   = frame_rgb[0]
    depth0 = frame_depth[0]
    seg0   = frame_seg[0]

    axes[0, 0].imshow(rgb0)
    axes[0, 0].set_title("RGB  (t=0)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(_colorize_depth(depth0))
    axes[0, 1].set_title("Depth  (t=0)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(_colorize_seg(seg0, id_to_name, actor_names))
    axes[0, 2].set_title("Segmentation  (t=0)")
    axes[0, 2].axis("off")

    # 图例 patch
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=_ACTOR_PALETTE[i % len(_ACTOR_PALETTE)], label=n)
        for i, n in enumerate(actor_names)
    ]
    axes[0, 3].legend(handles=legend_handles, loc="center", fontsize=9,
                      frameon=False)
    axes[0, 3].set_title("Actor legend")
    axes[0, 3].axis("off")
    axes[0, 4].axis("off")

    # ── 行 2: 5 个参考帧 → flow 到最后一帧 ────────────────────────────────────
    for ki, ref_idx in enumerate(kf_indices):
        # 取 flow 到尽可能靠近 T-1（对于 ref=T-1 则看 t=0）
        t_show = (T - 1) if ref_idx < T - 1 else 0
        flow_3d   = scene_flow[ki, t_show]         # (H,W,3) delta in ref-cam
        flow_xy   = flow_3d[..., :2]               # X (right), Y (down)
        valid     = valid_all[ref_idx] & valid_all[t_show]
        rgb_ref   = frame_rgb[ref_idx]

        vis = _flow_magnitude_overlay(rgb_ref, flow_xy, valid)
        axes[1, ki].imshow(vis)
        axes[1, ki].set_title(
            f"ref={ref_idx}→t={t_show}\n(keyframe {ki})", fontsize=8
        )
        axes[1, ki].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 5. CLI 入口
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    env_id: str = "ShellGameTouch-v0"
    """目标环境 ID，需与 data_dir 下的轨迹文件对应"""

    data_dir: str = "data_with_seed/MIKASA-Robo/unbatched/ShellGameTouch-v0"
    """unbatched .npz 轨迹文件所在目录"""

    save_dir: str = "sceneflow/MIKASA-Robo/ShellGameTouch-v0"
    """scene flow .npy 文件保存目录"""

    cam_name: str = "base_camera"
    """用于点云反投影的相机名，通常为 base_camera 或 hand_camera"""

    max_trajectories: Optional[int] = None
    """最多处理的轨迹数量（None = 全部）"""

    start_from: int = 0
    """从第几条轨迹开始处理（按文件序号排序）"""

    skip_existing: bool = True
    """跳过已存在的 sceneflow 文件"""

    visualize: bool = False
    """同时生成可视化 PNG（RGB / depth / seg / flow 热力图），保存在 save_dir/vis/ 目录"""


def main() -> None:
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 创建回放用环境 ─────────────────────────────────────────────────────────
    print(f"[SceneFlow] Creating env: {args.env_id}  (obs_mode=rgb+depth+segmentation)")
    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_delta_pos",
        render_mode="all",
        sim_backend="gpu" if torch.cuda.is_available() else "cpu",
        reward_mode="normalized_dense",
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    # ignore_terminations=True：即使 episode 提前成功也不会 auto-reset
    env_vec = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=False)
    base_env = env_vec.base_env

    # ── 定位轨迹文件 ──────────────────────────────────────────────────────────
    npz_files = sorted(
        Path(args.data_dir).glob("train_data_*.npz"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    npz_files = npz_files[args.start_from:]
    if args.max_trajectories is not None:
        npz_files = npz_files[: args.max_trajectories]

    print(f"[SceneFlow] Found {len(npz_files)} trajectories → saving to {args.save_dir}")

    vis_dir = os.path.join(args.save_dir, "vis")
    if args.visualize:
        os.makedirs(vis_dir, exist_ok=True)

    # ── 逐条处理 ──────────────────────────────────────────────────────────────
    n_done = 0
    for npz_path in tqdm(npz_files, desc="SceneFlow"):
        traj_idx  = int(npz_path.stem.split("_")[-1])
        save_path = os.path.join(args.save_dir, f"sceneflow_{traj_idx}.npy")
        vis_path  = os.path.join(vis_dir, f"vis_{traj_idx}.png")

        if args.skip_existing and os.path.exists(save_path):
            if not args.visualize or os.path.exists(vis_path):
                continue

        try:
            sf, extras = collect_sceneflow_one_traj(
                str(npz_path),
                env_vec=env_vec,
                base_env=base_env,
                env_id=args.env_id,
                cam_name=args.cam_name,
                device=device,
                return_frames=args.visualize,
            )
            np.save(save_path, sf)
            n_done += 1

            if args.visualize and extras:
                save_visualization(
                    scene_flow  = sf,
                    frame_rgb   = extras["frame_rgb"],
                    frame_depth = extras["frame_depth"],
                    frame_seg   = extras["frame_seg"],
                    id_to_name  = extras["id_to_name"],
                    actor_names = extras["actor_names"],
                    valid_all   = extras["valid_all"],
                    save_path   = vis_path,
                    traj_idx    = traj_idx,
                )
        except Exception:
            import traceback
            print(f"\n[ERROR] {npz_path.name}")
            traceback.print_exc()

    env_vec.close()
    print(f"[SceneFlow] Done. {n_done} trajectories saved to {args.save_dir}")
    print(f"[SceneFlow] Output shape per file: (5, T, H, W, 3)  float32")
    print(f"[SceneFlow]   axis 0 : 5 keyframe refs [0, T//4, T//2, 3T//4, T-1]")
    print(f"[SceneFlow]   axis 1 : timestep t")
    print(f"[SceneFlow]   axis 4 : Δ(x,y,z) in ref-camera OpenCV coords")
    if args.visualize:
        print(f"[SceneFlow] Visualization PNG saved to {vis_dir}")


if __name__ == "__main__":
    main()

# python mikasa_robo_suite/dataset_collectors/collect_sceneflow.py \
#     --env_id ShellGameTouch-v0 \
#     --data_dir data_with_seed/MIKASA-Robo/unbatched/ShellGameTouch-v0 \
#     --save_dir sceneflow/MIKASA-Robo/ShellGameTouch-v0 \
#     --cam_name base_camera \
#     --max_trajectories 5 \
#     --visualize
