"""
collect_live_pcd.py  –  MIKASA-Robo 在线单条轨迹采集 + SAPIEN 实时分割点云可视化
================================================================================

使用 oracle RL agent 在线采集 **一条** 轨迹，同时在 SAPIEN 的交互式 3D viewer 中
实时更新整个 scene 的点云，点的颜色按 **actor-level 实例分割** 着色（每个物体
一种固定颜色），可立即分辨机械臂、桌面、各个杯子等。

采集结束后保存：
  <save_dir>/traj_seed<N>.npz
    rgb         (T, H, W, 3)  uint8   —— 相机 RGB
    depth       (T, H, W, 1)  int16   —— 深度 [mm]
    pcd_world   (T, H, W, 3)  float32 —— 反投影点云（世界坐标，单位 m）
    seg         (T, H, W, 4)  uint32  —— SAPIEN 分割图（ch0=mesh, ch1=actor）

可视化说明
----------
  SAPIEN viewer 中，点云颜色 = 实例分割颜色（非 RGB 纹理）。
  启动时会在终端打印 actor_id → actor_name 映射表，方便对照颜色。

用法示例
--------
  python mikasa_robo_suite/dataset_collectors/collect_live_pcd.py \\
      --env-id ShellGameTouch-v0 \\
      --ckpt-dir . \\
      --save-dir ./live_pcd_data \\
      --seed 0 \\
      --stride 4

参数说明
--------
  --stride  N   可视化时的空间下采样步长；stride=4 表示每隔 4 像素取一个点
                ( 128×128 图 → 32×32 = 1024 点，适合实时渲染 )
"""

from __future__ import annotations

import os
import sys

# 将项目根目录加入 Python 路径（兼容从任意目录运行）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import sapien
import sapien.render as sr

from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from mani_skill.viewer import create_viewer
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from baselines.ppo.ppo_memtasks import AgentStateOnly, FlattenRGBDObservationWrapper
from mikasa_robo_suite.memory_envs import *          # 注册所有环境
from mikasa_robo_suite.utils.wrappers import *       # 注册所有 wrapper
from mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets import env_info


# ──────────────────────────────────────────────────────────────────────────────
# 坐标变换工具
# ──────────────────────────────────────────────────────────────────────────────

def depth_to_pts_cam(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    将深度图反投影到 OpenCV 相机坐标系。

    Parameters
    ----------
    depth_m : (H, W) float32  z-depth [metres]，0 = 无效
    K       : (3, 3) float32  相机内参矩阵（OpenCV 约定）

    Returns
    -------
    (H, W, 3) float32  OpenCV 相机坐标系下的 3-D 点
    """
    H, W = depth_m.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = depth_m.astype(np.float32)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def pts_cam_to_world(pts_cam: np.ndarray, cam2world_gl: np.ndarray) -> np.ndarray:
    """
    OpenCV 相机坐标系 → 世界坐标系。

    cam2world_gl 是 OpenGL 约定的 4×4 相机到世界变换矩阵，
    y 轴朝上、z 轴朝后（与 OpenCV 的 y 下 z 前相反）。

    Parameters
    ----------
    pts_cam     : (..., 3) float32  OpenCV 相机坐标系下的点
    cam2world_gl: (4, 4)  float32  OpenGL cam→world 矩阵

    Returns
    -------
    (..., 3) float32  世界坐标系下的点
    """
    sh = pts_cam.shape
    p = pts_cam.reshape(-1, 3).astype(np.float32)
    # OpenCV → OpenGL：翻转 y 和 z 轴
    p_gl = p * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    ones = np.ones((len(p_gl), 1), dtype=np.float32)
    ph = np.concatenate([p_gl, ones], axis=1)            # (N, 4) 齐次坐标
    world = (cam2world_gl.astype(np.float32) @ ph.T).T   # (N, 4)
    return world[:, :3].reshape(sh)


# ──────────────────────────────────────────────────────────────────────────────
# SAPIEN 点云注入（Monkey-patch 方案）
# ──────────────────────────────────────────────────────────────────────────────
#
# 背景：ManiSkill 在 gym.make() 内部调用 BaseEnv.__init__() → reset() →
#   _reconfigure() → [_load_scene, _setup_sensors] → _after_reconfigure()
#   → get_obs() → update_render()                ← 此处创建 render_system_group
#
# 之后 SAPIEN 禁止向任何场景添加新实体（全局 GPU render context 共享修改标记）。
#
# 解决方案：在 BaseEnv._after_reconfigure() 里注入点云实体。
#   此钩子在 _load_scene() 之后、get_obs() 之前运行，
#   正好是可以安全添加实体的唯一窗口。
# ──────────────────────────────────────────────────────────────────────────────

_PCD_CAPACITY = 128 * 128   # 最多点数（=图像全像素）


def install_pcd_hook(capacity: int = _PCD_CAPACITY) -> None:
    """
    Monkey-patch BaseEnv._reconfigure，在场景封锁（render_system_group 创建）
    之前注入 RenderPointCloudComponent。需在第一次 gym.make() 之前调用。

    技术背景
    --------
    ManiSkill 的 render_system_group 在第一次 get_obs() → update_render() 时
    创建，之后禁止向 SAPIEN 场景添加任何实体。
    _reconfigure() 本身不被子类重写（_after_reconfigure 会被重写），
    因此在 _reconfigure 末尾、_after_reconfigure 之前注入是最安全的窗口。
    """
    from mani_skill.envs.sapien_env import BaseEnv

    if getattr(BaseEnv, "_pcd_hook_installed", False):
        return

    _orig = BaseEnv._reconfigure

    def _hooked(env_self, options=dict()):
        _orig(env_self, options)
        # _reconfigure 结束后、get_obs() / update_render() 之前：安全添加实体。
        # 注意：ShellGame 等任务 reconfiguration_freq=1，每次 reset 都重建场景，
        # 因此必须每次都向 *新* sub_scenes[0] 注入新实体（不能用旧引用）。
        if not hasattr(env_self.scene, "sub_scenes"):
            return
        # ① 注入点云实体（预填哑点避免 "no vertex positions" 错误）
        ent = sapien.Entity()
        comp = sr.RenderPointCloudComponent(capacity)
        comp.set_vertices(np.zeros((1, 3), dtype=np.float32))
        comp.set_attribute("color", np.zeros((1, 4), dtype=np.float32))
        ent.add_component(comp)
        env_self.scene.sub_scenes[0].add_entity(ent)
        env_self._pcd_comp = comp   # 每次更新引用

        # ② 对 render_mode="human" 的环境：在 update_render() 封锁场景之前
        #    提前初始化 SAPIEN viewer，使 viewer 内部实体包含在 render_system_group 中。
        #    否则首次调用 env.render() 时 viewer.set_scene() 会在封锁后修改场景。
        if (getattr(env_self, "render_mode", None) == "human"
                and env_self._viewer is None):
            env_self._viewer = create_viewer(env_self._viewer_camera_config)
            env_self._setup_viewer()

    BaseEnv._reconfigure = _hooked
    BaseEnv._pcd_hook_installed = True


class LivePointCloudVis:
    """
    持有指向 base_env._pcd_comp 的引用，每帧调用 update() 刷新点云。
    实体已在 gym.make() 期间由 install_pcd_hook() 注入，无需再修改场景。
    """

    def __init__(self, base_env, capacity: int = _PCD_CAPACITY):
        if not hasattr(base_env, "_pcd_comp"):
            raise RuntimeError(
                "base_env 没有 _pcd_comp 属性，请在 gym.make() 之前调用 install_pcd_hook()"
            )
        self._comp: sr.RenderPointCloudComponent = base_env._pcd_comp
        self._cap = capacity

    def update(self, positions_world: np.ndarray, colors_rgb_01: np.ndarray) -> None:
        """
        Parameters
        ----------
        positions_world : (N, 3) float32  世界坐标 [m]
        colors_rgb_01   : (N, 3) float32  RGB 颜色，[0, 1] 范围
        """
        N = min(len(positions_world), self._cap)
        if N == 0:
            # 必须保留至少 1 个顶点，否则 SAPIEN viewer 报 "no vertex positions"
            # 设一个透明哑点（alpha=0），视觉上不可见
            self._comp.set_vertices(np.zeros((1, 3), dtype=np.float32))
            self._comp.set_attribute("color", np.zeros((1, 4), dtype=np.float32))
            return
        pos = np.ascontiguousarray(positions_world[:N], dtype=np.float32)
        col = np.ascontiguousarray(colors_rgb_01[:N], dtype=np.float32)
        alpha = np.ones((N, 1), dtype=np.float32)
        col_rgba = np.concatenate([col, alpha], axis=1)   # (N, 4) RGBA
        self._comp.set_vertices(pos)
        self._comp.set_attribute("color", col_rgba)


# ──────────────────────────────────────────────────────────────────────────────
# 传感器数据读取
# ──────────────────────────────────────────────────────────────────────────────

def _to_np(x) -> np.ndarray:
    """将 Tensor / numpy / list 统一转为 ndarray。"""
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def get_camera_frame(base_env, cam_name: str):
    """
    读取一帧相机数据（RGB + depth + segmentation + 相机参数）。

    Returns
    -------
    rgb   : (H, W, 3)  uint8   —— RGB 图像
    depth : (H, W, 1)  int16   —— 深度图 [mm]
    seg   : (H, W, 4)  uint32  —— 分割图（ch0=mesh-level, ch1=actor-level）
    K     : (3, 3)     float32 —— 相机内参矩阵
    c2w   : (4, 4)     float32 —— cam2world_gl（OpenGL 约定）
    全部返回 None 时表示该帧数据不可用。
    """
    obs = base_env.get_obs()
    sd = obs.get("sensor_data", {})
    sp = obs.get("sensor_param", {})

    if cam_name not in sd or cam_name not in sp:
        available = list(sd.keys())
        print(f"  [warn] cam '{cam_name}' not found. Available: {available}")
        return None, None, None, None, None

    cam_d = sd[cam_name]
    cam_p = sp[cam_name]

    rgb_raw   = cam_d.get("rgb")
    depth_raw = cam_d.get("depth")
    seg_raw   = cam_d.get("segmentation")
    K_raw     = cam_p.get("intrinsic_cv")
    c2w_raw   = cam_p.get("cam2world_gl")

    if any(x is None for x in (rgb_raw, depth_raw, K_raw, c2w_raw)):
        return None, None, None, None, None

    # squeeze batch dim (idx=0)
    rgb   = _to_np(rgb_raw)[0]                                       # (H, W, 3)
    depth = _to_np(depth_raw)[0]                                     # (H, W, 1)
    seg   = _to_np(seg_raw)[0] if seg_raw is not None else None      # (H, W, 4)
    K     = _to_np(K_raw)[0]                                         # (3, 3)
    c2w   = _to_np(c2w_raw)[0]                                       # (4, 4)
    return rgb, depth, seg, K, c2w


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation → 点云颜色映射
# ──────────────────────────────────────────────────────────────────────────────

# 16 种视觉上可区分的高饱和颜色（用于实例着色，索引 = actor_id % 16）
_SEG_PALETTE = np.array([
    [0.55, 0.55, 0.55],   #  0: 灰（背景/地面）
    [0.95, 0.15, 0.15],   #  1: 红
    [0.15, 0.90, 0.15],   #  2: 绿
    [0.15, 0.35, 1.00],   #  3: 蓝
    [1.00, 0.88, 0.10],   #  4: 黄
    [1.00, 0.15, 0.95],   #  5: 品红
    [0.10, 0.95, 0.95],   #  6: 青
    [1.00, 0.55, 0.10],   #  7: 橙
    [0.60, 0.10, 1.00],   #  8: 紫
    [0.10, 0.60, 0.30],   #  9: 深绿
    [1.00, 0.70, 0.70],   # 10: 粉红
    [0.70, 1.00, 0.70],   # 11: 浅绿
    [0.70, 0.70, 1.00],   # 12: 浅蓝
    [0.90, 0.60, 0.10],   # 13: 棕
    [0.40, 0.90, 0.90],   # 14: 水蓝
    [0.90, 0.40, 0.60],   # 15: 玫红
], dtype=np.float32)


def seg_actor_to_colors(seg_actor_flat: np.ndarray) -> np.ndarray:
    """
    将 actor-level 分割 ID 数组映射为 RGB 颜色（[0, 1] 范围）。

    Parameters
    ----------
    seg_actor_flat : (N,) uint32  actor 分割 ID（可以含 0=背景）

    Returns
    -------
    (N, 3) float32  对应的 RGB 颜色
    """
    idx = (seg_actor_flat.astype(np.int64)) % len(_SEG_PALETTE)
    return _SEG_PALETTE[idx]


def build_actor_id_map(base_env) -> dict[int, str]:
    """
    构建 actor_id → actor_name 映射表，并打印到终端供对照。

    Returns
    -------
    dict  {int actor_id : str actor_name}
    """
    # ManiSkillScene 通过 sub_scenes[0] 访问底层 sapien.Scene
    sapien_scene = base_env.scene.sub_scenes[0]
    actors = sapien_scene.get_all_actors()
    id_map: dict[int, str] = {}
    print("\n  ── Actor 分割颜色映射 ───────────────────────────────────────")
    print(f"  {'actor_id':>10}  {'color_idx':>9}  actor_name")
    print("  " + "-" * 50)
    for i, actor in enumerate(actors):
        # SAPIEN 3: Entity 使用属性而非方法（无 get_id / get_name）
        name = actor.name  # property
        try:
            # per_scene_id 是 SAPIEN 3 分割图 ch1 中存储的 entity ID
            aid = int(actor.per_scene_id)
        except AttributeError:
            aid = i          # fallback：用枚举索引
        cidx = aid % len(_SEG_PALETTE)
        id_map[aid] = name
        r, g, b = _SEG_PALETTE[cidx]
        print(f"  {aid:>10}  {cidx:>9}  {name}  (rgb≈[{r:.2f},{g:.2f},{b:.2f}])")
    print("  " + "-" * 50 + "\n")
    return id_map


# ──────────────────────────────────────────────────────────────────────────────
# 环境创建辅助
# ──────────────────────────────────────────────────────────────────────────────

def _build_state_env(env_id: str, wrappers_list) -> ManiSkillVectorEnv:
    """构建供 agent 使用的 state-only 环境（无 viewer）。"""
    env = gym.make(env_id, num_envs=1,
                   obs_mode="state",
                   render_mode="sensors",
                   sim_backend="gpu",
                   reward_mode="normalized_dense")
    for WC, kw in wrappers_list:
        env = WC(env, **kw)
    env = FlattenRGBDObservationWrapper(env,
                                        rgb=False, depth=False,
                                        state=True, oracle=False, joints=False)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(env, 1, ignore_terminations=True)


def _build_vis_env(env_id: str, wrappers_list) -> ManiSkillVectorEnv:
    """构建带 SAPIEN 交互式 viewer 的视觉环境（RGB + depth + seg）。"""
    env = gym.make(env_id, num_envs=1,
                   obs_mode="rgb+depth+segmentation",
                   render_mode="human",         # ← 打开 SAPIEN 交互式 viewer
                   sim_backend="gpu",
                   reward_mode="normalized_dense")
    for WC, kw in wrappers_list:
        env = WC(env, **kw)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(env, 1, ignore_terminations=True)


def _find_checkpoint(ckpt_dir: str, env_id: str) -> str:
    base = (Path(ckpt_dir) / "oracle_checkpoints" / "ppo_memtasks"
            / "state" / "normalized_dense" / env_id)
    ckpts = sorted(base.glob("**/final_success_ckpt.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {base}")
    return str(ckpts[0])


# ──────────────────────────────────────────────────────────────────────────────
# CLI 参数
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    env_id:   str = "ShellGameTouch-v0"
    """目标任务的 Gym 环境 ID"""
    ckpt_dir: str = "."
    """包含 oracle_checkpoints/ 子目录的根路径"""
    save_dir: str = "./live_pcd_data"
    """保存输出 .npz 的目录"""
    cam_name: str = "base_camera"
    """用于采集数据的相机名称"""
    seed: int = 0
    """环境初始化随机种子"""
    stride: int = 4
    """点云可视化的空间下采样步长（stride=4 → 约 1024 点）"""
    depth_min_m: float = 0.05
    """过滤深度下限 [m]，小于此值的点视为无效"""
    depth_max_m: float = 5.0
    """过滤深度上限 [m]，大于此值的点视为无效"""


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import tyro
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"env_id  : {args.env_id}")
    print(f"seed    : {args.seed}")
    print(f"cam     : {args.cam_name}")
    print(f"stride  : {args.stride}")
    print(f"save_dir: {args.save_dir}")
    print("=" * 60)

    # ── 1. 获取 wrappers 列表 & episode 时长 ─────────────────────────────────
    wl_state, episode_timeout = env_info(args.env_id)
    wl_vis,   _               = env_info(args.env_id)  # 独立拷贝

    # ── 2. 注入钩子（必须在 gym.make() 之前）────────────────────────────────
    # SAPIEN render_system_group 在 gym.make() 内部第一次 get_obs() 时创建，
    # 之后不允许向场景加入新实体。钩子在 _after_reconfigure() 中（get_obs() 之前）
    # 安全注入 RenderPointCloudComponent。
    install_pcd_hook(_PCD_CAPACITY)

    # ── 3. 构建两个环境（state env 供 agent 推断，vis env 供可视化+采集）────
    print("\n[1/4] Building state env...")
    env_state = _build_state_env(args.env_id, wl_state)

    print("[2/4] Building visual env (will open SAPIEN viewer)...")
    env_vis   = _build_vis_env(args.env_id, wl_vis)
    base_env  = env_vis.base_env

    # ── 4. 加载 oracle agent ────────────────────────────────────────────────
    ckpt = _find_checkpoint(args.ckpt_dir, args.env_id)
    print(f"[3/4] Loading checkpoint: {ckpt}")
    agent = AgentStateOnly(env_state).to(device)
    agent.load_state_dict(torch.load(ckpt, map_location=device))
    agent.eval()

    # ── 5. 重置两个环境 ───────────────────────────────────────────────────────
    print(f"[4/4] Resetting both envs (seed={args.seed})...")
    obs_state, _ = env_state.reset(seed=[args.seed])
    _,          _ = env_vis.reset(seed=[args.seed])

    if base_env.gpu_sim_enabled:
        base_env.scene._gpu_fetch_all()

    # ── 6. 绑定点云可视化组件 ────────────────────────────────────────────────
    # _pcd_comp 已由 install_pcd_hook() 在 gym.make() 内部注入，
    # 无需再修改 SAPIEN 场景。
    pc_vis = LivePointCloudVis(base_env, capacity=_PCD_CAPACITY)

    # 打印 actor_id → 颜色映射（帮助对照 viewer 中各颜色对应的物体）
    actor_id_map = build_actor_id_map(base_env)

    print("  SAPIEN viewer 已开启，点云颜色 = 实例分割颜色（每个物体一种颜色）")
    print(f"  开始采集 {episode_timeout} 步轨迹...\n")

    # ── 7. 轨迹采集主循环 ─────────────────────────────────────────────────────
    rgb_frames:   list[np.ndarray] = []   # (H, W, 3) uint8
    depth_frames: list[np.ndarray] = []   # (H, W, 1) int16
    pcd_frames:   list[np.ndarray] = []   # (H, W, 3) float32
    seg_frames:   list[np.ndarray] = []   # (H, W, 4) uint32

    for t in range(episode_timeout):
        # 7a. Agent 推断动作
        with torch.no_grad():
            for k in obs_state:
                obs_state[k] = obs_state[k].to(device)
            action = agent.get_action(obs_state, deterministic=True)

        # 7b. 两个环境同步 step
        obs_state, _, _, _, _ = env_state.step(action)
        env_vis.step(action)

        # 7c. 同步 GPU 物理状态到 CPU
        if base_env.gpu_sim_enabled:
            base_env.scene._gpu_fetch_all()

        # 7d. 读取传感器数据（含 segmentation）
        rgb, depth, seg, K, c2w = get_camera_frame(base_env, args.cam_name)
        if rgb is None:
            env_vis.render()
            continue

        # 7e. 深度图：int16 mm → float32 m
        depth_m = depth[..., 0].astype(np.float32) / 1000.0   # (H, W)

        # 7f. 反投影：像素 → OpenCV 相机坐标系 → 世界坐标系
        pts_cam   = depth_to_pts_cam(depth_m, K)               # (H, W, 3)
        pts_world = pts_cam_to_world(pts_cam, c2w)             # (H, W, 3)

        # 7g. 保存全分辨率数据
        rgb_frames.append(rgb)
        depth_frames.append(depth)
        pcd_frames.append(pts_world.astype(np.float32))
        if seg is not None:
            seg_frames.append(seg.astype(np.uint32))

        # 7h. 下采样 + 有效深度过滤
        s = args.stride
        pts_ds  = pts_world[::s, ::s].reshape(-1, 3).astype(np.float32)   # (M, 3)
        d_ds    = depth_m[::s, ::s].ravel()
        valid   = (d_ds > args.depth_min_m) & (d_ds < args.depth_max_m)

        # 7i. 按 actor 分割 ID 着色
        # ManiSkill 返回 (H,W,1) 直接是 actor ID；
        # SAPIEN 原始纹理是 (H,W,4) ch1=actor，ch0=mesh。
        # 自动选：有 >1 个通道就取 ch1，否则取 ch0。
        seg_ch = (1 if (seg is not None and seg.shape[2] > 1) else 0)
        if seg is not None:
            seg_actor_ds = seg[::s, ::s, seg_ch].ravel()                  # (M,)
            seg_colors   = seg_actor_to_colors(seg_actor_ds)               # (M, 3)
            vis_colors   = seg_colors
        else:
            # fallback: 用 RGB 着色
            vis_colors = rgb[::s, ::s].reshape(-1, 3).astype(np.float32) / 255.0

        # 7j. 推送到 SAPIEN 点云 entity 并刷新 viewer
        pc_vis.update(pts_ds[valid], vis_colors[valid])
        env_vis.render()

        # 进度日志
        n_unique_actors = (int(np.unique(seg[::s, ::s, seg_ch]).size)
                           if seg is not None else 0)
        d_valid = depth_m[depth_m > args.depth_min_m]
        d_min   = float(d_valid.min()) if len(d_valid) > 0 else 0.0
        d_max   = float(depth_m.max())
        print(f"  t={t:3d}  vis_pts={valid.sum():5d}  actors={n_unique_actors}  "
              f"depth=[{d_min:.2f}, {d_max:.2f}] m")

    # ── 8. 保存采集到的数据 ───────────────────────────────────────────────────
    T = len(rgb_frames)
    if T == 0:
        print("[warn] 未采集到任何帧，不保存")
    else:
        out_path = Path(args.save_dir) / f"traj_seed{args.seed}.npz"
        save_dict: dict = dict(
            rgb=np.stack(rgb_frames),           # (T, H, W, 3)  uint8
            depth=np.stack(depth_frames),       # (T, H, W, 1)  int16
            pcd_world=np.stack(pcd_frames),     # (T, H, W, 3)  float32
        )
        if seg_frames:
            save_dict["seg"] = np.stack(seg_frames)  # (T, H, W, 4)  uint32

        np.savez_compressed(str(out_path), **save_dict)

        rgb_arr = save_dict["rgb"]
        dep_arr = save_dict["depth"]
        pcd_arr = save_dict["pcd_world"]
        print(f"\n已保存 {T} 帧 → {out_path}")
        print(f"  rgb      : {rgb_arr.shape}  dtype=uint8")
        print(f"  depth    : {dep_arr.shape}  dtype=int16 [mm]")
        print(f"  pcd_world: {pcd_arr.shape}  dtype=float32 [m]")
        if "seg" in save_dict:
            seg_arr  = save_dict["seg"]
            seg_uch  = 1 if seg_arr.shape[3] > 1 else 0
            n_actors = len(np.unique(seg_arr[:, :, :, seg_uch]))
            print(f"  seg      : {seg_arr.shape}  dtype=uint32"
                  f"  (actor ch={seg_uch})  unique_actors≈{n_actors}")

    env_state.close()
    env_vis.close()


if __name__ == "__main__":
    main()
