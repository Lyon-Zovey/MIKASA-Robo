# MIKASA-Robo 数据采集与点云可视化说明

## 概述

本说明记录了如何将 `ManiSkill dev_wjj` 分支中 `RBSRecordEpisode` 的数据格式和 SAPIEN GUI 点云可视化，完整复刻到 MIKASA-Robo 的 RL rollout 数采脚本 `get_mikasa_robo_datasets.py` 中。

---

## 环境说明

- **MIKASA-Robo**：基于 ManiSkill3 的记忆型机械臂操作 benchmark
- **ManiSkill dev_wjj 分支**（`/home/CNF2026716696/Sim_Data/ManiSkill`）：通过 `pip install -e .` 安装为 editable，是当前 `mikasa-robo` conda 环境中 `mani_skill` 的实际来源
- **conda 环境**：`mikasa-robo`

> ⚠️ **重要**：由于 `mani_skill` 指向 dev_wjj 分支（editable install），该分支缺少部分旧版模块（`mani_skill.viewer`、`mani_skill.agents.robots.xmate3`），已在代码中用 `try/except ImportError` 做了兼容处理。

---

## 数据格式

每条轨迹产生以下输出，与 dev_wjj 的 `RBSRecordEpisode` 格式完全一致：

### 顶层目录（`data/MIKASA-Robo/vis/<env_id>/`）

```
traj_seed0_<timestamp>.rgb+depth+segmentation.pd_joint_delta_pos.physx_cpu.h5
traj_seed0_<timestamp>.rgb+depth+segmentation.pd_joint_delta_pos.physx_cpu.json
camera_data/
```

### `camera_data/traj_<N>/` 目录

| 文件 | 内容 | 形状/格式 |
|---|---|---|
| `rgb.mp4` | 832×480 RGB 视频，16fps | H.264 视频 |
| `depth_video.npy` | 每帧深度图（float16，单位米） | `(T, H, W)` |
| `depth_video_int16mm_dt.b2nd` | 深度压缩（int16mm + XOR delta + blosc2） | blosc2 |
| `seg.npy` / `seg.b2nd` | 每帧分割图（int16，seg_id） | `(T, H, W)` |
| `cam_intrinsics.npy` | 相机内参（每帧，OpenCV 约定） | `(T, 3, 3)` |
| `cam_poses.npy` | 相机外参 cam2world（每帧，OpenGL 约定） | `(T, 4, 4)` |
| `scene_point_flow_ref00000.npy` | SceneFlow，ref=第0帧（相机坐标系） | `(T, H, W, 3)` |
| `scene_point_flow_ref00022.npy` | SceneFlow，ref=第22帧（约轨迹1/4处） | `(T, H, W, 3)` |
| `scene_point_flow_ref00045.npy` | SceneFlow，ref=第45帧（轨迹中点） | `(T, H, W, 3)` |
| `scene_point_flow_ref00068.npy` | SceneFlow，ref=第68帧（约轨迹3/4处） | `(T, H, W, 3)` |
| `scene_point_flow_ref00090.npy` | SceneFlow，ref=最后一帧 | `(T, H, W, 3)` |
| `scene_point_flow_ref*.anchor.npy` | 对应 ref 帧的深度+seg 快照 | — |
| `scene_point_flow_ref*_v3_10b_h265_crf0.mp4/.json` | SceneFlow 压缩版 | H.265 视频 |
| `traj_<N>.h5` | per-episode 详细数据（见下） | HDF5 |

### `traj_<N>.h5` 内部结构

```
traj_0/
  obs/
    agent/          qpos(T,9)  qvel(T,9)
    extra/          tcp_pose(T,7)
    sensor_data/    base_camera/  hand_camera/
  actions(T-1, 8)
  rewards(T-1,)  terminated  truncated  success
  env_states/
    actors/         table-workspace mug_left mug_center mug_right red_ball  各(T,13)
    articulations/  panda_wristcam(T,31)
  id_poses/
    .attrs['1']  = 'link:panda_wristcam/panda_link0'
    .attrs['18'] = 'actor:025_mug-left-0'
    ... （seg_id → 名称 映射）
    16/   position(T,3)  quaternion(T,4)  camera_position(T,3)  camera_quaternion(T,4)
          .attrs: name, seg_id, mesh_file_path, mesh_file_paths_json, geometry_params_json
    ...   （每个 actor 和每个 panda link 各一个 group）
```

---

## SceneFlow 生成原理

SceneFlow 由 `rbs_record/convert_camera_depths.py` 从 `depth_video.npy`、`seg.npy`、`id_poses` 生成：

1. 用 ref 帧的深度图反投影得到每个像素的 3D 点（相机坐标系）
2. 找到该点所属的物体（通过 seg_id），计算该点在物体局部坐标系下的坐标
3. 对每一帧，通过物体的世界位姿将局部坐标转换回世界坐标，再转回相机坐标
4. 输出 `(T, H, W, 3)` 的点云追踪序列，每个像素对应同一个物理点在各帧的相机坐标

**关键修复（相对于 dev_wjj 原版）**：

- 原版 `convert_camera_depths.py` 对所有 anchor 文件都硬编码用 `seg[0]` 和 `pose[0]` 作为参考帧，导致在第 0 帧不可见的物体（如 ShellGameTouch-v0 的杯子，第 0 帧在 z=1000m 处）无法被追踪。
- **修复**：`track_anchor_file_exact` 现在使用 `seg[anchor_idx]` 和 `pose[anchor_idx]` 作为参考。

### 深度压缩格式（`depth_video_int16mm_dt.b2nd`）

- 将 float16 深度（米）乘以 200 转为 int16（0.005m 精度）
- 沿时间轴做 XOR delta 编码（`d[t] = raw[t] XOR raw[t-1]`）
- 用 blosc2 压缩（codec=zstd, filter=bitshuffle）
- 解压时需先 blosc2 解码，再逐帧反向 XOR 累加，再除以 200 还原米值

---

## 两种可视化模式

### Pass 1 — 数据采集（无点云）

首次运行，采集数据并生成 SceneFlow .npy：

```bash
python -m mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets \
  --env_id=ShellGameTouch-v0 \
  --visualize
```

输出位于 `data/MIKASA-Robo/vis/ShellGameTouch-v0/camera_data/traj_<N>/`

### Pass 2 — 点云重播可视化

用 Pass 1 生成的 `.npy` 重播，在 SAPIEN GUI 中叠加点云：

```bash
python -m mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets \
  --env_id=ShellGameTouch-v0 \
  --visualize \
  --no-postprocess-camera-data \
  --pointflow-npy=data/MIKASA-Robo/vis/ShellGameTouch-v0/camera_data/traj_<N>/scene_point_flow_ref00022.npy \
  --pointflow-exclude-names actor:table scene-builder actor:ground ground-plane \
  --pointflow-stride=2 \
  --pointflow-color-mode=id
```

> **为何用 ref00022 而不是 ref00000**：ShellGameTouch-v0 的三个杯子在第 0 帧位于 z=1000m（场外），约第 10 帧才落到桌面。用 ref00022 可确保三个杯子都被正确追踪。

### Live 模式 — 实时点云（无需 .npy）

```bash
python -m mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets \
  --env_id=ShellGameTouch-v0 \
  --visualize \
  --live-pcd \
  --live-pcd-stride=2 \
  --live-pcd-far-clip=5.0
```

---

## 点云可视化参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--pointflow-npy PATH` | 预计算的 SceneFlow .npy 路径 | 无 |
| `--pointflow-exclude-names A B ...` | 按名称排除的 actor/link（substring 匹配） | 无 |
| `--pointflow-keep-names A B ...` | 只保留匹配名称的 actor/link | 无 |
| `--pointflow-stride N` | 像素网格降采样步长（1=全量，2=1/4点） | 1 |
| `--pointflow-cam-radius-clip R` | 丢弃距相机原点超过 R 米的点（0=不限制） | 0.0 |
| `--pointflow-color-mode MODE` | 着色方式（见下） | `z` |
| `--live-pcd` | 启用实时 RGB-D 反投影点云 | False |
| `--live-pcd-stride N` | 实时点云降采样步长 | 2 |
| `--live-pcd-far-clip M` | 实时点云最大深度（米） | 5.0 |

### `--pointflow-color-mode` 选项

| 值 | 效果 |
|---|---|
| `z` | dev_wjj 原版：按世界 z 坐标红→绿渐变 |
| `id` | 每个 seg_id 分配独立色相（黄金比例 HSV 旋转，无碰撞） |
| `rgb` | 从 rgb.mp4 的 anchor 帧取原始像素颜色（写实点云） |

### 推荐过滤参数（ShellGameTouch-v0）

ShellGameTouch-v0 的 `table-workspace` + `ground` 覆盖约 95% 的像素（far=100m 导致），不过滤则点云变成一整张幕布：

```bash
--pointflow-exclude-names actor:table scene-builder actor:ground ground-plane
--pointflow-stride=2
```

过滤后约剩 7300 点（robot arm + 3 mugs + red ball），视觉效果类似 dev_wjj demo 环境。

---

## 与 dev_wjj RBSRecordEpisode 的差异

| 项目 | dev_wjj RBSRecordEpisode | 本实现 |
|---|---|---|
| 采集方式 | replay 轨迹 | RL rollout 实时采集 |
| sim_backend | physx_cpu（自动切换） | physx_cpu（visualize 分支强制） |
| 数据格式 | 完全一致 | 完全一致 |
| 点云默认着色 | z-gradient（红绿） | z-gradient（默认），可选 id/rgb |
| SceneFlow 参考帧 bug | 所有 anchor 都用 frame 0 | 已修复：使用 anchor_idx 帧 |

---

## 已修复的 Bug

### 1. `convert_camera_depths.py`：anchor 帧硬编码为 frame 0
- **问题**：`track_anchor_file_exact` 始终用 `seg[0]` 和 `pose[0]` 作为参考，导致第 0 帧不可见的物体（ShellGameTouch-v0 的杯子）在所有 anchor 文件中都缺失。
- **修复**：`_load_tracking_context` 返回完整 `seg_all (T,H,W)`；`track_anchor_file_exact` 接受 `anchor_idx` 参数并使用 `seg_all[anchor_idx]` 和 `pose[anchor_idx]`。

### 2. 深度解压：XOR delta 编码未逆向
- **问题**：`depth_video_int16mm_dt.b2nd` 使用 XOR delta 编码（`d[t] = raw[t] XOR raw[t-1]`），直接读取原始值会导致 t>0 的帧深度错误。
- **修复**：解压后逐帧做 XOR 前缀和还原原始 int16 值，再除以 200 得到米值。

### 3. `mani_skill.viewer` 模块缺失（dev_wjj 兼容性）
- **问题**：dev_wjj 分支将 `create_viewer` 移至 `mani_skill.utils.sapien_utils`，旧路径 `mani_skill.viewer` 不存在。
- **修复**：`_install_viewer_hook` 中用 `try/except ImportError` 兼容两种路径。

### 4. `mani_skill.agents.robots.xmate3` 缺失（dev_wjj 兼容性）
- **问题**：dev_wjj 分支移除了 xmate3 机器人，但 `shell_game_push/touch/pick.py` 在顶层 import 它。
- **修复**：三个文件均改为 `try/except ImportError`，`Xmate3Robotiq = None`（只在 `robot_uids=="xmate3_robotiq"` 分支才用，ShellGame 系列不受影响）。

### 5. 点云视觉效果："一整张幕布"问题
- **问题**：MIKASA-Robo 环境中 table 和 ground 的 seg_id 非零，占 ~95% 像素，全部投影后形成巨大平面遮挡。
- **修复**：`_shrink_pointflow_to_match_devwjj_look` 通过 `id_poses.attrs` 按名称过滤，默认排除 `actor:table`、`actor:ground` 等基础设施。

---

## 数据需求完成情况

| 需求 | 状态 | 字段/文件 |
|---|---|---|
| 832×480 RGB 视频 fps=16 | ✅ | `rgb.mp4` |
| 每帧深度图 | ✅ | `depth_video.npy` / `depth_video_int16mm_dt.b2nd` |
| 每帧分割图 | ✅ | `seg.npy` / `seg.b2nd` |
| 相机内参（每帧） | ✅ | `cam_intrinsics.npy` (T,3,3) |
| 相机外参（每帧） | ✅ | `cam_poses.npy` (T,4,4) |
| 物体世界位姿（per-actor、per-link） | ✅ | `id_poses/{sid}/position+quaternion` |
| 相机坐标系位姿 | ✅ | `id_poses/{sid}/camera_position+camera_quaternion` |
| mesh 文件路径及几何参数 | ✅ | `id_poses/{sid}.attrs` |
| SceneFlow（5个关键帧） | ✅ | `scene_point_flow_ref00000/22/45/68/90.npy` |
| SceneFlow 压缩 | ✅ | `*_v3_10b_h265_crf0.mp4/.json` |
| 夹爪抓取物体标注 | ❌ 待补充 | 需添加 `grasp_contact` 字段 |
| Object Flow（分物体独立存储） | ❌ 待确认需求 | 当前 SceneFlow 已含追踪信息 |
