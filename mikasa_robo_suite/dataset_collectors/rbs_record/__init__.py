"""Local copy of dev_wjj's RBSRecordEpisode wrapper and its camera-data
postprocess scripts, vendored into MIKASA-Robo so that RL rollouts produce
trajectories in the exact same layout / fields as dev_wjj's `rbs_replay`:

    <output_dir>/trajectory.h5
    <output_dir>/trajectory.json
    <output_dir>/camera_data/traj_<i>/rgb.mp4
    <output_dir>/camera_data/traj_<i>/depth_video.npy
    <output_dir>/camera_data/traj_<i>/seg.npy
    <output_dir>/camera_data/traj_<i>/cam_poses.npy
    <output_dir>/camera_data/traj_<i>/cam_intrinsics.npy
    <output_dir>/camera_data/traj_<i>/traj_<i>.h5         (id_poses, id_geometry_meta)
    (after postprocess) scene_point_flow_ref*.b2nd, depth.b2nd, seg.b2nd

The wrapper and the 4 postprocess scripts are verbatim from the dev_wjj branch
of ManiSkill; the only edit is `_postprocess_camera_dir`'s script-lookup path,
which now resolves the 4 postprocess scripts next to `rbs_record.py` instead
of at the ManiSkill repo root.
"""

from .rbs_record import RBSRecordEpisode

__all__ = ["RBSRecordEpisode"]
