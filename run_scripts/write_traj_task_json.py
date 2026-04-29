#!/usr/bin/env python3
"""
write_traj_task_json.py  —  为每条轨迹目录写入 traj_task.json

traj_task.json 内容：
  {
    "task_id":   "InterceptFast-v0",
    "traj_name": "traj_0",
    "actors": [
      {"seg_id": 16, "name": "actor:table-workspace[env0]"},
      {"seg_id": 18, "name": "actor:red_ball[env0]"},
      ...
    ],
    "links": [
      {"seg_id": 1, "name": "link:panda_wristcam/panda_link0[env0]"},
      ...
    ]
  }

task_id 来源（按优先级）：
  1. dataset-dir 下的 companion JSON（env_info.env_id）
  2. dataset-dir 目录名去掉末尾 -<num> 后缀（e.g. InterceptFast-v0-256 → InterceptFast-v0）
  3. dataset-dir 目录名原样

actors / links 来自 traj_N.h5 里的 id_poses.attrs。

用法
----
# 预览（不写文件）
python run_scripts/write_traj_task_json.py --dataset-dir /path/to/env-dataset

# 写入
python run_scripts/write_traj_task_json.py --dataset-dir /path/to/env-dataset --apply

# 覆盖已存在的 traj_task.json
python run_scripts/write_traj_task_json.py --dataset-dir /path/to/env-dataset --apply --overwrite
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import h5py

# ---------------------------------------------------------------------------
# 目录扫描（与 rename_h5_objects.py 保持一致）
# ---------------------------------------------------------------------------

_TRAJ_RE     = re.compile(r"^traj_(\d+)$")
_TRAJ_ANY_RE = re.compile(r"^traj_(\d+)(?:_.+)?$")


def _sort_key(p: Path) -> tuple[int, str]:
    m = _TRAJ_ANY_RE.match(p.name)
    return (int(m.group(1)), p.name) if m else (10**9, p.name)


def get_traj_dirs(dataset_dir: Path) -> list[Path]:
    """返回 camera_data/ 下所有 traj_N 及 traj_N_p... 目录（含 .h5 文件的）。"""
    camera_data = dataset_dir / "camera_data"
    if not camera_data.is_dir():
        raise FileNotFoundError(f"camera_data 目录不存在: {camera_data}")
    dirs = []
    for d in sorted(camera_data.iterdir(), key=_sort_key):
        if not d.is_dir() or not _TRAJ_ANY_RE.match(d.name):
            continue
        dirs.append(d)
    return dirs


def find_traj_h5(traj_dir: Path) -> Optional[Path]:
    """在 traj 目录里找第一个 .h5 文件。"""
    # 优先 traj_N.h5（纯标准目录）
    m = _TRAJ_ANY_RE.match(traj_dir.name)
    if m:
        traj_n = f"traj_{m.group(1)}.h5"
        p = traj_dir / traj_n
        if p.is_file():
            return p
    candidates = sorted(traj_dir.glob("*.h5"))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# task_id 解析
# ---------------------------------------------------------------------------

def _task_id_from_json(dataset_dir: Path) -> Optional[str]:
    """从 dataset_dir 下 companion JSON 里读 env_info.env_id。"""
    for jp in dataset_dir.glob("*.json"):
        try:
            with open(jp) as f:
                d = json.load(f)
            tid = d.get("env_info", {}).get("env_id") or d.get("env_id")
            if tid:
                return str(tid)
        except Exception:
            continue
    return None


def _task_id_from_dirname(dataset_dir: Path) -> str:
    """去掉目录名末尾的 -<数字> 后缀，得到 task id。
    例：InterceptFast-v0-256 → InterceptFast-v0
        ShellGameTouch-v0    → ShellGameTouch-v0
    """
    name = dataset_dir.name
    return re.sub(r"-\d+$", "", name)


def resolve_task_id(dataset_dir: Path) -> str:
    tid = _task_id_from_json(dataset_dir)
    if tid:
        return tid
    return _task_id_from_dirname(dataset_dir)


# ---------------------------------------------------------------------------
# id_poses 读取
# ---------------------------------------------------------------------------

def _find_id_poses(f: h5py.File) -> Optional[h5py.Group]:
    for key in f.keys():
        if _TRAJ_RE.match(key) and "id_poses" in f[key]:
            return f[key]["id_poses"]  # type: ignore[return-value]
    if "id_poses" in f:
        return f["id_poses"]  # type: ignore[return-value]
    return None


def read_actors_links(h5_path: Path) -> tuple[list[dict], list[dict]]:
    """返回 (actors, links)，每项 {'seg_id': int, 'name': str}。"""
    actors: list[dict] = []
    links:  list[dict] = []
    with h5py.File(h5_path, "r") as f:
        id_poses = _find_id_poses(f)
        if id_poses is None:
            return actors, links
        entries = sorted(
            ((int(k), str(v)) for k, v in id_poses.attrs.items()),
            key=lambda x: x[0],
        )
    for seg_id, name in entries:
        entry = {"seg_id": seg_id, "name": name}
        if name.startswith("actor:"):
            actors.append(entry)
        elif name.startswith("link:"):
            links.append(entry)
    return actors, links


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def run(dataset_dir: Path, apply: bool, overwrite: bool) -> None:
    task_id = resolve_task_id(dataset_dir)
    traj_dirs = get_traj_dirs(dataset_dir)

    if not traj_dirs:
        print("[ERROR] 没有找到任何 traj 目录。")
        sys.exit(1)

    mode_label = "[APPLY]" if apply else "[DRY-RUN]"
    print(f"{mode_label}  dataset-dir : {dataset_dir}")
    print(f"  task_id     : {task_id}")
    print(f"  traj 目录数  : {len(traj_dirs)}")
    print()

    written = skipped = errors = 0

    for traj_dir in traj_dirs:
        out_path = traj_dir / "traj_task.json"

        if out_path.exists() and not overwrite:
            print(f"  [SKIP]  {traj_dir.name}/traj_task.json 已存在（加 --overwrite 强制覆盖）")
            skipped += 1
            continue

        h5_path = find_traj_h5(traj_dir)
        if h5_path is None:
            print(f"  [WARN]  {traj_dir.name}: 找不到 .h5 文件，跳过")
            errors += 1
            continue

        try:
            actors, links = read_actors_links(h5_path)
        except Exception as e:
            print(f"  [WARN]  {traj_dir.name}: 读 h5 失败 ({e})，跳过")
            errors += 1
            continue

        payload = {
            "task_id":   task_id,
            "traj_name": traj_dir.name,
            "actors":    actors,
            "links":     links,
        }

        if apply:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"  [OK]    {traj_dir.name}/traj_task.json  "
                  f"({len(actors)} actors, {len(links)} links)")
        else:
            print(f"  [DRY]   {traj_dir.name}/traj_task.json  "
                  f"task_id={task_id!r}  actors={[a['name'] for a in actors]}")
        written += 1

    print()
    if apply:
        print(f"完成：写入 {written} 个，跳过 {skipped} 个，失败 {errors} 个。")
    else:
        print(f"预览完成：将写入 {written} 个，跳过 {skipped} 个，失败 {errors} 个。加 --apply 后实际写入。")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="为每条轨迹目录写入 traj_task.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="任务数据集根目录，需包含 camera_data/ 子目录",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="实际写入文件（默认仅 dry-run 预览）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="覆盖已存在的 traj_task.json（默认跳过）",
    )
    args = parser.parse_args()

    try:
        run(args.dataset_dir, apply=args.apply, overwrite=args.overwrite)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
