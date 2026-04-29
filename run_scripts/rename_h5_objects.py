#!/usr/bin/env python3
"""
rename_h5_objects.py  —  查看 / 批量修改 camera_data/traj_N/traj_N.h5 中的 object 名字

每个 traj_N.h5 里名字只存在两处：
  1. traj_N/id_poses.attrs[seg_id_str]      (seg_id → name 的顶层索引表)
  2. traj_N/id_poses/<seg_id>.attrs['name'] (每个 subgroup 的 name 字段)

两处同时改即可完成全部重命名。

用法示例
--------
# 1. 先查看当前所有 object 名字（只读 traj_0，速度快）
python run_scripts/rename_h5_objects.py --dataset-dir /path/to/ShellGameTouch-v0 --list

# 2a. 改单个 object（dry-run 预览）
python run_scripts/rename_h5_objects.py \\
    --dataset-dir /path/to/ShellGameTouch-v0 \\
    --rename "actor:025_mug-left-0" "actor:mug_left"

# 2b. 确认无误后写入
python run_scripts/rename_h5_objects.py \\
    --dataset-dir /path/to/ShellGameTouch-v0 \\
    --rename "actor:025_mug-left-0" "actor:mug_left" \\
    --apply

# 3a. 同时改多个 object（JSON 字符串或文件）
python run_scripts/rename_h5_objects.py \\
    --dataset-dir /path/to/ShellGameTouch-v0 \\
    --mapping '{"actor:025_mug-left-0": "actor:mug_left",
                "actor:025_mug-center-0": "actor:mug_center"}'

# 3b. 写入
python run_scripts/rename_h5_objects.py \\
    --dataset-dir /path/to/ShellGameTouch-v0 \\
    --mapping my_mapping.json \\
    --apply
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import h5py


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# 匹配 traj_N 以及 traj_N_p<pid>_<retry>_<hash> 两种目录
_TRAJ_RE      = re.compile(r"^traj_(\d+)$")
_TRAJ_ANY_RE  = re.compile(r"^traj_(\d+)(?:_.+)?$")


def get_traj_h5_files(dataset_dir: Path) -> list[Path]:
    """返回 camera_data/ 下所有 traj_N 及 traj_N_p... 目录中的 .h5 文件。"""
    camera_data = dataset_dir / "camera_data"
    if not camera_data.is_dir():
        raise FileNotFoundError(f"camera_data 目录不存在: {camera_data}")

    def _sort_key(p: Path) -> tuple[int, str]:
        m = _TRAJ_ANY_RE.match(p.name)
        return (int(m.group(1)), p.name) if m else (10**9, p.name)

    h5_files = []
    for traj_dir in sorted(camera_data.iterdir(), key=_sort_key):
        if not traj_dir.is_dir() or not _TRAJ_ANY_RE.match(traj_dir.name):
            continue
        # traj_N 目录里 h5 文件名 = traj_N.h5；
        # traj_N_p... 目录里 h5 文件名也可能是 traj_N.h5，直接找第一个 .h5
        h5_path = traj_dir / f"{traj_dir.name}.h5"
        if not h5_path.is_file():
            candidates = sorted(traj_dir.glob("*.h5"))
            h5_path = candidates[0] if candidates else None
        if h5_path and h5_path.is_file():
            h5_files.append(h5_path)
    return h5_files


def _find_id_poses(f: h5py.File) -> Optional[h5py.Group]:
    """在 h5 文件中定位 id_poses group（支持有无 traj_N 顶层包裹）。"""
    for key in f.keys():
        if _TRAJ_RE.match(key) and "id_poses" in f[key]:
            return f[key]["id_poses"]  # type: ignore[return-value]
    if "id_poses" in f:
        return f["id_poses"]  # type: ignore[return-value]
    return None


# ---------------------------------------------------------------------------
# list mode
# ---------------------------------------------------------------------------

def cmd_list(dataset_dir: Path) -> None:
    """扫描 traj_0.h5，打印所有 object 名字（seg_id、类型、名称）。"""
    h5_files = get_traj_h5_files(dataset_dir)
    if not h5_files:
        print("[ERROR] 没有找到任何 traj_N.h5 文件。")
        sys.exit(1)

    sample = h5_files[0]
    print(f"数据集: {dataset_dir}")
    print(f"共 {len(h5_files)} 条轨迹，以下名字来自 {sample.parent.name}/{sample.name}\n")

    with h5py.File(sample, "r") as f:
        id_poses = _find_id_poses(f)
        if id_poses is None:
            print("[ERROR] 找不到 id_poses group。")
            sys.exit(1)

        # 从顶层 attrs 读取 seg_id → name，按 seg_id 排序
        entries = sorted(
            ((int(k), str(v)) for k, v in id_poses.attrs.items()),
            key=lambda x: x[0],
        )

    # 分组打印
    actors = [(sid, name) for sid, name in entries if name.startswith("actor:")]
    links  = [(sid, name) for sid, name in entries if name.startswith("link:")]

    col_w = max((len(name) for _, name in entries), default=20) + 2

    def _print_group(title: str, rows: list[tuple[int, str]]) -> None:
        if not rows:
            return
        print(f"  {'seg_id':>6}  {'name'}")
        print(f"  {'------':>6}  {'-' * col_w}")
        for sid, name in rows:
            print(f"  {sid:>6}  {name}")
        print()

    print(f"[ actors — 可改名的 object，共 {len(actors)} 个 ]")
    _print_group("actors", actors)

    print(f"[ links  — 机器人关节，通常不需要改，共 {len(links)} 个 ]")
    _print_group("links", links)

    print("提示：用 --rename OLD NEW 改单个 object，或 --mapping JSON 批量改。")


# ---------------------------------------------------------------------------
# rename mode
# ---------------------------------------------------------------------------

Change = tuple[str, str, str]  # (location_desc, old_name, new_name)


def rename_in_h5(h5_path: Path, mapping: dict[str, str], apply: bool) -> list[Change]:
    """
    检查（并可选地写入）h5 文件中的名字改动。
    返回 [(location_desc, old_name, new_name), ...]。
    """
    mode = "r+" if apply else "r"
    changes: list[Change] = []

    with h5py.File(h5_path, mode) as f:
        id_poses = _find_id_poses(f)
        if id_poses is None:
            print(f"  [WARN] {h5_path}: 找不到 id_poses group，跳过")
            return changes

        # --- 位置 1：id_poses 顶层 attrs  (seg_id_str → name_string) ---
        for seg_id_str, old_name in list(id_poses.attrs.items()):
            old_name = str(old_name)
            if old_name in mapping:
                new_name = mapping[old_name]
                changes.append((f"id_poses.attrs['{seg_id_str}']", old_name, new_name))
                if apply:
                    id_poses.attrs[seg_id_str] = new_name

        # --- 位置 2：id_poses/<seg_id>.attrs['name'] ---
        for seg_id_key in id_poses.keys():
            subgroup = id_poses[seg_id_key]
            if not isinstance(subgroup, h5py.Group):
                continue
            if "name" not in subgroup.attrs:
                continue
            old_name = str(subgroup.attrs["name"])
            if old_name in mapping:
                new_name = mapping[old_name]
                changes.append(
                    (f"id_poses/{seg_id_key}.attrs['name']", old_name, new_name)
                )
                if apply:
                    subgroup.attrs["name"] = new_name

    return changes


def cmd_rename(dataset_dir: Path, mapping: dict[str, str], apply: bool) -> None:
    """打印 dry-run 或执行写入。"""
    mode_label = "[APPLY]" if apply else "[DRY-RUN]"
    print(f"{mode_label}  dataset-dir: {dataset_dir}")
    print(f"  改名映射 ({len(mapping)} 条):")
    for old, new in mapping.items():
        print(f"    {old!r}  →  {new!r}")
    print()

    h5_files = get_traj_h5_files(dataset_dir)
    print(f"共发现 {len(h5_files)} 个 traj_N.h5 文件\n")

    total_changes = 0
    for h5_path in h5_files:
        changes = rename_in_h5(h5_path, mapping, apply=apply)
        if changes:
            print(f"  {h5_path.parent.name}/{h5_path.name}:")
            for loc, old, new in changes:
                print(f"    [{loc}]  {old!r}  →  {new!r}")
            total_changes += len(changes)

    print()
    if total_changes == 0:
        print("没有任何名字匹配 mapping，无改动。")
    else:
        verb = "已写入" if apply else "将改动（未写入，加 --apply 后生效）"
        print(f"合计 {verb} {total_changes} 处。")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="查看 / 批量修改 camera_data/traj_N/traj_N.h5 中的 object 名字",
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

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--list",
        action="store_true",
        help="列出当前数据集中所有 object 名字（只读，不修改任何文件）",
    )
    action.add_argument(
        "--rename",
        nargs=2,
        metavar=("OLD_NAME", "NEW_NAME"),
        help="改单个 object 的名字，例如: --rename actor:025_mug-left-0 actor:mug_left",
    )
    action.add_argument(
        "--mapping",
        metavar="JSON_OR_FILE",
        help=(
            "同时改多个 object：inline JSON 字符串 '{\"old\": \"new\", ...}' "
            "或指向 JSON 文件的路径"
        ),
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="实际写入改动（默认仅 dry-run 预览，不修改任何文件）",
    )
    args = parser.parse_args()

    # --apply 只对 rename/mapping 有意义
    if args.list and args.apply:
        parser.error("--apply 与 --list 不能同时使用")

    try:
        if args.list:
            cmd_list(args.dataset_dir)
            return

        # 构建 mapping
        if args.rename:
            old_name, new_name = args.rename
            mapping: dict[str, str] = {old_name: new_name}
        else:
            raw = args.mapping.strip()
            if raw.startswith("{"):
                try:
                    mapping = json.loads(raw)
                except json.JSONDecodeError as e:
                    parser.error(f"--mapping JSON 解析失败: {e}")
            else:
                mapping_path = Path(raw)
                if not mapping_path.is_file():
                    parser.error(f"mapping 文件不存在: {mapping_path}")
                with open(mapping_path) as fp:
                    mapping = json.load(fp)

        if not mapping:
            parser.error("mapping 为空，退出。")

        cmd_rename(args.dataset_dir, mapping, apply=args.apply)

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
