#!/usr/bin/env python3
"""从仓库根目录统一调用荔枝果实或荔枝花检测功能。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PACKAGE_DIRS = {
    "fruit": "lychee_fruit_detector_delivery",
    "flower": "lychee_flower_detector_delivery",
}

ACTION_SCRIPTS = {
    "single": "infer_single.py",
    "folder": "infer_folder.py",
    "evaluate": "evaluate_model.py",
}


def parse_args() -> argparse.Namespace:
    """解析识别目标、运行功能及需要转交给子程序的参数。"""
    parser = argparse.ArgumentParser(
        description="荔枝视觉检测统一入口",
        epilog=(
            "示例：python run.py flower folder --input-dir ./photos --device cpu\n"
            "      python run.py fruit evaluate --input-dir ./dataset --device cpu"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        choices=tuple(PACKAGE_DIRS),
        help="fruit表示荔枝果实，flower表示荔枝大花簇",
    )
    parser.add_argument(
        "action",
        choices=tuple(ACTION_SCRIPTS),
        help="single单图，folder批量图片，evaluate带标签验证集评估",
    )
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="传递给具体检测程序的其余参数",
    )
    return parser.parse_args()


def resolve_script(root_dir: Path, target: str, action: str) -> Path:
    """根据目标和功能定位仓库内对应的Python入口文件。"""
    script_path = root_dir / PACKAGE_DIRS[target] / ACTION_SCRIPTS[action]
    if not script_path.is_file():
        raise FileNotFoundError(f"运行入口不存在：{script_path}")
    return script_path


def main() -> int:
    """使用当前Python解释器启动选定的检测或评估程序。"""
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    try:
        script_path = resolve_script(root_dir, args.target, args.action)
        completed = subprocess.run(
            [sys.executable, str(script_path), *args.arguments],
            cwd=str(script_path.parent),
            check=False,
        )
    except Exception as error:
        print(f"启动失败：{error}", file=sys.stderr)
        return 1
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
