#!/usr/bin/env python3
"""使用带真实标注的荔枝大花簇验证集评估检测模型并输出验收指标。"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from ultralytics import YOLO

from infer_single import (
    COUNT_TARGET,
    MODEL_ID,
    MODEL_VERSION,
    calculate_sha256,
    write_json,
)


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    """解析荔枝花验证集评估所需的命令行参数。"""
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="荔枝大花簇 YOLO 验证集评估")
    parser.add_argument(
        "--weights",
        default=str(package_dir / "models" / "best.pt"),
        help="YOLO 权重路径",
    )
    parser.add_argument(
        "--input-dir",
        default=str(package_dir / "input_images"),
        help="带标签验证集根目录，默认读取 input_images/val/images 和 labels",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="可选的 data.yaml 路径；不填写时自动根据 input-dir 生成",
    )
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument(
        "--output-dir",
        default=str(package_dir / "output" / "evaluation"),
        help="评估结果输出目录",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="评估输入尺寸")
    parser.add_argument("--conf", type=float, default=0.001, help="评估置信度下限")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU 阈值")
    parser.add_argument("--batch", type=int, default=1, help="评估批大小")
    parser.add_argument("--device", default=None, help="评估设备，例如 0、cpu 或 cuda:0")
    parser.add_argument(
        "--require-map50",
        type=float,
        default=0.80,
        help="mAP50 验收阈值",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path]:
    """校验评估路径和数值参数并返回规范化路径。"""
    weights_path = Path(args.weights).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve() if args.data else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"权重不存在：{weights_path}")
    if data_path is not None and not data_path.is_file():
        raise FileNotFoundError(f"数据集配置不存在：{data_path}")
    if data_path is None and not input_dir.is_dir():
        raise FileNotFoundError(f"验证集目录不存在：{input_dir}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("conf 必须位于 0 到 1 之间")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("iou 必须位于 0 到 1 之间")
    if not 0.0 <= args.require_map50 <= 1.0:
        raise ValueError("require-map50 必须位于 0 到 1 之间")
    if args.imgsz <= 0 or args.batch <= 0:
        raise ValueError("imgsz 和 batch 必须大于 0")
    return weights_path, input_dir, data_path, output_dir


def inspect_labeled_split(images_dir: Path, labels_dir: Path) -> dict[str, Any]:
    """检查验证图片与YOLO标签是否配对并返回数据集统计信息。"""
    image_files = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    label_files = sorted(labels_dir.rglob("*.txt"))
    image_stems = {path.stem for path in image_files}
    label_stems = {path.stem for path in label_files}
    missing_labels = sorted(image_stems - label_stems)
    extra_labels = sorted(label_stems - image_stems)
    if not image_files:
        raise ValueError(f"验证集没有找到受支持的图片：{images_dir}")
    if missing_labels:
        examples = "、".join(missing_labels[:5])
        raise ValueError(
            f"有 {len(missing_labels)} 张图片缺少同名标签文件，例如：{examples}；"
            "无目标图片也需要保留一个空的 .txt 标签文件"
        )
    return {
        "image_count": len(image_files),
        "label_count": len(label_files),
        "missing_label_count": len(missing_labels),
        "extra_label_count": len(extra_labels),
        "extra_label_examples": extra_labels[:20],
    }


def create_dataset_config(
    input_dir: Path,
    split: str,
    output_dir: Path,
    class_names: dict[int, str],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """自动识别带标签验证集目录并生成Ultralytics数据配置。"""
    candidates = [
        (input_dir / split / "images", input_dir / split / "labels"),
        (input_dir / "images", input_dir / "labels"),
    ]
    selected = next(
        (
            (images, labels)
            for images, labels in candidates
            if images.is_dir() and labels.is_dir()
        ),
        None,
    )
    if selected is None:
        raise FileNotFoundError(
            "没有找到带标签验证集。请使用 input_images\\val\\images 和 "
            "input_images\\val\\labels 目录结构。"
        )
    images_dir, labels_dir = selected
    dataset_check = inspect_labeled_split(images_dir, labels_dir)
    config: dict[str, Any] = {
        "path": str(images_dir.parent.resolve()),
        "train": "images",
        "val": "images",
        "names": {int(key): str(value) for key, value in class_names.items()},
    }
    if split == "test":
        config["test"] = "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_data.yaml"
    resolved_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return resolved_path, config, dataset_check


def resolve_dataset_config(
    data_path: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """规范化用户提供的数据集配置并保存本次实际配置。"""
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"data.yaml 内容不是有效映射：{data_path}")
    configured_root = Path(str(config.get("path", data_path.parent))).expanduser()
    if not configured_root.is_absolute():
        configured_root = (data_path.parent / configured_root).resolve()
    if not configured_root.exists():
        configured_root = data_path.parent
    config["path"] = str(configured_root.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_data.yaml"
    resolved_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return resolved_path, config


def to_float_list(values: Any) -> list[float]:
    """将NumPy或张量形式的指标序列转换为普通浮点数列表。"""
    if values is None:
        return []
    return [float(value) for value in values]


def build_metrics_payload(
    metrics: Any,
    model: YOLO,
    weights_path: Path,
    data_path: Path | None,
    resolved_data_path: Path,
    config: dict[str, Any],
    dataset_check: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """将Ultralytics验证指标整理为稳定的荔枝花验收报告。"""
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    class_precision = to_float_list(getattr(metrics.box, "p", None))
    class_recall = to_float_list(getattr(metrics.box, "r", None))
    class_map50 = to_float_list(getattr(metrics.box, "ap50", None))
    class_map50_95 = to_float_list(getattr(metrics.box, "maps", None))
    per_class: list[dict[str, Any]] = []
    for class_id, class_name in model.names.items():
        index = int(class_id)
        per_class.append(
            {
                "class_id": index,
                "class_name": str(class_name),
                "precision": class_precision[index] if index < len(class_precision) else None,
                "recall": class_recall[index] if index < len(class_recall) else None,
                "map50": class_map50[index] if index < len(class_map50) else None,
                "map50_95": class_map50_95[index] if index < len(class_map50_95) else None,
            }
        )
    return {
        "schema_version": 1,
        "success": True,
        "task": "object_detection_evaluation",
        "count_target": COUNT_TARGET,
        "model": {
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "weights_file": weights_path.name,
            "weights_sha256": calculate_sha256(weights_path),
        },
        "dataset": {
            "source_data_yaml": str(data_path) if data_path else None,
            "resolved_data_yaml": str(resolved_data_path),
            "split": args.split,
            "image_count": dataset_check.get("image_count"),
            "label_count": dataset_check.get("label_count"),
            "missing_label_count": dataset_check.get("missing_label_count", 0),
            "extra_label_count": dataset_check.get("extra_label_count", 0),
            "extra_label_examples": dataset_check.get("extra_label_examples", []),
            "class_names": config.get("names"),
        },
        "parameters": {
            "image_size": args.imgsz,
            "confidence_floor": args.conf,
            "nms_iou_threshold": args.iou,
            "matching_iou_threshold_for_map50": 0.50,
            "batch": args.batch,
            "device": args.device or "auto",
        },
        "iou_definition": {
            "formula": "IoU = area(prediction ∩ ground truth) / area(prediction ∪ ground truth)",
            "description": "类别正确且预测框与真实框的IoU不低于0.5时参与mAP50的正确匹配判定。",
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
            "per_class": per_class,
        },
        "acceptance": {
            "required_map50": args.require_map50,
            "passed": map50 >= args.require_map50,
            "difference": map50 - args.require_map50,
        },
        "speed_ms_per_image": {
            key: round(float(value), 3) for key, value in metrics.speed.items()
        },
        "plot_directory": str(Path(metrics.save_dir).resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_metrics_files(payload: dict[str, Any], output_dir: Path) -> None:
    """将评估结果写成JSON、CSV、文本和独立mAP50文件。"""
    write_json(payload, output_dir / "metrics.json")
    metrics = payload["metrics"]
    acceptance = payload["acceptance"]
    status = "达标" if acceptance["passed"] else "未达标"
    text_lines = [
        "荔枝大花簇目标检测模型评估结果",
        f"数据划分：{payload['dataset']['split']}",
        f"图片数量：{payload['dataset']['image_count']}",
        f"标签文件数量：{payload['dataset']['label_count']}",
        f"缺少标签：{payload['dataset']['missing_label_count']}",
        f"多余标签：{payload['dataset']['extra_label_count']}",
        f"Precision：{metrics['precision']:.6f}",
        f"Recall：{metrics['recall']:.6f}",
        f"mAP50：{metrics['map50']:.6f}",
        f"mAP50-95：{metrics['map50_95']:.6f}",
        f"验收要求：mAP50 ≥ {acceptance['required_map50']:.2%}",
        f"验收结论：{status}",
        "IoU公式：IoU = 预测框与真实框交集面积 / 预测框与真实框并集面积",
    ]
    (output_dir / "metrics.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    map_text = (
        f"mAP50={metrics['map50']:.6f}\n"
        f"required_mAP50={acceptance['required_map50']:.6f}\n"
        f"passed={'YES' if acceptance['passed'] else 'NO'}\n"
    )
    (output_dir / "map.txt").write_text(map_text, encoding="utf-8")
    (output_dir / "mpa.txt").write_text(map_text, encoding="utf-8")
    fieldnames = [
        "scope",
        "class_name",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "required_map50",
        "passed",
    ]
    rows = [
        {
            "scope": "overall",
            "class_name": "all",
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "map50": metrics["map50"],
            "map50_95": metrics["map50_95"],
            "required_map50": acceptance["required_map50"],
            "passed": acceptance["passed"],
        }
    ]
    for class_metrics in metrics["per_class"]:
        rows.append(
            {
                "scope": "class",
                "class_name": class_metrics["class_name"],
                "precision": class_metrics["precision"],
                "recall": class_metrics["recall"],
                "map50": class_metrics["map50"],
                "map50_95": class_metrics["map50_95"],
                "required_map50": acceptance["required_map50"],
                "passed": (
                    class_metrics["map50"] >= acceptance["required_map50"]
                    if class_metrics["map50"] is not None
                    else False
                ),
            }
        )
    with (output_dir / "metrics.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def convert_prediction_images_to_png(result_dir: Path) -> list[str]:
    """将验证过程生成的预测结果图转换为规范要求的PNG格式。"""
    png_names: list[str] = []
    for source_path in sorted(result_dir.glob("*_pred.jpg")):
        target_path = source_path.with_suffix(".png")
        with Image.open(source_path) as image:
            image.save(target_path, format="PNG")
        png_names.append(target_path.name)
    return png_names


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """运行完整验证集评估并生成验收报告和可视化结果。"""
    weights_path, input_dir, data_path, output_dir = validate_args(args)
    model = YOLO(str(weights_path))
    if data_path is None:
        resolved_data_path, config, dataset_check = create_dataset_config(
            input_dir,
            args.split,
            output_dir,
            model.names,
        )
    else:
        resolved_data_path, config = resolve_dataset_config(data_path, output_dir)
        dataset_check = {
            "image_count": None,
            "label_count": None,
            "missing_label_count": 0,
            "extra_label_count": 0,
            "extra_label_examples": [],
        }
    validation_args: dict[str, Any] = {
        "data": str(resolved_data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "batch": args.batch,
        "plots": True,
        "project": str(output_dir),
        "name": "test_result",
        "exist_ok": True,
        "verbose": False,
    }
    if args.device:
        validation_args["device"] = args.device
    metrics = model.val(**validation_args)
    prediction_png_files = convert_prediction_images_to_png(Path(metrics.save_dir))
    payload = build_metrics_payload(
        metrics,
        model,
        weights_path,
        data_path,
        resolved_data_path,
        config,
        dataset_check,
        args,
    )
    payload["prediction_png_files"] = prediction_png_files
    write_metrics_files(payload, output_dir)
    return payload


def main() -> int:
    """执行荔枝花模型评估并在终端显示mAP50验收结论。"""
    args = parse_args()
    try:
        payload = run_evaluation(args)
    except Exception as error:
        print(f"模型评估失败：{error}", file=sys.stderr)
        return 1
    metrics = payload["metrics"]
    acceptance = payload["acceptance"]
    print(f"评估完成：mAP50={metrics['map50']:.4f}")
    print(
        f"验收要求：mAP50≥{acceptance['required_map50']:.2%}，"
        f"结果={'达标' if acceptance['passed'] else '未达标'}"
    )
    print(f"评估目录：{Path(args.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
