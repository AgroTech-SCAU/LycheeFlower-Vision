#!/usr/bin/env python3
"""为荔枝果实 YOLO 模型提供只加载一次权重的文件夹批量检测入口。"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from infer_single import (
    COUNT_TARGET,
    MODEL_ID,
    MODEL_VERSION,
    SCHEMA_VERSION,
    calculate_sha256,
    classify_error,
    copy_original_image,
    save_overlay,
    serialize_detections,
    write_json,
)


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    """解析输入文件夹、权重、输出目录和批量推理参数。"""
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="荔枝果实 YOLO 文件夹批量检测")
    parser.add_argument(
        "--input-dir",
        default=str(package_dir / "input_images"),
        help="待检测图片文件夹",
    )
    parser.add_argument(
        "--weights",
        default=str(package_dir / "models" / "best1.pt"),
        help="YOLO 权重路径",
    )
    parser.add_argument(
        "--output-dir",
        default=str(package_dir / "output"),
        help="批量结果输出目录",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=1696, help="推理输入尺寸")
    parser.add_argument("--device", default=None, help="推理设备，例如 0、cpu 或 cuda:0")
    parser.add_argument("--recursive", action="store_true", help="递归读取所有子文件夹")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """校验批量推理路径和数值参数并返回规范化路径。"""
    input_dir = Path(args.input_dir).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入文件夹不存在：{input_dir}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"权重不存在：{weights_path}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("conf 必须位于 0 到 1 之间")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("iou 必须位于 0 到 1 之间")
    if args.imgsz <= 0:
        raise ValueError("imgsz 必须大于 0")
    return input_dir, weights_path, output_dir


def discover_images(input_dir: Path, recursive: bool) -> list[Path]:
    """按文件名顺序查找文件夹中的受支持图片。"""
    candidates = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: str(path.relative_to(input_dir)).casefold(),
    )


def allocate_result_dir(
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    allocated: set[Path],
) -> Path:
    """为每张输入图片分配不会与其他图片冲突的结果目录。"""
    relative_path = image_path.relative_to(input_dir)
    candidate = output_dir / relative_path.parent / relative_path.stem
    if candidate in allocated:
        candidate = candidate.with_name(
            f"{relative_path.stem}__{relative_path.suffix.lstrip('.').lower()}"
        )
    allocated.add(candidate)
    return candidate


def predict_one(
    model: YOLO,
    image_path: Path,
    result_dir: Path,
    weights_path: Path,
    weights_sha256: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """使用已加载的模型检测一张图片并保存全部结果。"""
    result_dir.mkdir(parents=True, exist_ok=True)
    original_path = copy_original_image(image_path, result_dir)
    prediction_args: dict[str, Any] = {
        "source": str(image_path),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        prediction_args["device"] = args.device

    results = model.predict(**prediction_args)
    if len(results) != 1:
        raise RuntimeError(f"单图推理返回了 {len(results)} 个结果")

    result = results[0]
    detections = serialize_detections(result)
    overlay_path = result_dir / "overlay.jpg"
    save_overlay(result.plot(), overlay_path)
    inference_time_ms = round(float(result.speed.get("inference", 0.0)), 3)
    height, width = (int(value) for value in result.orig_shape)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "success": True,
        "error_code": None,
        "task": "object_detection",
        "count_target": COUNT_TARGET,
        "model": {
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "weights_file": weights_path.name,
            "weights_sha256": weights_sha256,
        },
        "input": {
            "source_file": image_path.name,
            "archived_file": original_path.name,
            "width": width,
            "height": height,
        },
        "parameters": {
            "confidence_threshold": args.conf,
            "nms_iou_threshold": args.iou,
            "image_size": args.imgsz,
            "device": args.device or "auto",
        },
        "raw_count": len(detections),
        "detections": detections,
        "inference_time_ms": inference_time_ms,
        "warnings": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "original": original_path.name,
            "overlay": overlay_path.name,
            "result": "result.json",
        },
    }
    write_json(payload, result_dir / "result.json")
    return payload


def save_image_failure(result_dir: Path, image_path: Path, error: Exception) -> None:
    """在单张图片失败时保存结构化错误结果并允许批次继续。"""
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "success": False,
        "error_code": classify_error(error),
        "task": "object_detection",
        "count_target": COUNT_TARGET,
        "model": {"model_id": MODEL_ID, "model_version": MODEL_VERSION},
        "input": {"source_file": image_path.name},
        "raw_count": None,
        "detections": [],
        "warnings": [],
        "message": str(error),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(payload, result_dir / "result.json")


def write_summary_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    """将批量检测摘要写入便于表格查看的 CSV 文件。"""
    fieldnames = [
        "source_image",
        "success",
        "raw_count",
        "inference_time_ms",
        "error_code",
        "result_json",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    """加载一次模型并依次处理输入文件夹中的所有图片。"""
    input_dir, weights_path, output_dir = validate_args(args)
    images = discover_images(input_dir, args.recursive)
    if not images:
        raise RuntimeError(f"输入文件夹中没有受支持的图片：{input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_sha256 = calculate_sha256(weights_path)
    model = YOLO(str(weights_path))
    allocated: set[Path] = set()
    records: list[dict[str, Any]] = []

    for index, image_path in enumerate(images, start=1):
        result_dir = allocate_result_dir(image_path, input_dir, output_dir, allocated)
        relative_image = image_path.relative_to(input_dir).as_posix()
        try:
            payload = predict_one(
                model,
                image_path,
                result_dir,
                weights_path,
                weights_sha256,
                args,
            )
            record = {
                "source_image": relative_image,
                "success": True,
                "raw_count": payload["raw_count"],
                "inference_time_ms": payload["inference_time_ms"],
                "error_code": "",
                "result_json": (result_dir / "result.json")
                .relative_to(output_dir)
                .as_posix(),
            }
            print(f"[{index}/{len(images)}] 成功：{relative_image}，数量={payload['raw_count']}")
        except Exception as error:
            save_image_failure(result_dir, image_path, error)
            record = {
                "source_image": relative_image,
                "success": False,
                "raw_count": "",
                "inference_time_ms": "",
                "error_code": classify_error(error),
                "result_json": (result_dir / "result.json")
                .relative_to(output_dir)
                .as_posix(),
            }
            print(f"[{index}/{len(images)}] 失败：{relative_image}，{error}", file=sys.stderr)
        records.append(record)

    success_count = sum(bool(record["success"]) for record in records)
    failure_count = len(records) - success_count
    total_detections = sum(
        int(record["raw_count"])
        for record in records
        if record["success"] and record["raw_count"] != ""
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task": "object_detection_batch",
        "count_target": COUNT_TARGET,
        "model": {
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "weights_file": weights_path.name,
            "weights_sha256": weights_sha256,
        },
        "input_dir": str(input_dir),
        "recursive": bool(args.recursive),
        "image_count": len(records),
        "success_count": success_count,
        "failure_count": failure_count,
        "sum_of_per_image_raw_count": total_detections,
        "count_warning": "多张图片可能包含重复果实，逐图数量之和不代表去重后的真实果实总数。",
        "records": records,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(summary, output_dir / "batch_summary.json")
    write_summary_csv(records, output_dir / "batch_summary.csv")
    return summary


def main() -> int:
    """执行文件夹批量检测并返回进程退出码。"""
    args = parse_args()
    try:
        summary = run_batch(args)
    except Exception as error:
        print(f"批量推理失败：{error}", file=sys.stderr)
        return 1

    print(
        "批量检测完成："
        f"总计 {summary['image_count']} 张，"
        f"成功 {summary['success_count']} 张，"
        f"失败 {summary['failure_count']} 张。"
    )
    print(f"汇总目录：{Path(args.output_dir).expanduser().resolve()}")
    return 0 if summary["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
