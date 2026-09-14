#!/usr/bin/env python3
"""为荔枝大花簇 YOLO 模型提供标准化的单图检测入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


MODEL_ID = "litchi_flower_cluster_detect"
MODEL_VERSION = "v1.0"
COUNT_TARGET = "large_flower_cluster"
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    """解析权重、图像、输出目录和推理参数。"""
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="荔枝大花簇 YOLO 单图目标检测")
    parser.add_argument("--image", required=True, help="待检测图片路径")
    parser.add_argument(
        "--weights",
        default=str(package_dir / "models" / "best.pt"),
        help="YOLO 权重路径",
    )
    parser.add_argument("--output-dir", default="output", help="结果输出目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    parser.add_argument("--device", default=None, help="推理设备，例如 0、cpu 或 cuda:0")
    return parser.parse_args()


def calculate_sha256(file_path: Path) -> str:
    """计算指定文件的 SHA256 校验值。"""
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def copy_original_image(image_path: Path, output_dir: Path) -> Path:
    """以不改变文件内容的方式复制原始输入图片。"""
    suffix = image_path.suffix.lower() or ".jpg"
    target_path = output_dir / f"original{suffix}"
    shutil.copy2(image_path, target_path)
    return target_path


def save_overlay(image: Any, output_path: Path) -> None:
    """以兼容中文路径的方式保存检测框可视化图片。"""
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("无法编码可视化图片")
    encoded.tofile(str(output_path))


def serialize_detections(result: Any) -> list[dict[str, Any]]:
    """将 YOLO 检测框转换为稳定的 JSON 记录。"""
    detections: list[dict[str, Any]] = []
    names = result.names
    if result.boxes is None:
        return detections

    for index, box in enumerate(result.boxes, start=1):
        class_id = int(box.cls.item())
        coordinates = [round(float(value), 2) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "detection_id": f"D{index:04d}",
                "class_id": class_id,
                "class_name": str(names[class_id]),
                "confidence": round(float(box.conf.item()), 6),
                "bbox_xyxy": coordinates,
            }
        )
    return detections


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    """以 UTF-8 编码写入格式化 JSON 结果。"""
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    """加载模型并完成单张图片的目标检测与结果归档。"""
    image_path = Path(args.image).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在：{image_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"权重不存在：{weights_path}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("conf 必须位于 0 到 1 之间")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("iou 必须位于 0 到 1 之间")

    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = copy_original_image(image_path, output_dir)
    model = YOLO(str(weights_path))
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
    overlay_path = output_dir / "overlay.jpg"
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
            "weights_sha256": calculate_sha256(weights_path),
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
    write_json(payload, output_dir / "result.json")
    return payload


def classify_error(error: Exception) -> str:
    """根据异常类型生成稳定的机器错误码。"""
    if isinstance(error, FileNotFoundError):
        message = str(error)
        return "INVALID_IMAGE" if "图片" in message else "MODEL_UNAVAILABLE"
    if isinstance(error, ValueError):
        return "INVALID_ARGUMENT"
    return "INFERENCE_FAILED"


def write_failure_result(args: argparse.Namespace, error: Exception) -> None:
    """在推理失败时尽可能保存结构化错误结果。"""
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "success": False,
        "error_code": classify_error(error),
        "task": "object_detection",
        "count_target": COUNT_TARGET,
        "model": {"model_id": MODEL_ID, "model_version": MODEL_VERSION},
        "raw_count": None,
        "detections": [],
        "warnings": [],
        "message": str(error),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(payload, output_dir / "result.json")


def main() -> int:
    """执行命令行推理流程并返回进程退出码。"""
    args = parse_args()
    try:
        payload = run_inference(args)
    except Exception as error:
        write_failure_result(args, error)
        print(f"推理失败：{error}", file=sys.stderr)
        return 1

    print(f"检测完成：发现 {payload['raw_count']} 个荔枝大花簇目标")
    print(f"结果目录：{Path(args.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
