# 荔枝大花簇检测模块

本目录是荔枝大花簇YOLO目标检测子模块，默认类别为 `lizhihua`、模型为 `models/best.pt`、输入尺寸为640。

## 文件作用

```text
├─ models/best.pt        # 模型权重
├─ input_images/val/     # 可选的本地验证集位置
├─ output/               # 默认输出位置
├─ infer_single.py       # 单图检测
├─ infer_folder.py       # 文件夹批量检测
├─ evaluate_model.py     # 带标签验证集评估
├─ model_info.json       # 模型信息
└─ requirements.txt      # 子模块依赖
```

## 直接运行

```bash
python infer_single.py --image /path/to/flower.jpg --output-dir ./output/single
python infer_folder.py --input-dir /path/to/images --output-dir ./output/batch
python evaluate_model.py --input-dir /path/to/dataset --output-dir ./output/evaluation --device cpu
```

验证集目录需要包含同级的 `images` 和 `labels` 文件夹。评估会输出Precision、Recall、mAP50、mAP50-95、混淆矩阵、曲线和PNG检测结果。

也可在仓库根目录运行：

```bash
python run.py flower folder --input-dir /path/to/images
python run.py flower evaluate --input-dir /path/to/dataset --device cpu
```

所有默认路径均相对于本模块自动定位，仓库移动或克隆后无需修改Python源码。
