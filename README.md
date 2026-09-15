# Lychee Vision Capability

可独立克隆和运行的荔枝视觉目标检测模块，包含两套互相独立的YOLO模型：

| 目标参数 | 识别对象 | 模型 | 默认输入尺寸 |
|---|---|---|---:|
| `fruit` | 荔枝果实 | `lychee_fruit_detector_delivery/models/best1.pt` | 1696 |
| `flower` | 荔枝大花簇 | `lychee_flower_detector_delivery/models/best.pt` | 640 |

两套模型执行目标检测，不执行实例分割。仓库内所有运行路径均根据代码所在位置自动确定，不依赖某一台电脑的绝对路径。

## 1. 克隆和安装

```bash
git clone https://github.com/AgroTech-SCAU/LycheeFlower-Vision.git
cd lizhihua
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Ubuntu或其他Linux：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

如果设备需要CUDA加速，请先按照设备CUDA版本安装对应的PyTorch，再安装本仓库依赖。

## 2. 统一运行入口

所有命令都可以从仓库根目录通过 `run.py` 调用。

### 单张图片识别

```bash
python run.py fruit single --image ./photos/lychee.jpg --output-dir ./result/fruit_single
python run.py flower single --image ./photos/flower.jpg --output-dir ./result/flower_single
```

### 文件夹批量识别

```bash
python run.py fruit folder --input-dir ./photos/fruit --output-dir ./result/fruit
python run.py flower folder --input-dir ./photos/flower --output-dir ./result/flower
```

需要递归读取子目录时，在命令末尾添加 `--recursive`。CPU运行可添加 `--device cpu`，NVIDIA显卡通常使用 `--device 0`。

### 带标签验证集评估

验证集使用YOLO目标检测格式：

```text
dataset/
├─ images/
│  ├─ image_001.jpg
│  └─ ...
└─ labels/
   ├─ image_001.txt
   └─ ...
```

运行：

```bash
python run.py fruit evaluate --input-dir ./dataset/fruit --output-dir ./result/fruit_evaluation --device cpu
python run.py flower evaluate --input-dir ./dataset/flower --output-dir ./result/flower_evaluation --device cpu
```

评估输出包括：

- `metrics.txt`：中文指标及mAP50验收结论。
- `metrics.csv`：可用WPS表格或Excel打开。
- `metrics.json`：结构化指标和运行参数。
- `map.txt`、`mpa.txt`：mAP50及是否达到80%。
- `test_result`：混淆矩阵、PR/P/R/F1曲线和PNG预测结果图。

## 3. 直接使用子模块

如不使用统一入口，也可以进入对应目录直接运行：

```bash
cd lychee_flower_detector_delivery
python evaluate_model.py --device cpu
```

两个子目录各自的 `README.md` 包含更具体的参数和输出说明。

## 4. GitHub上传说明

- `models/*.pt` 是运行必需文件，应随仓库上传；当前单个模型均未超过GitHub的100MB单文件限制。
- 验证图片、标签、历史输出、缓存和ZIP文件已由 `.gitignore` 排除，不会进入Git提交。
- 克隆后的用户自行把图片或验证集放到任意目录，再通过 `--input-dir` 或 `--image` 指定。
- 正式发布前请根据项目要求补充合适的开源许可证。

## 5. 输出含义

- Precision：预测为目标的框中有多少是正确的。
- Recall：真实目标中有多少被模型识别出来。
- IoU：预测框与真实框交集面积除以并集面积。
- mAP50：以IoU=0.5作为匹配标准计算的平均精度。

单图检测数量不能直接跨多张图片相加作为去重后的真实数量。
