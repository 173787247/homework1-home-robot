# 作业1：家居机器人 - Grounding, Counting 和 VQA

## 项目概述

本项目实现了一个可以识别常见家居环境的家居机器人，具备以下功能：
- **Grounding**: 对图像中的物体进行定位和识别
- **Counting**: 统计图像中物体的数量
- **VQA (Visual Question Answering)**: 回答关于图像的问题

## 数据集

使用 NYU Depth V2 数据集：
- 数据集链接：https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- 包含室内场景的RGB图像和深度信息

## 模型架构

采用以下模型组合：
- **MM Grounding DINO**: 用于物体检测和定位（Grounding）
- **BLIP-2**: 用于视觉问答（VQA）和图像理解
- 或者使用 **LLaVA-NeXT** 作为替代方案

## 项目结构

```
homework1-home-robot/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖包
├── config.yaml              # 配置文件
├── download_data.py         # 数据下载脚本
├── train.py                 # 训练脚本
├── evaluate.py              # 评估脚本
├── inference.py             # 推理脚本
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── grounding_dino.py    # Grounding DINO 模型
│   │   ├── blip2.py             # BLIP-2 模型
│   │   └── llava.py             # LLaVA-NeXT 模型（可选）
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # 数据集加载
│   │   └── preprocessing.py     # 数据预处理
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── grounding.py         # Grounding 任务
│   │   ├── counting.py          # Counting 任务
│   │   └── vqa.py               # VQA 任务
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py    # 可视化工具
│       └── metrics.py          # 评估指标
└── notebooks/
    └── demo.ipynb            # 演示笔记本
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 下载数据

```bash
python download_data.py
```

### 2. 训练模型

```bash
python train.py --config config.yaml
```

### 3. 评估模型

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

### 4. 推理

```bash
python inference.py --image path/to/image.jpg --task grounding
python inference.py --image path/to/image.jpg --task counting
python inference.py --image path/to/image.jpg --task vqa --question "What is in this room?"
```

## 功能演示

### Grounding 示例
```python
from src.tasks.grounding import GroundingTask

task = GroundingTask()
result = task.ground("chair", image_path)
# 返回：检测到的椅子位置和置信度
```

### Counting 示例
```python
from src.tasks.counting import CountingTask

task = CountingTask()
count = task.count("chair", image_path)
# 返回：图像中椅子的数量
```

### VQA 示例
```python
from src.tasks.vqa import VQATask

task = VQATask()
answer = task.answer("What furniture is in this room?", image_path)
# 返回：问题的答案
```

## 实验结果

训练完成后，模型在以下任务上的表现：
- Grounding mAP: [待填充]
- Counting Accuracy: [待填充]
- VQA Accuracy: [待填充]

## 参考文献

- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
- BLIP-2: https://github.com/salesforce/BLIP
- LLaVA-NeXT: https://github.com/haotian-liu/LLaVA
- NYU Depth V2 Dataset: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html

