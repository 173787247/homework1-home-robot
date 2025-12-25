# 作业1 Docker 测试指南

## 环境要求

- ✅ Docker Desktop（已安装并运行）
- ✅ NVIDIA RTX 5080 GPU
- ✅ NVIDIA Container Toolkit（Docker Desktop 已包含）

## 数据集下载

### NYU Depth V2 数据集

数据集官方链接：https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html

**Labeled Dataset（约2.8GB）**：
- 包含 1449 对对齐的 RGB 和深度图像
- 密集的多类别标签
- 下载链接：http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat

### 下载方式

#### 方式1：使用脚本下载（推荐）

```bash
cd homework1-home-robot
python download_data.py
```

#### 方式2：手动下载

1. 访问：https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
2. 下载 "Labeled dataset (~2.8 GB)"
3. 将文件保存到：`./data/nyu_depth_v2/nyu_depth_v2_labeled.mat`

## Docker 测试步骤

### 1. 验证 GPU 支持

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

如果能看到 GPU 信息，说明 Docker GPU 支持正常。

### 2. 构建 Docker 镜像

```bash
cd homework1-home-robot
docker build -t homework1-home-robot:gpu .
```

### 3. 使用 Docker Compose 运行（推荐）

```bash
# 启动容器
docker-compose -f docker-compose.gpu.yml up -d

# 查看日志
docker-compose -f docker-compose.gpu.yml logs -f

# 进入容器
docker exec -it homework1-home-robot-gpu bash
```

### 4. 在容器内测试

#### 测试1：检查 GPU 和依赖

```bash
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

#### 测试2：测试模块导入

```bash
python test_imports.py
```

#### 测试3：测试 Grounding 功能

```bash
# 使用 NYU Depth V2 数据集中的图像
python inference.py \
  --image /app/data/nyu_depth_v2/sample_image.jpg \
  --task grounding \
  --text "chair . table . lamp" \
  --output /app/output/grounding_result.jpg \
  --device cuda
```

#### 测试4：测试 Counting 功能

```bash
python inference.py \
  --image /app/data/nyu_depth_v2/sample_image.jpg \
  --task counting \
  --text "chair" \
  --device cuda
```

#### 测试5：测试 VQA 功能

```bash
python inference.py \
  --image /app/data/nyu_depth_v2/sample_image.jpg \
  --task vqa \
  --text "What furniture is in this room?" \
  --device cuda
```

### 5. 从 NYU Depth V2 数据集提取测试图像

如果数据集已下载，可以创建一个脚本来提取图像：

```python
# extract_test_images.py
import scipy.io
import numpy as np
from PIL import Image
from pathlib import Path

# 加载数据集
mat_file = Path("data/nyu_depth_v2/nyu_depth_v2_labeled.mat")
data = scipy.io.loadmat(str(mat_file))

# 提取前5张图像作为测试
images = data['images']  # Shape: (1449, 3, 480, 640)
output_dir = Path("data/nyu_depth_v2/test_images")
output_dir.mkdir(parents=True, exist_ok=True)

for i in range(min(5, len(images))):
    # 转换为 (H, W, C) 格式
    img = np.transpose(images[i], (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    pil_img.save(output_dir / f"test_image_{i:04d}.jpg")
    print(f"已保存: test_image_{i:04d}.jpg")

print(f"\n测试图像已保存到: {output_dir}")
```

运行提取脚本：

```bash
python extract_test_images.py
```

## 完整测试流程

### 步骤1：准备数据

```bash
# 在容器内或本地
cd homework1-home-robot
python download_data.py  # 下载数据集
python extract_test_images.py  # 提取测试图像
```

### 步骤2：运行完整测试

```bash
# 在容器内
cd /app

# 测试所有功能
for task in grounding counting vqa; do
  echo "测试 $task..."
  python inference.py \
    --image /app/data/nyu_depth_v2/test_images/test_image_0000.jpg \
    --task $task \
    --text "chair" \
    --output /app/output/${task}_result.jpg \
    --device cuda
done
```

### 步骤3：查看结果

```bash
# 结果保存在 /app/output/ 目录
ls -lh /app/output/
```

## 使用现有 Docker 镜像

如果您已经有相关的 Docker 镜像，可以直接使用：

```bash
# 例如使用 PyTorch 官方镜像
docker run --gpus all -it \
  -v $(pwd):/app \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel \
  bash
```

## 故障排除

### 问题1：GPU 不可用

检查 Docker Desktop 设置：
- Settings → Resources → Advanced
- 确保 "Use the WSL 2 based engine" 已启用
- 确保 GPU 支持已启用

### 问题2：数据集下载失败

手动下载数据集：
1. 访问：https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
2. 下载 `nyu_depth_v2_labeled.mat`
3. 放置到 `./data/nyu_depth_v2/` 目录

### 问题3：模型下载慢

可以在 Dockerfile 中配置镜像源，或使用本地模型缓存。

## 参考

- NYU Depth V2 数据集：https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- Grounding DINO：https://github.com/IDEA-Research/GroundingDINO
- BLIP-2：https://github.com/salesforce/BLIP

