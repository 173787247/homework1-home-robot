"""
从 NYU Depth V2 数据集中提取测试图像
支持 MATLAB v7.3 格式（使用 h5py）
"""
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import sys
import io

# 设置输出编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    import h5py
    USE_H5PY = True
except ImportError:
    try:
        import scipy.io
        USE_H5PY = False
    except ImportError:
        raise ImportError("需要安装 h5py 或 scipy 来读取 .mat 文件")


def load_mat_file(mat_file_path):
    """
    加载 .mat 文件（支持 v7.3 格式）
    
    Returns:
        dict: 包含 images, depths, labels 等数据
    """
    mat_file = Path(mat_file_path)
    if not mat_file.exists():
        raise FileNotFoundError(f"数据集文件不存在: {mat_file}")
    
    if USE_H5PY:
        # 使用 h5py 读取 MATLAB v7.3 格式
        print(f"使用 h5py 加载 MATLAB v7.3 文件: {mat_file}")
        with h5py.File(str(mat_file), 'r') as f:
            # MATLAB v7.3 格式中，数据以引用形式存储
            images_ref = f['images']
            # 获取实际数据
            images = np.array(images_ref)
            # 转置：MATLAB 存储为 (3, 480, 640, N)，需要转为 (N, 480, 640, 3)
            if images.ndim == 4 and images.shape[0] == 3:
                images = np.transpose(images, (3, 1, 2, 0))
            return {'images': images}
    else:
        # 使用 scipy.io 读取旧格式
        print(f"使用 scipy.io 加载 .mat 文件: {mat_file}")
        data = scipy.io.loadmat(str(mat_file))
        images = data['images']  # Shape: (1449, 3, 480, 640)
        # 转换为 (N, H, W, C) 格式
        if images.ndim == 4 and images.shape[1] == 3:
            images = np.transpose(images, (0, 2, 3, 1))
        return {'images': images}


def extract_images(mat_file_path, output_dir, num_images=5):
    """
    从 NYU Depth V2 .mat 文件中提取图像
    
    Args:
        mat_file_path: .mat 文件路径
        output_dir: 输出目录
        num_images: 提取的图像数量
    """
    print(f"正在加载数据集: {mat_file_path}")
    data = load_mat_file(mat_file_path)
    
    # 提取图像数据
    images = data['images']  # Shape: (N, H, W, C) 或 (N, 3, H, W)
    print(f"数据集包含 {len(images)} 张图像")
    print(f"图像形状: {images.shape}")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取图像
    num_images = min(num_images, len(images))
    print(f"\n正在提取 {num_images} 张测试图像...")
    
    for i in range(num_images):
        # 获取图像
        img = images[i]
        
        # 处理不同的图像格式
        if img.ndim == 3:
            # (H, W, C) 格式
            if img.shape[2] == 3:
                # RGB 图像
                img = img.astype(np.uint8)
            else:
                # 可能是 (C, H, W) 格式
                img = np.transpose(img, (1, 2, 0))
                img = img.astype(np.uint8)
        elif img.ndim == 2:
            # 灰度图，转换为 RGB
            img = np.stack([img, img, img], axis=2).astype(np.uint8)
        else:
            raise ValueError(f"不支持的图像维度: {img.ndim}")
        
        # 归一化到 0-255（如果值在 0-1 范围内）
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        pil_img = Image.fromarray(img)
        
        # 保存图像
        output_path = output_dir / f"test_image_{i:04d}.jpg"
        pil_img.save(output_path)
        print(f"  [{i+1}/{num_images}] 已保存: {output_path.name}")
    
    print(f"\n[OK] 测试图像已保存到: {output_dir}")
    print(f"   共 {num_images} 张图像")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="从 NYU Depth V2 数据集提取测试图像")
    parser.add_argument("--mat_file", type=str, 
                       default="./data/nyu_depth_v2/nyu_depth_v2_labeled.mat",
                       help="NYU Depth V2 .mat 文件路径")
    parser.add_argument("--output_dir", type=str,
                       default="./data/nyu_depth_v2/test_images",
                       help="输出目录")
    parser.add_argument("--num_images", type=int, default=5,
                       help="提取的图像数量")
    
    args = parser.parse_args()
    
    try:
        extract_images(args.mat_file, args.output_dir, args.num_images)
    except FileNotFoundError as e:
        print(f"[ERROR] 错误: {e}")
        print("\n请先下载数据集:")
        print("1. 访问: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html")
        print("2. 下载 'Labeled dataset (~2.8 GB)'")
        print(f"3. 保存到: {args.mat_file}")
        print("\n或运行: python download_data.py")
    except Exception as e:
        print(f"[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
