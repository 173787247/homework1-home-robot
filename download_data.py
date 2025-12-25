"""
下载 NYU Depth V2 数据集的脚本
支持断点续传和进度显示
"""
import os
import urllib.request
import argparse
from pathlib import Path
import sys


def download_file_with_resume(url, dest_path):
    """
    下载文件并支持断点续传
    
    Args:
        url: 下载链接
        dest_path: 保存路径
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查已下载的大小
    existing_size = 0
    if dest_path.exists():
        existing_size = dest_path.stat().st_size
        print(f"发现已存在的文件: {dest_path}")
        print(f"已下载: {existing_size / (1024**3):.2f} GB")
    
    # 创建请求
    req = urllib.request.Request(url)
    if existing_size > 0:
        req.add_header('Range', f'bytes={existing_size}-')
        print("继续下载...")
    else:
        print(f"开始下载: {url}")
        print(f"保存到: {dest_path}")
    
    try:
        # 打开连接
        with urllib.request.urlopen(req) as response:
            # 获取文件总大小
            total_size = int(response.headers.get('Content-Length', 0))
            if 'Content-Range' in response.headers:
                # 从 Content-Range 获取总大小
                content_range = response.headers['Content-Range']
                total_size = int(content_range.split('/')[-1])
            
            if total_size > 0:
                print(f"文件总大小: {total_size / (1024**3):.2f} GB")
            
            # 打开文件（追加模式如果已存在）
            mode = 'ab' if existing_size > 0 else 'wb'
            with open(dest_path, mode) as f:
                downloaded = existing_size
                block_size = 8192
                
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # 显示进度
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        downloaded_gb = downloaded / (1024**3)
                        total_gb = total_size / (1024**3)
                        print(f"\r进度: {percent:.1f}% ({downloaded_gb:.2f} GB / {total_gb:.2f} GB)", 
                              end='', flush=True)
                    else:
                        downloaded_gb = downloaded / (1024**3)
                        print(f"\r已下载: {downloaded_gb:.2f} GB", end='', flush=True)
        
        print("\n下载完成!")
        
        # 验证文件大小
        final_size = dest_path.stat().st_size
        if total_size > 0 and final_size != total_size:
            print(f"警告: 文件大小不匹配 (期望: {total_size}, 实际: {final_size})")
            return False
        else:
            print(f"文件大小: {final_size / (1024**3):.2f} GB")
            return True
            
    except urllib.error.HTTPError as e:
        if e.code == 416:  # Range Not Satisfiable
            print("\n文件已完整下载!")
            return True
        else:
            print(f"\n下载失败: HTTP {e.code}")
            raise
    except Exception as e:
        print(f"\n下载失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="下载 NYU Depth V2 数据集")
    parser.add_argument("--data_dir", type=str, default="./data/nyu_depth_v2",
                       help="数据保存目录")
    parser.add_argument("--force", action="store_true",
                       help="强制重新下载（删除已存在的文件）")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # NYU Depth V2 数据集下载链接
    dataset_urls = {
        "nyu_depth_v2_labeled.mat": "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
    }
    
    print("=" * 60)
    print("NYU Depth V2 数据集下载工具")
    print("=" * 60)
    print("\n注意：")
    print("1. 数据集文件较大（约2.8GB），请确保有足够的磁盘空间")
    print("2. 支持断点续传，如果下载中断可以重新运行继续下载")
    print("3. 如果下载失败，请手动从以下链接下载：")
    print("   https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html")
    print("4. 下载后请将文件放置在:", data_dir)
    print("=" * 60)
    
    # 下载文件
    for filename, url in dataset_urls.items():
        file_path = data_dir / filename
        
        # 如果强制重新下载，删除已存在的文件
        if args.force and file_path.exists():
            print(f"\n删除已存在的文件: {file_path}")
            file_path.unlink()
        
        # 检查是否已经完整下载
        if file_path.exists():
            file_size = file_path.stat().st_size
            expected_size = 2.8 * (1024**3)  # 约 2.8 GB
            
            if file_size >= expected_size * 0.95:  # 允许 5% 的误差
                print(f"\n文件已存在且大小正常: {file_path}")
                print(f"文件大小: {file_size / (1024**3):.2f} GB")
                response = input("是否重新下载? (y/n): ")
                if response.lower() != 'y':
                    print("跳过下载")
                    continue
        
        try:
            print(f"\n开始下载: {filename}")
            success = download_file_with_resume(url, file_path)
            if success:
                print(f"\n[OK] {filename} 下载成功!")
            else:
                print(f"\n[WARN] {filename} 下载完成但大小可能不完整")
        except KeyboardInterrupt:
            print("\n\n下载已中断")
            print("可以重新运行此脚本继续下载（支持断点续传）")
            sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] 下载失败: {e}")
            print(f"请手动下载: {url}")
            print(f"保存到: {file_path}")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("下载完成！")
    print("=" * 60)
    print(f"\n数据集位置: {data_dir}")
    print("现在可以运行: python extract_test_images.py")


if __name__ == "__main__":
    main()

