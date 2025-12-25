"""
测试脚本：检查所有模块是否可以正常导入
"""
import sys
import traceback
import io

# 设置输出编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_imports():
    """测试所有模块导入"""
    errors = []
    
    # 测试主模块导入
    modules_to_test = [
        ("src.models.grounding_dino", "GroundingDINOModel"),
        ("src.models.blip2", "BLIP2Model"),
        ("src.tasks.grounding", "GroundingTask"),
        ("src.tasks.counting", "CountingTask"),
        ("src.tasks.vqa", "VQATask"),
        ("src.data.dataset", "NYUDepthV2Dataset"),
        ("src.utils.visualization", "visualize_results"),
    ]
    
    print("=" * 50)
    print("开始测试模块导入...")
    print("=" * 50)
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"[OK] {module_name}.{class_name} - 导入成功")
        except ImportError as e:
            print(f"[WARN] {module_name}.{class_name} - 导入失败（缺少依赖）: {e}")
            errors.append((module_name, class_name, str(e)))
        except AttributeError as e:
            print(f"[ERROR] {module_name}.{class_name} - 类不存在: {e}")
            errors.append((module_name, class_name, str(e)))
        except Exception as e:
            print(f"[ERROR] {module_name}.{class_name} - 错误: {e}")
            errors.append((module_name, class_name, str(e)))
    
    print("=" * 50)
    if errors:
        print(f"发现 {len(errors)} 个问题:")
        for module, cls, error in errors:
            print(f"  - {module}.{cls}: {error}")
        print("\n注意: 某些导入失败可能是因为缺少依赖包，这是正常的。")
        print("只要代码结构正确，安装依赖后即可正常运行。")
    else:
        print("[OK] 所有模块导入成功！")
    print("=" * 50)
    
    return len(errors) == 0

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

