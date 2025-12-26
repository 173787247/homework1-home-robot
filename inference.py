"""
推理脚本：对单张图像进行 grounding、counting 或 VQA
"""
import argparse
from PIL import Image
from src.tasks.grounding import GroundingTask
from src.tasks.counting import CountingTask
from src.tasks.vqa import VQATask
from src.utils.visualization import visualize_results


def main():
    parser = argparse.ArgumentParser(description="家居机器人推理脚本")
    parser.add_argument("--image", type=str, required=True,
                       help="输入图像路径")
    parser.add_argument("--task", type=str, required=True,
                       choices=["grounding", "counting", "vqa"],
                       help="任务类型")
    parser.add_argument("--text", type=str, default="",
                       help="文本提示（grounding/counting）或问题（vqa）")
    parser.add_argument("--output", type=str, default="output.jpg",
                       help="输出图像路径")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备类型")
    
    args = parser.parse_args()
    
    # 加载图像
    try:
        image = Image.open(args.image).convert('RGB')
        print(f"已加载图像: {args.image}")
    except Exception as e:
        print(f"加载图像失败: {e}")
        return
    
    # 执行任务
    if args.task == "grounding":
        if not args.text:
            print("错误: grounding 任务需要 --text 参数")
            return
        
        print(f"执行 Grounding 任务: {args.text}")
        task = GroundingTask(device=args.device)
        results = task.ground(args.text, image)
        
        print(f"检测到 {len(results)} 个物体:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['label']}: {result['score']:.3f} "
                  f"bbox={result['bbox']}")
        
        # 可视化结果
        visualize_results(image, results, args.output)
        print(f"结果已保存到: {args.output}")
    
    elif args.task == "counting":
        if not args.text:
            print("错误: counting 任务需要 --text 参数")
            return
        
        print(f"执行 Counting 任务: {args.text}")
        task = CountingTask(device=args.device)
        count = task.count(args.text, image)
        
        print(f"检测到 {count} 个 '{args.text}'")
    
    elif args.task == "vqa":
        if not args.text:
            print("错误: vqa 任务需要 --text 参数（问题）")
            return
        
        print(f"执行 VQA 任务")
        print(f"问题: {args.text}")
        task = VQATask(device=args.device)
        answer = task.answer(args.text, image)
        
        print(f"答案: {answer}")


if __name__ == "__main__":
    main()


