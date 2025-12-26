"""
可视化工具
"""
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict
import os


def visualize_results(image: Image.Image, results: List[Dict], 
                     output_path: str = "output.jpg"):
    """
    可视化检测结果
    
    Args:
        image: PIL Image 对象
        results: 检测结果列表
        output_path: 输出路径
    """
    # 创建副本用于绘制
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    # 绘制边界框和标签
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, result in enumerate(results):
        bbox = result['bbox']
        label = result['label']
        score = result['score']
        
        color = colors[i % len(colors)]
        
        # 绘制边界框
        draw.rectangle(bbox, outline=color, width=3)
        
        # 绘制标签
        label_text = f"{label}: {score:.2f}"
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 标签背景
        draw.rectangle(
            [bbox[0], bbox[1] - text_height - 4, 
             bbox[0] + text_width + 4, bbox[1]],
            fill=color
        )
        
        # 标签文本
        draw.text(
            (bbox[0] + 2, bbox[1] - text_height - 2),
            label_text,
            fill=(255, 255, 255),
            font=font
        )
    
    # 保存结果
    draw_image.save(output_path)
    print(f"可视化结果已保存到: {output_path}")


