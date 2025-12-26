"""
Counting 任务：统计图像中物体的数量
"""
from PIL import Image
from typing import Union
import os
from ..models.grounding_dino import GroundingDINOModel


class CountingTask:
    """Counting 任务类"""
    
    def __init__(self, model_path: str = None, device: str = "cuda"):
        """
        初始化 Counting 任务
        
        Args:
            model_path: Grounding DINO 模型路径
            device: 设备类型
        """
        self.model = GroundingDINOModel(model_path=model_path, device=device)
    
    def count(self, object_name: str, image: Union[str, Image.Image],
              threshold: float = 0.3) -> int:
        """
        统计图像中指定物体的数量
        
        Args:
            object_name: 物体名称，如 "chair", "table"
            image: 图像路径或 PIL Image 对象
            threshold: 检测阈值
            
        Returns:
            物体数量
        """
        # 加载图像
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"图像文件不存在: {image}")
            image = Image.open(image).convert('RGB')
        
        # 使用 Grounding DINO 检测物体
        text_prompt = object_name
        results = self.model.detect(
            image=image,
            text_prompt=text_prompt,
            box_threshold=threshold
        )
        
        # 统计数量
        count = len([r for r in results if r['score'] >= threshold])
        
        return count
    
    def count_multiple(self, object_names: list, 
                      image: Union[str, Image.Image]) -> dict:
        """
        统计多个物体的数量
        
        Args:
            object_names: 物体名称列表
            image: 图像路径或 PIL Image 对象
            
        Returns:
            字典，键为物体名称，值为数量
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        counts = {}
        for obj_name in object_names:
            counts[obj_name] = self.count(obj_name, image)
        
        return counts


