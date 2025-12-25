"""
Grounding 任务：物体定位和识别
"""
from PIL import Image
from typing import List, Dict, Union
import os
from ..models.grounding_dino import GroundingDINOModel


class GroundingTask:
    """Grounding 任务类"""
    
    def __init__(self, model_path: str = None, device: str = "cuda"):
        """
        初始化 Grounding 任务
        
        Args:
            model_path: Grounding DINO 模型路径
            device: 设备类型
        """
        self.model = GroundingDINOModel(model_path=model_path, device=device)
    
    def ground(self, text_prompt: str, image: Union[str, Image.Image],
               box_threshold: float = 0.3) -> List[Dict]:
        """
        对图像中的物体进行定位
        
        Args:
            text_prompt: 文本提示，如 "chair . table . lamp"
            image: 图像路径或 PIL Image 对象
            box_threshold: 边界框阈值
            
        Returns:
            检测结果列表
        """
        # 加载图像
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"图像文件不存在: {image}")
            image = Image.open(image).convert('RGB')
        
        # 执行检测
        results = self.model.detect(
            image=image,
            text_prompt=text_prompt,
            box_threshold=box_threshold
        )
        
        return results
    
    def ground_multiple(self, text_prompts: List[str], 
                       image: Union[str, Image.Image]) -> Dict[str, List[Dict]]:
        """
        对多个物体进行定位
        
        Args:
            text_prompts: 文本提示列表
            image: 图像路径或 PIL Image 对象
            
        Returns:
            字典，键为文本提示，值为检测结果
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        results = {}
        for prompt in text_prompts:
            results[prompt] = self.ground(prompt, image)
        
        return results

