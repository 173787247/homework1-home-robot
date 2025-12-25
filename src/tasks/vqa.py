"""
VQA 任务：视觉问答
"""
from PIL import Image
from typing import Union
import os
from ..models.blip2 import BLIP2Model


class VQATask:
    """VQA 任务类"""
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b",
                 device: str = "cuda", precision: str = "fp16"):
        """
        初始化 VQA 任务
        
        Args:
            model_name: BLIP-2 模型名称
            device: 设备类型
            precision: 精度类型
        """
        self.model = BLIP2Model(
            model_name=model_name,
            device=device,
            precision=precision
        )
    
    def answer(self, question: str, image: Union[str, Image.Image],
               max_length: int = 50) -> str:
        """
        回答关于图像的问题
        
        Args:
            question: 问题文本
            image: 图像路径或 PIL Image 对象
            max_length: 最大答案长度
            
        Returns:
            答案文本
        """
        # 加载图像
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"图像文件不存在: {image}")
            image = Image.open(image).convert('RGB')
        
        # 生成答案
        answer = self.model.answer_question(image, question)
        
        return answer
    
    def describe(self, image: Union[str, Image.Image]) -> str:
        """
        描述图像内容
        
        Args:
            image: 图像路径或 PIL Image 对象
            
        Returns:
            图像描述
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        description = self.model.describe_image(image)
        return description

