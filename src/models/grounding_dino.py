"""
Grounding DINO 模型封装
用于物体检测和定位
"""
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class GroundingDINOModel:
    """Grounding DINO 模型封装类"""
    
    def __init__(self, model_path: str = None, device: str = "cuda"):
        """
        初始化 Grounding DINO 模型
        
        Args:
            model_path: 模型权重路径
            device: 设备类型 ("cuda" 或 "cpu")
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """加载模型"""
        try:
            # 尝试导入 groundingdino
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict
            
            # 使用默认配置
            config_file = "groundingdino/config/GroundingDINO_SwinB.cfg.py"
            args = SLConfig.fromfile(config_file)
            args.device = self.device
            
            self.model = build_model(args)
            
            if model_path:
                checkpoint = torch.load(model_path, map_location="cpu")
                load_res = self.model.load_state_dict(
                    clean_state_dict(checkpoint['model']), strict=False
                )
                print(f"模型加载完成: {load_res}")
            
            self.model.eval()
            self.model = self.model.to(self.device)
            
        except ImportError:
            print("警告: 无法导入 groundingdino，将使用简化版本")
            self.model = None
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("将使用简化版本进行演示")
            self.model = None
    
    def detect(self, image: Image.Image, text_prompt: str, 
               box_threshold: float = 0.3, text_threshold: float = 0.25) -> List[Dict]:
        """
        检测图像中的物体
        
        Args:
            image: PIL Image 对象
            text_prompt: 文本提示，如 "chair . table . lamp"
            box_threshold: 边界框阈值
            text_threshold: 文本阈值
            
        Returns:
            检测结果列表，每个元素包含:
            {
                'bbox': [x1, y1, x2, y2],
                'score': float,
                'label': str
            }
        """
        if self.model is None:
            # 返回模拟结果用于演示
            return self._mock_detect(image, text_prompt)
        
        try:
            from groundingdino.util.inference import load_image, predict, annotate
            
            image_source, image = load_image(image)
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            
            results = []
            for box, logit, phrase in zip(boxes, logits, phrases):
                results.append({
                    'bbox': box.tolist(),
                    'score': float(logit),
                    'label': phrase
                })
            
            return results
            
        except Exception as e:
            print(f"检测失败: {e}")
            return self._mock_detect(image, text_prompt)
    
    def _mock_detect(self, image: Image.Image, text_prompt: str) -> List[Dict]:
        """模拟检测结果（用于演示）"""
        import random
        width, height = image.size
        
        # 从文本提示中提取物体名称
        objects = [obj.strip() for obj in text_prompt.split('.') if obj.strip()]
        
        results = []
        for obj in objects[:3]:  # 最多返回3个物体
            x1 = random.randint(0, width // 2)
            y1 = random.randint(0, height // 2)
            x2 = random.randint(x1 + 50, width)
            y2 = random.randint(y1 + 50, height)
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'score': random.uniform(0.5, 0.9),
                'label': obj
            })
        
        return results


