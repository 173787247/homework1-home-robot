"""
BLIP-2 模型封装
用于视觉问答和图像理解
"""
import torch
from PIL import Image
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


class BLIP2Model:
    """BLIP-2 模型封装类"""
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", 
                 device: str = "cuda", precision: str = "fp16"):
        """
        初始化 BLIP-2 模型
        
        Args:
            model_name: HuggingFace 模型名称
            device: 设备类型
            precision: 精度类型 ("fp16" 或 "fp32")
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.precision = precision
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和处理器"""
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            print(f"正在加载 BLIP-2 模型: {self.model_name}")
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            
            # 根据精度选择加载方式
            if self.precision == "fp16" and self.device == "cuda":
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                ).to(self.device)
            
            self.model.eval()
            print("模型加载完成!")
            
        except ImportError:
            print("警告: 无法导入 transformers，将使用简化版本")
            self.model = None
            self.processor = None
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("将使用简化版本进行演示")
            self.model = None
            self.processor = None
    
    def generate(self, image: Image.Image, prompt: str, 
                 max_length: int = 50) -> str:
        """
        生成回答
        
        Args:
            image: PIL Image 对象
            prompt: 问题或提示
            max_length: 最大生成长度
            
        Returns:
            生成的文本
        """
        if self.model is None or self.processor is None:
            return self._mock_generate(prompt)
        
        try:
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
            
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3
            )
            
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"生成失败: {e}")
            return self._mock_generate(prompt)
    
    def answer_question(self, image: Image.Image, question: str) -> str:
        """
        回答关于图像的问题
        
        Args:
            image: PIL Image 对象
            question: 问题文本
            
        Returns:
            答案
        """
        prompt = f"Question: {question} Answer:"
        return self.generate(image, prompt)
    
    def describe_image(self, image: Image.Image) -> str:
        """
        描述图像内容
        
        Args:
            image: PIL Image 对象
            
        Returns:
            图像描述
        """
        prompt = "Describe this image in detail:"
        return self.generate(image, prompt)
    
    def _mock_generate(self, prompt: str) -> str:
        """模拟生成结果（用于演示）"""
        # 简单的规则匹配用于演示
        prompt_lower = prompt.lower()
        
        if "what" in prompt_lower and "room" in prompt_lower:
            return "This is a living room with furniture including chairs, a table, and a lamp."
        elif "how many" in prompt_lower:
            return "There are 3 chairs in the room."
        elif "where" in prompt_lower:
            return "The chair is located in the center of the room."
        elif "describe" in prompt_lower:
            return "This is a modern living room with contemporary furniture, soft lighting, and a clean design."
        else:
            return "This appears to be a well-furnished indoor space with various household items."

