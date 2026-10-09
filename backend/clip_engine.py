import torch
import cn_clip.clip as clip
from PIL import Image
import json
import os
from typing import List, Dict, Tuple

class ClipEngine:
    def __init__(self, model_name="ViT-L-14", device=None, model_root="./models"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model_root = model_root
        
        # 加载模型
        print(f"加载CLIP模型 {model_name} 到 {self.device}...")
        self.model, self.preprocess = clip.load_from_name(
            model_name, 
            device=self.device, 
            download_root=model_root
        )
        print("CLIP模型加载完成")
    
    def encode_text(self, text: str) -> torch.Tensor:
        """编码文本"""
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        return text_features
    
    def encode_image_path(self, image_path: str) -> torch.Tensor:
        """编码单张图片"""
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        return image_features
    
    def compute_similarity(self, text_features: torch.Tensor, image_features: torch.Tensor) -> float:
        """计算余弦相似度"""
        # 确保为浮点类型，避免类型不匹配
        text_features = text_features.half()
        image_features = image_features.half()
        text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (text_norm @ image_norm.t()).squeeze().item()
        return similarity
    
    def search(self, query: str, vectors_data: List[Dict], top_k: int = 7) -> List[Tuple[str, float]]:
        """
        检索最相似的图片
        返回: [(path, similarity_score), ...]
        """
        text_features = self.encode_text(query)
        # 转为 float 确保后续计算一致
        text_features = text_features.half()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        results = []
        for item in vectors_data:
            # 转换特征向量为tensor
            image_vector = torch.tensor(item["features"]).to(self.device).half()
            image_vector = image_vector / image_vector.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            similarity = (text_features @ image_vector.t()).squeeze().item()
            results.append((item["path"], similarity))
        
        # 排序并返回top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]