import dashscope
from dashscope import MultiModalConversation
from typing import List, Tuple
import time

class QwenRanker:
    def __init__(self, api_key: str):
        self.api_key = api_key
        dashscope.api_key = api_key
    
    def rank_images(self, query: str, images: List[Tuple[str, float]], top_k: int = None) -> dict:
        """
        使用Qwen-VL精排图片
        返回: {
            '2': [(path, similarity), ...],  # 紧密相关
            '1': [(path, similarity), ...],  # 部分相关
            '0': [(path, similarity), ...]   # 不相关
        }
        """
        if not images:
            return {'2': [], '1': [], '0': []}
        
        # 准备消息
        messages = [{
            "role": "user",
            "content": [
                {"text": self._build_prompt(query, len(images))},
                {"text": f"文本描述：{query}"},
            ]
        }]
        
        # 添加图片（base64）
        from backend.image_processor import image_to_base64
        for path, _ in images:
            b64 = image_to_base64(path)
            if b64:
                messages[0]["content"].append({"image": b64})
        
        # 调用API
        try:
            response = MultiModalConversation.call(
                model="qwen3-vl-flash",
                messages=messages,
                stream=True,
                enable_thinking=False,
                enable_search=False,  # 离线场景关闭搜索
            )
            
            # 收集响应
            full_text = ''
            for chunk in response:
                if chunk.output and chunk.output.choices:
                    content = chunk.output.choices[0].message.content
                    if content and isinstance(content, list) and len(content) > 0:
                        text = content[0].get("text", "")
                        if text:
                            full_text += text
            
            # 解析结果
            return self._parse_result(full_text, images)
            
        except Exception as e:
            print(f"Qwen排序失败: {e}")
            # 失败时按相似度降序全部放入'2'
            return {'2': images, '1': [], '0': []}
    
    def _build_prompt(self, query: str, image_count: int) -> str:
        """构建提示词"""
        return f"""对以下这几张图片关于给定文字进行分类，按照给定图片的顺序给出答案2、1或者0。
2表示该图片与文本关系紧密，几乎就是文本描述的内容；
1表示该图片与文本关系有一定关联，但不直接是文本描述的内容；
0表示该图片与文本完全无关。
你的输出格式必须严格按照以下格式：一个长度为{image_count}的字符串，字符串中的每个字符必须是2、1或者0，且不能有任何其他字符，包括空格。
例如，如果有5张图片，输出可能是：21020。请严格按照这个格式输出，不要添加任何解释或者其他文本。"""
    
    def _parse_result(self, result_text: str, images: List[Tuple[str, float]]) -> dict:
        """解析Qwen返回的结果"""
        # 清理结果（只保留数字）
        cleaned = ''.join(filter(lambda x: x in '012', result_text.strip()))
        
        if len(cleaned) != len(images):
            print(f"警告: 期望{len(images)}个结果，实际得到{len(cleaned)}个")
            # 补齐或截断
            if len(cleaned) < len(images):
                cleaned = cleaned + '0' * (len(images) - len(cleaned))
            else:
                cleaned = cleaned[:len(images)]
        
        # 分类
        result = {'2': [], '1': [], '0': []}
        for idx, label in enumerate(cleaned):
            if idx < len(images):
                result[label].append(images[idx])
        
        return result