import base64
import io
from PIL import Image

def image_to_base64(image_path, max_size=512, quality=85):
    """
    将图片转为压缩的Base64编码
    """
    try:
        img = Image.open(image_path)
        
        # 转换为RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # 等比例压缩
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        
        # 高质量压缩为JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"图片处理失败 {image_path}: {e}")
        return None

def validate_image(path):
    """验证是否为有效图片"""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False