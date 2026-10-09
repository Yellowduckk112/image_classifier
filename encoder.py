import cn_clip.clip as clip
from PIL import Image
import torch
import os
import json
from pathlib import Path

# 配置路径
IMAGES_DIR = "./data/images"
VECTORS_FILE = "./data/vectors.json"
MODEL_ROOT = "./models"

def encode_image(image_path, model, preprocess, device="cpu"):
    """编码单张图片"""
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features.cpu().numpy()

def main():
    # 检查图片目录
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ 图片目录不存在: {IMAGES_DIR}")
        print("请创建目录并将图片放入其中")
        return
    
    # 获取所有图片
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(Path(IMAGES_DIR).glob(f'*{ext}'))
        image_files.extend(Path(IMAGES_DIR).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ 在 {IMAGES_DIR} 中未找到图片文件")
        return
    
    print(f"📁 找到 {len(image_files)} 张图片")
    
    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  使用设备: {device}")
    print("📦 加载CLIP模型...")
    model, preprocess = clip.load_from_name("ViT-L-14", device=device, download_root=MODEL_ROOT)
    print("✅ 模型加载完成")
    
    # 编码所有图片
    output = []
    for idx, img_path in enumerate(image_files, 1):
        print(f"⏳ 编码 {idx}/{len(image_files)}: {img_path.name}")
        try:
            image_features = encode_image(str(img_path), model, preprocess, device)
            output.append({
                "path": str(img_path),
                "filename": img_path.name,
                "features": image_features.tolist()
            })
        except Exception as e:
            print(f"   ❌ 编码失败: {e}")
    
    # 保存结果
    with open(VECTORS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=1)
    
    print(f"✅ 完成！向量索引已保存到: {VECTORS_FILE}")

if __name__ == "__main__":
    main()