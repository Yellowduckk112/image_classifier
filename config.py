import os

# API配置（请替换为自己的Key）
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

# 模型配置
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_MODEL_ROOT = "./models"

# 检索配置
DEFAULT_TOP_K = 10
IMAGE_MAX_SIZE = 512  # 压缩图片到这个尺寸以内

# 路径配置
IMAGES_DIR = "./data/images"
VECTORS_FILE = "./data/vectors.json"

# Flask配置
SECRET_KEY = "dev-secret-key-change-in-production"
DEBUG = True