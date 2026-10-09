import torch
import cn_clip.clip as clip
from PIL import Image
import json
import functools
import io
import matplotlib.pyplot as plt
import base64
import dashscope
from dashscope import MultiModalConversation

class image:
    def __init__(self, path, vector=None):
        self.path = path
        self.vector = vector

def encode_text(text, model, device="cpu"):
    # 编码文本
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    return text_features

@functools.lru_cache(maxsize=1000)
def encode_image(image_path, max_size=512):
    img = Image.open(image_path)
    
    # 转换为 RGB（去除透明通道等）
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # 等比例压缩
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    
    # 高质量压缩
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85, optimize=True)
    
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# 加载模型（自动下载到本地缓存）
device = "cuda" if torch.cuda.is_available() else "cpu"
test_text = "二次元美少女"  # 替换为你的测试文本
model, preprocess = clip.load_from_name("ViT-L-14", device=device, download_root="./models")
k = 7  # 替换为你想要的 top k 数量

text_features = encode_text(test_text, model, device)
paths = []
similaritys = []

with open("vectors.json", "r") as f:
    load_data = json.load(f)
data = load_data

for i in range(len(data)):
    data[i]["base64"] = encode_image(data[i]["path"])

for item in data:
    image_vector = torch.tensor(item["features"])
    # print(type(image_vector[0][0]), type(text_features[0][0]), image_vector.shape, text_features.shape)
    image_vector = image_vector.to(device)
    similarity = (text_features.float() @ image_vector.t()) / (text_features.norm() * image_vector.norm())
    paths.append(item["path"])
    similaritys.append(similarity[0][0].item())

k = min(k, len(similaritys))
top_k = sorted(zip(similaritys, paths, [item["base64"] for item in data]), key=lambda x: x[0], reverse=True)[:k]

messages = [
    {
        "role": "user",
        "content": [
            {"text": "对以下这几张图片关于给定文字进行分类，按照给定图片的顺序给出答案2、1或者0。2表示该图片与文本关系紧密，\
             几乎就是文本描述的内容;1表示该图片与文本关系有一定关联，但不直接是文本描述的内容;0表示该图片与文本完全无关。你的输\
             出格式必须严格按照以下格式：一个长度为给定图片数量的字符串，字符串中的每个字符必须是2、1或者0，且不能有任何其他字符，\
             包括空格。例如，如果有5张图片，输出可能是：21020。请严格按照这个格式输出，不要添加任何解释或者其他文本。"},
            {"text": f"文本描述：{test_text}"},
        ]
    }
]
for similarity, path, base64_data in top_k:
    messages[0]["content"].append({"image": base64_data})

response = MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-101192fb0181444e9108b1eef4283fa5",
    model="qwen3-vl-flash",
    messages=messages,
    stream=True,
    # enable_thinking 参数开启思考过程
    enable_thinking=False,
    enable_search=True,
    # thinking_budget 参数设置最大推理过程 Token 数
    thinking_budget=81920,

)

full_text = ''
for chunk in response:
    if chunk.output and chunk.output.choices and chunk.output.choices[0].message and chunk.output.choices[0].message.content:
        content = chunk.output.choices[0].message.content[0]["text"]
        if content:
            full_text += content

# 输出最终完整回复
print(full_text)

for i in range(len(top_k)):
    if full_text[i] == "2":
        image = Image.open(top_k[i][1]).convert("RGB")
        plt.imshow(image)
        plt.title(f"Similarity: {top_k[i][0]:.4f}")
        plt.axis("off")
        plt.show()

for i in range(len(top_k)):
    if full_text[i] == "1":
        image = Image.open(top_k[i][1]).convert("RGB")
        plt.imshow(image)
        plt.title(f"Similarity: {top_k[i][0]:.4f}")
        plt.axis("off")
        plt.show()