from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
from pathlib import Path

from backend.clip_engine import ClipEngine
from backend.qwen_ranker import QwenRanker
from backend.image_processor import image_to_base64
import config

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# 初始化引擎
clip_engine = None
qwen_ranker = None
vectors_cache = None

def init_engines():
    """延迟初始化引擎"""
    global clip_engine, qwen_ranker, vectors_cache
    
    if clip_engine is None:
        clip_engine = ClipEngine(
            model_name=config.CLIP_MODEL_NAME,
            model_root=config.CLIP_MODEL_ROOT
        )
    
    if qwen_ranker is None and config.DASHSCOPE_API_KEY and config.DASHSCOPE_API_KEY != "your-api-key-here":
        qwen_ranker = QwenRanker(config.DASHSCOPE_API_KEY)
    
    # 加载向量缓存
    if vectors_cache is None and os.path.exists(config.VECTORS_FILE):
        with open(config.VECTORS_FILE, 'r') as f:
            vectors_cache = json.load(f)

@app.route('/')
def index():
    """提供前端页面"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """搜索接口"""
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', config.DEFAULT_TOP_K)
        use_rank = data.get('use_rank', True)
        
        if not query:
            return jsonify({'error': '请输入搜索内容'}), 400
        
        # 初始化引擎
        init_engines()
        
        if clip_engine is None:
            return jsonify({'error': 'CLIP引擎未初始化'}), 500
        
        if vectors_cache is None:
            return jsonify({'error': '请先运行encoder.py生成向量索引'}), 400
        
        # 1. CLIP检索
        results = clip_engine.search(query, vectors_cache, top_k)
        
        # 2. Qwen精排（可选）
        if use_rank and qwen_ranker:
            ranked = qwen_ranker.rank_images(query, results)
        else:
            # 不排序，按相似度降序全部归入'2'
            ranked = {'2': results, '1': [], '0': []}
        
        # 3. 准备返回数据
        response = {
            'query': query,
            'results': {
                'relevant': [],
                'partial': [],
                'irrelevant': []
            }
        }
        
        for path, score in ranked['2']:
            b64 = image_to_base64(path)
            if b64:
                response['results']['relevant'].append({
                    'path': path,
                    'filename': os.path.basename(path),
                    'similarity': score,
                    'base64': b64
                })
        
        for path, score in ranked['1']:
            b64 = image_to_base64(path)
            if b64:
                response['results']['partial'].append({
                    'path': path,
                    'filename': os.path.basename(path),
                    'similarity': score,
                    'base64': b64
                })
        
        for path, score in ranked['0']:
            b64 = image_to_base64(path)
            if b64:
                response['results']['irrelevant'].append({
                    'path': path,
                    'filename': os.path.basename(path),
                    'similarity': score,
                    'base64': b64
                })
        
        return jsonify(response)
    
    except Exception as e:
        print(f"搜索错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/images/<path:filename>')
def get_image(filename):
    """提供图片文件"""
    return send_from_directory(config.IMAGES_DIR, filename)

@app.route('/api/status', methods=['GET'])
def status():
    """检查系统状态"""
    init_engines()
    
    has_vectors = os.path.exists(config.VECTORS_FILE)
    has_api_key = config.DASHSCOPE_API_KEY and config.DASHSCOPE_API_KEY != "your-api-key-here"
    
    return jsonify({
        'clip_loaded': clip_engine is not None,
        'qwen_available': qwen_ranker is not None and has_api_key,
        'vectors_exists': has_vectors,
        'images_count': len([f for f in os.listdir(config.IMAGES_DIR) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(config.IMAGES_DIR) else 0
    })

if __name__ == '__main__':
    # 确保必要的目录存在
    Path(config.IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.CLIP_MODEL_ROOT).mkdir(parents=True, exist_ok=True)
    
    app.run(debug=config.DEBUG, host='0.0.0.0', port=5000)