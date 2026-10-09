from backend.app import app

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          🖼️  CLIP 图片检索系统                          ║
    ║                                                          ║
    ║  访问地址: http://localhost:5000                        ║
    ║                                                          ║
    ║  使用前请先运行: python encoder.py                      ║
    ║  生成图片向量索引                                       ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    app.run(debug=True, host='0.0.0.0', port=5000)