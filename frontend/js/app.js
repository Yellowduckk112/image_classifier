// 状态管理
const state = {
    isLoading: false,
    currentQuery: '',
    results: null
};

// DOM元素
const elements = {
    searchInput: document.getElementById('searchInput'),
    searchBtn: document.getElementById('searchBtn'),
    useQwen: document.getElementById('useQwen'),
    topKSelect: document.getElementById('topKSelect'),
    loading: document.getElementById('loading'),
    resultStats: document.getElementById('resultStats'),
    relevantGrid: document.getElementById('relevantGrid'),
    partialGrid: document.getElementById('partialGrid'),
    irrelevantGrid: document.getElementById('irrelevantGrid'),
    relevantCount: document.getElementById('relevantCount'),
    partialCount: document.getElementById('partialCount'),
    irrelevantCount: document.getElementById('irrelevantCount'),
    sysStatus: document.getElementById('sysStatus'),
    imageCount: document.getElementById('imageCount'),
    qwenStatus: document.getElementById('qwenStatus')
};

// 事件绑定
elements.searchBtn.addEventListener('click', handleSearch);
elements.searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSearch();
});

// 初始化
init();

async function init() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        elements.sysStatus.textContent = data.clip_loaded ? '✅ 就绪' : '❌ 未加载';
        elements.sysStatus.style.color = data.clip_loaded ? '#4ade80' : '#f87171';
        
        elements.imageCount.textContent = data.images_count;
        
        if (data.qwen_available) {
            elements.qwenStatus.textContent = '✅ 已启用';
            elements.qwenStatus.style.color = '#4ade80';
        } else {
            elements.qwenStatus.textContent = '⚠️ 未配置API Key';
            elements.qwenStatus.style.color = '#fbbf24';
            document.getElementById('useQwen').disabled = true;
        }
    } catch (error) {
        console.error('状态检查失败:', error);
        elements.sysStatus.textContent = '❌ 连接失败';
    }
}

async function handleSearch() {
    const query = elements.searchInput.value.trim();
    if (!query) {
        alert('请输入搜索内容');
        return;
    }
    
    if (state.isLoading) return;
    
    // 清空结果
    clearResults();
    
    // 显示加载
    state.isLoading = true;
    elements.loading.style.display = 'block';
    elements.searchBtn.disabled = true;
    elements.searchBtn.textContent = '搜索中...';
    
    try {
        const topK = parseInt(elements.topKSelect.value);
        const useQwen = elements.useQwen.checked;
        
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                top_k: topK,
                use_rank: useQwen
            })
        });
        
        if (!response.ok) {
            throw new Error('搜索失败');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        console.error('搜索错误:', error);
        alert('搜索失败，请检查后端服务');
    } finally {
        state.isLoading = false;
        elements.loading.style.display = 'none';
        elements.searchBtn.disabled = false;
        elements.searchBtn.textContent = '🔍 搜索';
    }
}

function displayResults(data) {
    const results = data.results;
    
    // 统计信息
    const total = results.relevant.length + results.partial.length + results.irrelevant.length;
    elements.resultStats.style.display = 'block';
    elements.resultStats.innerHTML = `
        找到 <strong>${total}</strong> 张相关图片
        （✅ 紧密: ${results.relevant.length} | 🔶 部分: ${results.partial.length} | ❌ 不相关: ${results.irrelevant.length}）
        <span style="color:#888;margin-left:12px;">查询: "${data.query}"</span>
    `;
    
    // 渲染各组
    renderGrid(elements.relevantGrid, results.relevant);
    renderGrid(elements.partialGrid, results.partial);
    renderGrid(elements.irrelevantGrid, results.irrelevant);
    
    elements.relevantCount.textContent = results.relevant.length;
    elements.partialCount.textContent = results.partial.length;
    elements.irrelevantCount.textContent = results.irrelevant.length;
}

function renderGrid(gridElement, images) {
    gridElement.innerHTML = '';
    
    if (images.length === 0) {
        gridElement.innerHTML = '<p style="color:#666;padding:20px;text-align:center;">暂无图片</p>';
        return;
    }
    
    images.forEach(img => {
        const card = document.createElement('div');
        card.className = 'image-card';
        card.innerHTML = `
            <img src="${img.base64}" alt="${img.filename}" loading="lazy" />
            <div class="info">
                <div class="filename" title="${img.filename}">📄 ${img.filename}</div>
                <div class="similarity">相似度: ${(img.similarity * 100).toFixed(2)}%</div>
            </div>
        `;
        gridElement.appendChild(card);
    });
}

function clearResults() {
    elements.relevantGrid.innerHTML = '';
    elements.partialGrid.innerHTML = '';
    elements.irrelevantGrid.innerHTML = '';
    elements.resultStats.style.display = 'none';
    elements.relevantCount.textContent = '0';
    elements.partialCount.textContent = '0';
    elements.irrelevantCount.textContent = '0';
}