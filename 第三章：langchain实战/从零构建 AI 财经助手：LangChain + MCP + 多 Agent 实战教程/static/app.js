/**
 * 财经智能助手 - 暗色卡片式UI
 * 富文本 Markdown 渲染 + SSE 流式通信
 */

class FinanceChatApp {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.userInput = document.getElementById('userInput');
        this.sendButton = document.getElementById('sendButton');
        this.isProcessing = false;
        this.currentEventSource = null;
        this.thinkingStartTime = null;
        this.toolCallCount = 0;
        this.init();
    }

    init() {
        this.sendButton.addEventListener('click', () => this.handleSend());
        this.userInput.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                this.handleSend();
            }
        });
        this.userInput.addEventListener('input', () => this.autoResize());

        // 快捷按钮
        document.querySelectorAll('.quick-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const query = btn.dataset.query;
                if (query && !this.isProcessing) {
                    this.userInput.value = query;
                    this.handleSend();
                }
            });
        });

        this.userInput.focus();
    }

    autoResize() {
        this.userInput.style.height = 'auto';
        this.userInput.style.height = Math.min(this.userInput.scrollHeight, 120) + 'px';
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    // ============================================================
    // 发送消息
    // ============================================================

    async handleSend() {
        const message = this.userInput.value.trim();
        if (!message || this.isProcessing) return;

        this.isProcessing = true;
        this.sendButton.disabled = true;
        this.toolCallCount = 0;

        // 添加用户消息
        this.addUserMessage(message);
        this.userInput.value = '';
        this.userInput.style.height = 'auto';

        // 创建机器人消息容器
        const botMsg = this.createBotMessage();
        this.thinkingStartTime = Date.now();

        try {
            await this.streamResponse(message, botMsg);
        } catch (err) {
            this.addError(botMsg, '请求失败: ' + err.message);
        } finally {
            this.isProcessing = false;
            this.sendButton.disabled = false;
            this.userInput.focus();
        }
    }

    // ============================================================
    // 消息 DOM 操作
    // ============================================================

    addUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'msg msg-user';
        div.innerHTML = `<div class="msg-bubble">${this.escapeHtml(text)}</div>`;
        this.chatMessages.appendChild(div);
        this.scrollToBottom();
    }

    createBotMessage() {
        const div = document.createElement('div');
        div.className = 'msg msg-bot';
        div.innerHTML = `
            <div class="msg-body">
                <div class="thinking-section" id="thinking-${Date.now()}">
                    <div class="thinking-header" onclick="this.parentElement.classList.toggle('collapsed')">
                        <div class="thinking-icon">🔍</div>
                        <span class="thinking-text">深度思考中...</span>
                        <span class="thinking-toggle">▲</span>
                    </div>
                    <div class="thinking-steps"></div>
                </div>
                <div class="msg-content" style="display:none"></div>
            </div>
        `;
        this.chatMessages.appendChild(div);
        this.scrollToBottom();
        return div;
    }

    addThinkingStep(botMsg, label, details) {
        const steps = botMsg.querySelector('.thinking-steps');
        if (!steps) return;

        // 标记之前的步骤为完成
        steps.querySelectorAll('.thinking-step.in-progress').forEach(el => {
            el.classList.remove('in-progress');
            el.querySelector('.step-check').textContent = '✓';
        });

        const step = document.createElement('div');
        step.className = 'thinking-step in-progress';
        step.innerHTML = `
            <span class="step-check">○</span>
            <span class="step-label">${this.escapeHtml(label)}</span>
        `;
        steps.appendChild(step);

        if (details) {
            const detailDiv = document.createElement('div');
            detailDiv.className = 'step-details';
            if (Array.isArray(details)) {
                details.forEach(d => {
                    detailDiv.innerHTML += `<div class="step-detail-item">${this.escapeHtml(d)}</div>`;
                });
            } else {
                detailDiv.innerHTML = `<div class="step-detail-item">${this.escapeHtml(details)}</div>`;
            }
            steps.appendChild(detailDiv);
        }

        this.scrollToBottom();
    }

    finishThinking(botMsg) {
        const section = botMsg.querySelector('.thinking-section');
        if (!section) return;

        // 标记所有步骤完成
        section.querySelectorAll('.thinking-step.in-progress').forEach(el => {
            el.classList.remove('in-progress');
            el.querySelector('.step-check').textContent = '✓';
        });

        // 添加"生成答案"步骤
        const steps = section.querySelector('.thinking-steps');
        const step = document.createElement('div');
        step.className = 'thinking-step';
        step.innerHTML = `<span class="step-check">✓</span><span class="step-label">生成答案</span>`;
        steps.appendChild(step);

        // 更新思考时间
        const elapsed = ((Date.now() - this.thinkingStartTime) / 1000).toFixed(1);
        section.querySelector('.thinking-text').textContent = `已深度思考(用时${elapsed}秒)`;
    }

    addError(botMsg, text) {
        // 隐藏思考区
        const thinking = botMsg.querySelector('.thinking-section');
        if (thinking) thinking.style.display = 'none';

        const contentDiv = botMsg.querySelector('.msg-content');
        contentDiv.style.display = 'block';
        contentDiv.innerHTML = `<div class="msg-error">${this.escapeHtml(text)}</div>`;
        this.scrollToBottom();
    }

    // ============================================================
    // SSE 流式通信
    // ============================================================

    streamResponse(message, botMsg) {
        return new Promise((resolve, reject) => {
            const eventSource = new EventSource(
                `/api/chat/stream?message=${encodeURIComponent(message)}`
            );
            this.currentEventSource = eventSource;
            let fullText = '';
            let hasContent = false;

            let timeoutTimer = setTimeout(() => {
                eventSource.close();
                if (!hasContent) {
                    this.addError(botMsg, '请求超时，请稍后重试');
                }
                resolve();
            }, 60000);

            const resetTimeout = () => {
                clearTimeout(timeoutTimer);
                timeoutTimer = setTimeout(() => {
                    eventSource.close();
                    resolve();
                }, 60000);
            };

            // 工具调用事件
            eventSource.addEventListener('tool_call', (e) => {
                resetTimeout();
                try {
                    const data = JSON.parse(e.data);
                    this.toolCallCount++;
                    const toolLabel = this.getToolLabel(data.name);
                    const toolDetail = this.getToolDetail(data.name, data.args);
                    this.addThinkingStep(botMsg, toolLabel, toolDetail);
                } catch (err) {
                    console.error('解析工具调用失败:', err);
                }
            });

            // 文本消息事件
            eventSource.addEventListener('message', (e) => {
                resetTimeout();
                if (!hasContent) {
                    hasContent = true;
                    this.finishThinking(botMsg);
                    const contentDiv = botMsg.querySelector('.msg-content');
                    contentDiv.style.display = 'block';
                }
                fullText += e.data;
                this.renderContent(botMsg, fullText);
            });

            // 完成事件
            eventSource.addEventListener('done', () => {
                clearTimeout(timeoutTimer);
                eventSource.close();

                // 如果没有收到任何内容
                if (!hasContent) {
                    this.finishThinking(botMsg);
                    const contentDiv = botMsg.querySelector('.msg-content');
                    contentDiv.style.display = 'block';
                }

                // 最终渲染（完整版）
                if (fullText) {
                    this.renderContent(botMsg, fullText, true);
                }
                resolve();
            });

            // 错误事件
            eventSource.addEventListener('error', (e) => {
                clearTimeout(timeoutTimer);
                if (e.data) {
                    try {
                        const data = JSON.parse(e.data);
                        this.addError(botMsg, data.error || '处理请求时出错');
                    } catch {}
                }
                eventSource.close();
                resolve();
            });

            eventSource.onerror = () => {
                clearTimeout(timeoutTimer);
                if (!hasContent) {
                    this.addError(botMsg, '连接失败，请检查后端服务是否运行');
                }
                eventSource.close();
                resolve();
            };
        });
    }

    // ============================================================
    // 工具名称映射
    // ============================================================

    getToolLabel(name) {
        const labels = {
            'get_stock_info': '查询股票信息',
            'get_stock_history': '获取历史行情',
            'get_financial_statement': '分析财务报表',
            'get_stock_news': '搜索相关新闻',
            'get_recommendations': '查看分析师评级',
            'compare_stocks': '对比股票数据',
            'search_financial_news': '搜索财经资讯',
            'think': '分析推理中',
        };
        return labels[name] || `调用 ${name}`;
    }

    getToolDetail(name, args) {
        if (!args) return null;
        if (args.ticker) return `股票代码: ${args.ticker}`;
        if (args.tickers) return `对比: ${args.tickers}`;
        if (args.query) return `搜索: ${args.query}`;
        return null;
    }

    // ============================================================
    // Markdown 渲染器
    // ============================================================

    renderContent(botMsg, text, isFinal = false) {
        const contentDiv = botMsg.querySelector('.msg-content');
        const html = this.markdownToRichHtml(text, isFinal);
        contentDiv.innerHTML = html;
        this.scrollToBottom();
    }

    markdownToRichHtml(text, isFinal) {
        const blocks = this.parseBlocks(text);
        return this.renderBlocks(blocks, isFinal);
    }

    // ============================================================
    // Markdown 块级解析
    // ============================================================

    parseBlocks(text) {
        const blocks = [];
        const lines = text.split('\n');
        let i = 0;

        while (i < lines.length) {
            const line = lines[i];

            // 空行跳过
            if (!line.trim()) { i++; continue; }

            // H1
            if (line.startsWith('# ') && !line.startsWith('## ')) {
                blocks.push({ type: 'h1', content: line.slice(2).trim() });
                i++;
                continue;
            }

            // H2
            if (line.startsWith('## ')) {
                blocks.push({ type: 'h2', content: line.slice(3).trim() });
                i++;
                continue;
            }

            // H3
            if (line.startsWith('### ')) {
                blocks.push({ type: 'h3', content: line.slice(4).trim() });
                i++;
                continue;
            }

            // 分隔线
            if (/^---+$/.test(line.trim())) {
                blocks.push({ type: 'hr' });
                i++;
                continue;
            }

            // 引用块
            if (line.startsWith('> ')) {
                let quote = '';
                while (i < lines.length && lines[i].startsWith('> ')) {
                    quote += lines[i].slice(2) + '\n';
                    i++;
                }
                blocks.push({ type: 'blockquote', content: quote.trim() });
                continue;
            }

            // 表格
            if (line.includes('|') && line.trim().startsWith('|')) {
                const tableLines = [];
                while (i < lines.length && lines[i].includes('|') && lines[i].trim().startsWith('|')) {
                    tableLines.push(lines[i]);
                    i++;
                }
                const table = this.parseTable(tableLines);
                if (table) blocks.push({ type: 'table', ...table });
                continue;
            }

            // 无序列表
            if (/^[-*]\s/.test(line)) {
                const items = [];
                while (i < lines.length && /^[-*]\s/.test(lines[i])) {
                    items.push(lines[i].replace(/^[-*]\s+/, ''));
                    i++;
                }
                blocks.push({ type: 'ul', items });
                continue;
            }

            // 有序列表
            if (/^\d+[.)]\s/.test(line)) {
                const items = [];
                while (i < lines.length && /^\d+[.)]\s/.test(lines[i])) {
                    items.push(lines[i].replace(/^\d+[.)]\s+/, ''));
                    i++;
                }
                blocks.push({ type: 'ol', items });
                continue;
            }

            // 段落
            let para = '';
            while (i < lines.length && lines[i].trim() &&
                   !lines[i].startsWith('#') &&
                   !lines[i].startsWith('> ') &&
                   !(lines[i].includes('|') && lines[i].trim().startsWith('|')) &&
                   !/^[-*]\s/.test(lines[i]) &&
                   !/^\d+[.)]\s/.test(lines[i]) &&
                   !/^---+$/.test(lines[i].trim())) {
                para += lines[i] + '\n';
                i++;
            }
            if (para.trim()) {
                blocks.push({ type: 'paragraph', content: para.trim() });
            }
        }

        return blocks;
    }

    parseTable(lines) {
        if (lines.length < 2) return null;

        const parseLine = (line) =>
            line.split('|').map(c => c.trim()).filter(c => c !== '');

        const headers = parseLine(lines[0]);
        // 跳过分隔行
        const startRow = /^[\s|:-]+$/.test(lines[1]) ? 2 : 1;
        const rows = [];
        for (let i = startRow; i < lines.length; i++) {
            if (/^[\s|:-]+$/.test(lines[i])) continue;
            rows.push(parseLine(lines[i]));
        }

        return { headers, rows };
    }

    // ============================================================
    // 块级渲染
    // ============================================================

    renderBlocks(blocks, isFinal) {
        let html = '';
        let i = 0;

        while (i < blocks.length) {
            const block = blocks[i];

            switch (block.type) {
                case 'h1':
                    html += `<h1>${this.formatInline(block.content)}</h1>`;
                    break;

                case 'h2': {
                    // 将 H2 及其后续内容收集为一个卡片组
                    const cardContent = this.collectCardContent(blocks, i);
                    html += cardContent.html;
                    i = cardContent.nextIndex - 1;
                    break;
                }

                case 'h3':
                    html += `<h3>${this.formatInline(block.content)}</h3>`;
                    break;

                case 'blockquote':
                    html += this.renderBlockquote(block.content);
                    break;

                case 'table':
                    html += this.renderTable(block);
                    break;

                case 'ul':
                    html += `<ul>${block.items.map(item => `<li>${this.formatInline(item)}</li>`).join('')}</ul>`;
                    break;

                case 'ol':
                    html += `<ol>${block.items.map(item => `<li>${this.formatInline(item)}</li>`).join('')}</ol>`;
                    break;

                case 'hr':
                    html += '<hr>';
                    break;

                case 'paragraph':
                    html += `<p>${this.formatInline(block.content)}</p>`;
                    break;
            }
            i++;
        }

        return html;
    }

    // ============================================================
    // H2 卡片收集（将 H2 + 后续段落渲染为卡片）
    // ============================================================

    collectCardContent(blocks, startIndex) {
        const sections = [];
        let i = startIndex;

        // 收集连续的 H2 + 内容段
        while (i < blocks.length && blocks[i].type === 'h2') {
            const section = { title: blocks[i].content, body: [] };
            i++;
            while (i < blocks.length && blocks[i].type !== 'h2' && blocks[i].type !== 'h1') {
                section.body.push(blocks[i]);
                i++;
            }
            sections.push(section);
        }

        if (sections.length === 0) {
            return { html: '', nextIndex: startIndex + 1 };
        }

        // 检查是否是统计类内容（只有一个 section 且包含统计表格）
        if (sections.length === 1 && sections[0].body.some(b => b.type === 'table')) {
            const section = sections[0];
            const table = section.body.find(b => b.type === 'table');
            if (this.isStatsTable(table)) {
                return {
                    html: this.renderStatsCard(section.title, table, section.body),
                    nextIndex: i,
                };
            }
        }

        // 多个 H2 section → 分析卡片
        let html = '<div class="card card-analysis">';
        for (const section of sections) {
            const icon = this.extractEmoji(section.title);
            const titleText = section.title.replace(/^[\p{Emoji_Presentation}\p{Extended_Pictographic}]\s*/u, '');
            html += `<div class="analysis-section">`;
            html += `<div class="analysis-title">`;
            if (icon) html += `<span class="section-icon">${icon}</span>`;
            html += `<span>${this.formatInline(titleText)}</span></div>`;
            html += `<div class="analysis-body">`;
            for (const bodyBlock of section.body) {
                if (bodyBlock.type === 'paragraph') {
                    html += `<p>${this.formatInline(bodyBlock.content)}</p>`;
                } else if (bodyBlock.type === 'ul') {
                    html += `<ul>${bodyBlock.items.map(item => `<li>${this.formatInline(item)}</li>`).join('')}</ul>`;
                } else if (bodyBlock.type === 'ol') {
                    html += `<ol>${bodyBlock.items.map(item => `<li>${this.formatInline(item)}</li>`).join('')}</ol>`;
                } else if (bodyBlock.type === 'table') {
                    html += this.renderTable(bodyBlock);
                }
            }
            html += `</div></div>`;
        }
        html += '</div>';

        return { html, nextIndex: i };
    }

    // ============================================================
    // 特殊渲染：统计卡片
    // ============================================================

    isStatsTable(table) {
        if (!table || !table.rows) return false;
        // 统计表格通常有2列，且值列包含数字/百分比
        if (table.headers.length !== 2 && table.rows[0]?.length !== 2) return false;
        return table.rows.some(row => /[\d.]+%?/.test(row[0]) || /[\d.]+%?/.test(row[1]));
    }

    renderStatsCard(title, table, allBody) {
        const icon = this.extractEmoji(title);
        const titleText = title.replace(/^[\p{Emoji_Presentation}\p{Extended_Pictographic}]\s*/u, '');

        // 确定值列和标签列
        const stats = table.rows.map(row => {
            let value, label;
            if (/[\d.]+%?/.test(row[0])) {
                value = row[0];
                label = row[1] || table.headers[1];
            } else {
                label = row[0];
                value = row[1];
            }
            return { value, label };
        });

        const colors = ['red', 'red', 'green', 'blue'];

        let html = '<div class="card card-stats">';
        html += `<div class="card-header">`;
        if (icon) html += `<span class="card-header-icon">${icon}</span>`;
        html += `${this.formatInline(titleText)}</div>`;
        html += '<div class="stats-grid">';
        stats.forEach((stat, idx) => {
            const colorClass = colors[idx % colors.length];
            const val = stat.value.replace(/%/, '<span class="stat-unit">%</span>');
            html += `<div class="stat-item">
                <div class="stat-value ${colorClass}">${val}</div>
                <div class="stat-label">${this.escapeHtml(stat.label)}</div>
            </div>`;
        });
        html += '</div></div>';

        // 渲染非表格的其他内容
        for (const bodyBlock of allBody) {
            if (bodyBlock.type === 'table') continue;
            if (bodyBlock.type === 'paragraph') {
                html += `<p>${this.formatInline(bodyBlock.content)}</p>`;
            }
        }

        return html;
    }

    // ============================================================
    // 特殊渲染：引用块→结论卡片
    // ============================================================

    renderBlockquote(content) {
        // 检查是否包含粗体标题格式（作为结论卡片）
        if (content.includes('**')) {
            const titleMatch = content.match(/^\*\*(.+?)\*\*/);
            const title = titleMatch ? titleMatch[1] : '核心结论';
            const body = titleMatch ? content.slice(titleMatch[0].length).trim() : content;

            let html = '<div class="card card-conclusion">';
            html += `<div class="card-header"><span class="card-header-icon">💡</span>${this.escapeHtml(title)}</div>`;
            html += `<div class="card-body">${this.formatInline(body)}</div>`;
            html += '</div>';
            return html;
        }

        return `<blockquote>${this.formatInline(content)}</blockquote>`;
    }

    // ============================================================
    // 表格渲染
    // ============================================================

    renderTable(table) {
        if (!table.headers || !table.rows) return '';

        let html = '<div class="card card-table">';
        html += '<table>';

        // 表头
        html += '<thead><tr>';
        table.headers.forEach(h => {
            html += `<th>${this.escapeHtml(h)}</th>`;
        });
        html += '</tr></thead>';

        // 表身
        html += '<tbody>';
        table.rows.forEach(row => {
            html += '<tr>';
            row.forEach((cell, idx) => {
                let cellHtml = this.escapeHtml(cell);
                // 检测涨跌幅颜色
                if (/^[+-]?\d+\.?\d*%$/.test(cell.trim())) {
                    const isNeg = cell.trim().startsWith('-');
                    cellHtml = `<span class="${isNeg ? 'change-negative' : 'change-positive'}">${cellHtml}</span>`;
                }
                html += `<td>${cellHtml}</td>`;
            });
            html += '</tr>';
        });
        html += '</tbody></table></div>';

        return html;
    }

    // ============================================================
    // 行内格式化
    // ============================================================

    formatInline(text) {
        if (!text) return '';

        let s = this.escapeHtml(text);

        // 粗体
        s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        // 斜体
        s = s.replace(/\*(.+?)\*/g, '<em>$1</em>');
        // 行内代码
        s = s.replace(/`(.+?)`/g, '<code>$1</code>');
        // 链接
        s = s.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
        // 换行
        s = s.replace(/\n/g, '<br>');

        return s;
    }

    // ============================================================
    // 辅助函数
    // ============================================================

    extractEmoji(text) {
        const match = text.match(/^([\p{Emoji_Presentation}\p{Extended_Pictographic}])/u);
        return match ? match[1] : null;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// ============================================================
// 初始化
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    window.app = new FinanceChatApp();
});
