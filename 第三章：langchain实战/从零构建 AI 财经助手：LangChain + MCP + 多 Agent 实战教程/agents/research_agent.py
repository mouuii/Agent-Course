"""
研究 Agent - 专注于信息搜集
负责收集市场新闻、公司动态等外部信息
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
import yfinance as yf
from duckduckgo_search import DDGS
import json

# ============================================================
# 研究专用工具
# ============================================================


@tool
def search_financial_news(query: str) -> str:
    """搜索财经新闻和市场信息。

    专门用于：
    - 搜索最新市场动态
    - 查找行业新闻
    - 获取公司相关新闻报道

    Args:
        query: 搜索关键词，如"苹果公司财报"、"科技股行情"
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=6))

        if not results:
            return f"未找到关于 '{query}' 的搜索结果"

        news_list = []
        for r in results:
            news_list.append({
                "标题": r.get("title", ""),
                "摘要": r.get("body", "")[:200],
                "来源": r.get("href", ""),
            })

        return json.dumps(news_list, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"搜索时出错: {str(e)}"


@tool
def get_stock_news(ticker: str) -> str:
    """获取特定股票的最新新闻。

    专门用于：
    - 获取公司官方新闻
    - 查看媒体报道
    - 了解最新动态

    Args:
        ticker: 股票代码，如 AAPL, MSFT, 600519.SS
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return f"未找到 {ticker} 的相关新闻"

        news_list = []
        for item in news[:8]:
            content = item.get("content", {})
            news_list.append({
                "标题": content.get("title", item.get("title", "N/A")),
                "发布者": content.get("provider", {}).get("displayName", "N/A"),
                "链接": content.get("canonicalUrl", {}).get("url")
                        or item.get("link", "N/A"),
            })

        return json.dumps(news_list, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"获取 {ticker} 新闻时出错: {str(e)}"


@tool
def get_market_sentiment(ticker: str) -> str:
    """获取市场情绪指标，包括分析师推荐。

    专门用于：
    - 获取分析师评级
    - 查看目标价格
    - 了解市场共识

    Args:
        ticker: 股票代码
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # 获取推荐摘要
        sentiment = {
            "股票代码": ticker.upper(),
            "推荐评级": info.get("recommendationKey", "N/A"),
            "目标均价": info.get("targetMeanPrice", "N/A"),
            "目标最高价": info.get("targetHighPrice", "N/A"),
            "目标最低价": info.get("targetLowPrice", "N/A"),
            "分析师数量": info.get("numberOfAnalystOpinions", "N/A"),
            "当前价格": info.get("currentPrice") or info.get("regularMarketPrice", "N/A"),
        }

        # 计算潜在涨幅
        if sentiment["目标均价"] != "N/A" and sentiment["当前价格"] != "N/A":
            upside = (sentiment["目标均价"] / sentiment["当前价格"] - 1) * 100
            sentiment["目标涨幅"] = f"{upside:.2f}%"

        return json.dumps(sentiment, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"获取 {ticker} 市场情绪时出错: {str(e)}"


# ============================================================
# 创建研究 Agent
# ============================================================

# 研究 Agent 专用提示词
RESEARCH_SYSTEM_PROMPT = """你是一位专业的财经研究员，专注于信息收集和市场动态研究。

**你的核心职责：**
1. 收集股票和公司的最新新闻
2. 搜索行业动态和市场趋势
3. 获取分析师评级和市场情绪
4. 整理外部信息为后续分析提供支持

**工作方式：**
- 使用 search_financial_news 搜索广泛的市场信息
- 使用 get_stock_news 获取特定公司的新闻
- 使用 get_market_sentiment 了解分析师观点
- 注重信息的时效性和可靠性
- 提供信息来源和发布时间

**输出格式：**
- 清晰列出收集到的信息
- 标注信息来源
- 突出关键发现
- 用中文整理汇报

**重要：**
你专注于信息收集，不做深度财务分析。
财务数据分析由分析 Agent 负责。
"""


def create_research_agent(llm: ChatOpenAI):
    """创建研究 Agent

    Args:
        llm: 语言模型实例

    Returns:
        配置好的研究 Agent
    """
    research_tools = [
        search_financial_news,
        get_stock_news,
        get_market_sentiment,
    ]

    return create_agent(
        llm,
        tools=research_tools,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
    )


def run_research(llm: ChatOpenAI, query: str) -> str:
    """运行研究任务

    Args:
        llm: 语言模型实例
        query: 研究查询

    Returns:
        研究结果
    """
    agent = create_research_agent(llm)
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content
