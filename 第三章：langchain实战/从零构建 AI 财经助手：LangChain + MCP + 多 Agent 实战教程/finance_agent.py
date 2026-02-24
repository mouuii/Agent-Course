"""
财经类 Agent 程序
使用智谱 GLM 模型 + LangChain + LangGraph 构建
支持股票查询、财务分析、新闻搜索等功能
"""

import warnings
warnings.filterwarnings("ignore")

import json
from datetime import datetime, timedelta

import yfinance as yf
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent

# ============================================================
# 智谱 GLM 模型配置
# ============================================================

ZHIPU_API_KEY = "ccb9818e987149dd8cc2541ff7c9df57.AMe31g7tJHwgPo9q"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL_NAME = "glm-5"

llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=ZHIPU_API_KEY,
    openai_api_base=ZHIPU_BASE_URL,
    temperature=0.3,
)

# ============================================================
# 工具定义
# ============================================================


@tool
def get_stock_info(ticker: str) -> str:
    """获取股票的基本信息，包括当前价格、市值、市盈率等关键指标。

    Args:
        ticker: 股票代码，如 AAPL, MSFT, 600519.SS (贵州茅台), 000001.SZ (平安银行)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # 提取关键信息
        result = {
            "股票名称": info.get("longName") or info.get("shortName", "N/A"),
            "股票代码": ticker.upper(),
            "当前价格": info.get("currentPrice") or info.get("regularMarketPrice", "N/A"),
            "货币": info.get("currency", "N/A"),
            "前收盘价": info.get("previousClose", "N/A"),
            "开盘价": info.get("open", "N/A"),
            "日最高价": info.get("dayHigh", "N/A"),
            "日最低价": info.get("dayLow", "N/A"),
            "52周最高": info.get("fiftyTwoWeekHigh", "N/A"),
            "52周最低": info.get("fiftyTwoWeekLow", "N/A"),
            "市值": _format_number(info.get("marketCap")),
            "市盈率(TTM)": _round(info.get("trailingPE")),
            "市盈率(前瞻)": _round(info.get("forwardPE")),
            "每股收益(TTM)": _round(info.get("trailingEps")),
            "股息率": _format_pct(info.get("dividendYield")),
            "Beta": _round(info.get("beta")),
            "总营收": _format_number(info.get("totalRevenue")),
            "利润率": _format_pct(info.get("profitMargins")),
            "行业": info.get("industry", "N/A"),
            "板块": info.get("sector", "N/A"),
            "公司简介": (info.get("longBusinessSummary") or "N/A")[:200],
        }

        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"获取 {ticker} 信息时出错: {str(e)}"


@tool
def get_stock_history(ticker: str, period: str = "1mo") -> str:
    """获取股票的历史价格数据。

    Args:
        ticker: 股票代码
        period: 时间范围，可选值: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            return f"未找到 {ticker} 的历史数据"

        # 汇总统计
        summary = {
            "股票代码": ticker.upper(),
            "查询周期": period,
            "数据起始": str(hist.index[0].date()),
            "数据截止": str(hist.index[-1].date()),
            "起始价格": round(hist['Close'].iloc[0], 2),
            "最新价格": round(hist['Close'].iloc[-1], 2),
            "期间涨跌": f"{round((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100, 2)}%",
            "期间最高": round(hist['High'].max(), 2),
            "期间最低": round(hist['Low'].min(), 2),
            "平均成交量": int(hist['Volume'].mean()),
        }

        # 最近5个交易日数据
        recent = hist.tail(5)
        recent_data = []
        for date, row in recent.iterrows():
            recent_data.append({
                "日期": str(date.date()),
                "收盘价": round(row['Close'], 2),
                "最高价": round(row['High'], 2),
                "最低价": round(row['Low'], 2),
                "成交量": int(row['Volume']),
            })

        result = {
            "汇总": summary,
            "近期交易数据": recent_data,
        }

        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"获取 {ticker} 历史数据时出错: {str(e)}"


@tool
def get_financial_statement(ticker: str, statement_type: str = "income") -> str:
    """获取公司财务报表数据。

    Args:
        ticker: 股票代码
        statement_type: 报表类型，可选值:
            - income: 利润表
            - balance: 资产负债表
            - cashflow: 现金流量表
    """
    try:
        stock = yf.Ticker(ticker)

        if statement_type == "income":
            df = stock.financials
            title = "利润表"
        elif statement_type == "balance":
            df = stock.balance_sheet
            title = "资产负债表"
        elif statement_type == "cashflow":
            df = stock.cashflow
            title = "现金流量表"
        else:
            return f"不支持的报表类型: {statement_type}，请使用 income/balance/cashflow"

        if df is None or df.empty:
            return f"未找到 {ticker} 的{title}数据"

        # 取最近2期数据进行对比
        result = {"股票代码": ticker.upper(), "报表类型": title, "数据": {}}
        for col in df.columns[:2]:
            period_data = {}
            for idx in df.index:
                val = df.loc[idx, col]
                if val is not None and str(val) != "nan":
                    period_data[str(idx)] = _format_number(float(val))
            result["数据"][str(col.date())] = period_data

        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"获取 {ticker} 财务报表时出错: {str(e)}"


@tool
def get_stock_news(ticker: str) -> str:
    """获取与股票相关的最新新闻。

    Args:
        ticker: 股票代码
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
def get_recommendations(ticker: str) -> str:
    """获取分析师对股票的评级和推荐。

    Args:
        ticker: 股票代码
    """
    try:
        stock = yf.Ticker(ticker)

        # 获取推荐
        rec = stock.recommendations
        if rec is not None and not rec.empty:
            recent_rec = rec.tail(10)
            rec_data = []
            for date, row in recent_rec.iterrows():
                entry = {}
                for col in recent_rec.columns:
                    entry[col] = str(row[col])
                entry["日期"] = str(date)
                rec_data.append(entry)
        else:
            rec_data = "暂无分析师推荐数据"

        # 获取推荐摘要
        info = stock.info
        rec_summary = {
            "推荐评级": info.get("recommendationKey", "N/A"),
            "目标均价": info.get("targetMeanPrice", "N/A"),
            "目标最高价": info.get("targetHighPrice", "N/A"),
            "目标最低价": info.get("targetLowPrice", "N/A"),
            "分析师数量": info.get("numberOfAnalystOpinions", "N/A"),
        }

        result = {
            "股票代码": ticker.upper(),
            "推荐摘要": rec_summary,
            "最近推荐记录": rec_data,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"获取 {ticker} 推荐数据时出错: {str(e)}"


@tool
def compare_stocks(tickers: str) -> str:
    """对比多只股票的关键指标。

    Args:
        tickers: 逗号分隔的股票代码列表，如 "AAPL,MSFT,GOOGL"
    """
    try:
        ticker_list = [t.strip() for t in tickers.split(",")]
        comparison = []

        for ticker in ticker_list:
            stock = yf.Ticker(ticker)
            info = stock.info
            comparison.append({
                "股票代码": ticker.upper(),
                "名称": info.get("shortName", "N/A"),
                "当前价格": info.get("currentPrice") or info.get("regularMarketPrice", "N/A"),
                "市值": _format_number(info.get("marketCap")),
                "市盈率(TTM)": _round(info.get("trailingPE")),
                "市盈率(前瞻)": _round(info.get("forwardPE")),
                "每股收益": _round(info.get("trailingEps")),
                "股息率": _format_pct(info.get("dividendYield")),
                "营收增长": _format_pct(info.get("revenueGrowth")),
                "利润率": _format_pct(info.get("profitMargins")),
                "Beta": _round(info.get("beta")),
                "52周涨跌": _format_pct(
                    (info.get("currentPrice", 0) / info.get("fiftyTwoWeekLow", 1) - 1)
                    if info.get("currentPrice") and info.get("fiftyTwoWeekLow")
                    else None
                ),
            })

        return json.dumps(comparison, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"对比股票时出错: {str(e)}"


@tool
def search_financial_news(query: str) -> str:
    """搜索财经新闻和信息。

    Args:
        query: 搜索关键词，如"苹果公司财报"、"A股市场行情"
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
def think(reflection: str) -> str:
    """用于在研究过程中进行思考和反思的工具。

    在分析股票或做出投资建议前，使用此工具来：
    1. 整理已收集到的信息
    2. 评估数据的完整性
    3. 规划下一步分析方向
    4. 形成初步的分析结论

    Args:
        reflection: 详细的思考和分析内容
    """
    return f"思考已记录: {reflection}"


# ============================================================
# 辅助函数
# ============================================================


def _format_number(value):
    if value is None:
        return "N/A"
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}{abs_val / 1e12:.2f}万亿"
    elif abs_val >= 1e8:
        return f"{sign}{abs_val / 1e8:.2f}亿"
    elif abs_val >= 1e4:
        return f"{sign}{abs_val / 1e4:.2f}万"
    else:
        return f"{sign}{abs_val:.2f}"


def _round(value, n=2):
    if value is None:
        return "N/A"
    return round(value, n)


def _format_pct(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


# ============================================================
# 系统提示词
# ============================================================

SYSTEM_PROMPT = """你是一位专业的财经研究分析师，擅长股票分析、财务数据解读和投资研究。

**你的职责：**
1. 分析股票的价格走势和基本面
2. 研究公司财务报表和关键指标
3. 搜索并整理相关财经新闻
4. 对比分析多只股票的投资价值
5. 提供基于数据的客观分析

**分析框架：**
- 首先获取股票基本信息和价格数据
- 分析关键财务指标：市盈率、市值、营收、利润率等
- 查看分析师评级和目标价
- 关注最新新闻和市场动态
- 综合以上信息给出分析结论

**工具使用策略：**
- 对于股票查询，先用 get_stock_info 获取概览
- 需要历史数据时用 get_stock_history
- 需要财务报表时用 get_financial_statement
- 需要新闻时用 get_stock_news 或 search_financial_news
- 对比股票时用 compare_stocks
- 分析师评级用 get_recommendations
- 复杂分析前先用 think 工具梳理思路

**输出格式（非常重要，请严格遵守）：**

你的回复会被前端渲染为富文本卡片，请使用以下 Markdown 格式以获得最佳展示效果：

1. **开头**：用1-2句话概括回答方向（普通段落）

2. **核心结论**：用引用块(>)包裹，首行加粗作为标题：
> **核心结论**
> **重要结论加粗显示。**补充说明文字。

3. **关键数据**：用"## 📊 标题"加两列表格展示，会自动渲染为大字统计卡片：
## 📊 关键数据
| 数值 | 说明 |
|------|------|
| 78% | 某指标 |
| 3.3% | 某指标 |

4. **分析要点**：用多个连续"## emoji 标题"，会自动合并为分析卡片：
## 📈 要点一标题
分析内容...
## 💰 要点二标题
分析内容...
## 🎯 要点三标题
分析内容...

5. **对比数据**：用多列表格，涨跌幅数据会自动着色：
| 名称 | 市盈率 | 涨跌幅 |
|------|--------|--------|
| 某股 | 12.5 | -1.5% |

6. **风险提示**：在文末用普通段落说明风险

**格式注意事项：**
- 每个分析要点的 ## 标题前请加一个 emoji（如 📈💰🎯📊🔍💡🏢📰）
- 关键数字用 **加粗** 标记
- 保持段落简洁，每段不超过3句话
- 对比表格的涨跌幅请带正负号和百分号（如 +2.5% 或 -1.3%）

**重要规则：**
- 只回答与财经和股票市场相关的问题
- 非财经类问题请礼貌拒绝："抱歉，我只能协助处理股票市场和财经分析相关的问题。请向我询问股票、公司或财务指标方面的内容。"
- 所有分析必须基于数据，不做无依据的预测
- 投资建议需附带风险提示
- 用中文回答
"""

# ============================================================
# 创建 Agent
# ============================================================

tools = [
    get_stock_info,
    get_stock_history,
    get_financial_statement,
    get_stock_news,
    get_recommendations,
    compare_stocks,
    search_financial_news,
    think,
]

agent = create_agent(
    llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# 交互式运行
# ============================================================


def run_agent(query: str) -> str:
    """运行 agent 并返回最终回答"""
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


def stream_agent(query: str):
    """流式运行 agent，实时打印输出"""
    events = agent.stream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="messages",
    )

    for msg, metadata in events:
        # 打印工具调用信息
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\n  [调用工具] {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")
        # 打印最终文本
        elif hasattr(msg, "content") and msg.content and metadata.get("langgraph_node") == "agent":
            print(msg.content, end="", flush=True)

    print()  # 最终换行


def main():
    print("=" * 60)
    print("  财经研究 Agent")
    print("  模型: 智谱 GLM-4-Flash")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    print()

    while True:
        try:
            query = input("📊 请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        print()
        try:
            stream_agent(query)
        except Exception as e:
            print(f"\n处理出错: {e}")
        print()


if __name__ == "__main__":
    main()
