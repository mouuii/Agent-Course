"""
分析 Agent - 专注于数据分析
负责分析财务数据、股票指标和量化比较
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
import yfinance as yf
import json

# ============================================================
# 分析专用工具
# ============================================================


def _format_number(value):
    """格式化数字为可读形式"""
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
    """保留小数位"""
    if value is None:
        return "N/A"
    return round(value, n)


def _format_pct(value):
    """格式化百分比"""
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


@tool
def get_stock_info(ticker: str) -> str:
    """获取股票的基本信息和关键财务指标。

    专门用于：
    - 查看公司基本面
    - 获取估值指标（PE, PB等）
    - 了解财务健康度

    Args:
        ticker: 股票代码，如 AAPL, MSFT, 600519.SS
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # 提取核心分析指标
        result = {
            "基本信息": {
                "股票名称": info.get("longName") or info.get("shortName", "N/A"),
                "股票代码": ticker.upper(),
                "行业": info.get("industry", "N/A"),
                "板块": info.get("sector", "N/A"),
            },
            "价格指标": {
                "当前价格": info.get("currentPrice") or info.get("regularMarketPrice", "N/A"),
                "货币": info.get("currency", "N/A"),
                "52周最高": info.get("fiftyTwoWeekHigh", "N/A"),
                "52周最低": info.get("fiftyTwoWeekLow", "N/A"),
                "52周涨跌幅": _format_pct(
                    (info.get("currentPrice", 0) - info.get("fiftyTwoWeekLow", 0)) /
                    info.get("fiftyTwoWeekLow", 1)
                    if info.get("currentPrice") and info.get("fiftyTwoWeekLow")
                    else None
                ),
            },
            "估值指标": {
                "市值": _format_number(info.get("marketCap")),
                "市盈率(TTM)": _round(info.get("trailingPE")),
                "市盈率(前瞻)": _round(info.get("forwardPE")),
                "市净率": _round(info.get("priceToBook")),
                "每股收益(TTM)": _round(info.get("trailingEps")),
                "股息率": _format_pct(info.get("dividendYield")),
            },
            "财务指标": {
                "总营收": _format_number(info.get("totalRevenue")),
                "营收增长": _format_pct(info.get("revenueGrowth")),
                "利润率": _format_pct(info.get("profitMargins")),
                "ROE": _format_pct(info.get("returnOnEquity")),
                "ROA": _format_pct(info.get("returnOnAssets")),
            },
            "风险指标": {
                "Beta": _round(info.get("beta")),
                "负债率": _format_pct(info.get("debtToEquity")),
            },
        }

        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"获取 {ticker} 信息时出错: {str(e)}"


@tool
def get_financial_statement(ticker: str, statement_type: str = "income") -> str:
    """获取并分析公司财务报表。

    专门用于：
    - 深度财务分析
    - 趋势对比
    - 财务健康度评估

    Args:
        ticker: 股票代码
        statement_type: 报表类型
            - income: 利润表（收入、利润、费用）
            - balance: 资产负债表（资产、负债、权益）
            - cashflow: 现金流量表（经营、投资、融资）
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
            return f"不支持的报表类型: {statement_type}"

        if df is None or df.empty:
            return f"未找到 {ticker} 的{title}数据"

        # 取最近2期数据
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
def compare_stocks(tickers: str) -> str:
    """对比多只股票的关键指标。

    专门用于：
    - 横向对比分析
    - 行业内比较
    - 投资选择评估

    Args:
        tickers: 逗号分隔的股票代码，如 "AAPL,MSFT,GOOGL"
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
                "市净率": _round(info.get("priceToBook")),
                "股息率": _format_pct(info.get("dividendYield")),
                "营收增长": _format_pct(info.get("revenueGrowth")),
                "利润率": _format_pct(info.get("profitMargins")),
                "ROE": _format_pct(info.get("returnOnEquity")),
                "Beta": _round(info.get("beta")),
                "52周涨幅": _format_pct(
                    (info.get("currentPrice", 0) / info.get("fiftyTwoWeekLow", 1) - 1)
                    if info.get("currentPrice") and info.get("fiftyTwoWeekLow")
                    else None
                ),
            })

        return json.dumps(comparison, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"对比股票时出错: {str(e)}"


@tool
def get_stock_history(ticker: str, period: str = "3mo") -> str:
    """获取股票历史价格数据和统计分析。

    专门用于：
    - 价格趋势分析
    - 波动率计算
    - 交易量分析

    Args:
        ticker: 股票代码
        period: 时间范围 (1mo, 3mo, 6mo, 1y, 2y)
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            return f"未找到 {ticker} 的历史数据"

        # 统计分析
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
            "平均价格": round(hist['Close'].mean(), 2),
            "价格标准差": round(hist['Close'].std(), 2),
            "平均成交量": int(hist['Volume'].mean()),
        }

        return json.dumps(summary, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"获取 {ticker} 历史数据时出错: {str(e)}"


# ============================================================
# 创建分析 Agent
# ============================================================

# 分析 Agent 专用提示词
ANALYSIS_SYSTEM_PROMPT = """你是一位专业的财务分析师，专注于数据分析和量化评估。

**你的核心职责：**
1. 分析公司财务报表和关键指标
2. 评估股票估值水平
3. 对比分析多只股票
4. 计算和解读财务比率
5. 评估投资价值和风险

**工作方式：**
- 使用 get_stock_info 获取核心财务指标
- 使用 get_financial_statement 深度分析财报
- 使用 compare_stocks 进行横向对比
- 使用 get_stock_history 分析价格趋势
- 注重数据的准确性和分析的逻辑性

**分析框架：**
1. **估值分析**：PE、PB、PEG等指标评估
2. **盈利能力**：利润率、ROE、ROA分析
3. **成长性**：营收增长、利润增长趋势
4. **财务健康**：负债率、现金流状况
5. **风险评估**：Beta、波动率等指标

**输出格式：**
- 使用数据支撑观点
- 提供量化对比
- 给出明确的分析结论
- 指出关键风险点
- 用中文回答

**重要：**
你专注于数据分析，不收集新闻信息。
市场动态和新闻由研究 Agent 负责。
"""


def create_analysis_agent(llm: ChatOpenAI):
    """创建分析 Agent

    Args:
        llm: 语言模型实例

    Returns:
        配置好的分析 Agent
    """
    analysis_tools = [
        get_stock_info,
        get_financial_statement,
        compare_stocks,
        get_stock_history,
    ]

    return create_agent(
        llm,
        tools=analysis_tools,
        system_prompt=ANALYSIS_SYSTEM_PROMPT,
    )


def run_analysis(llm: ChatOpenAI, query: str) -> str:
    """运行分析任务

    Args:
        llm: 语言模型实例
        query: 分析查询

    Returns:
        分析结果
    """
    agent = create_analysis_agent(llm)
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content
