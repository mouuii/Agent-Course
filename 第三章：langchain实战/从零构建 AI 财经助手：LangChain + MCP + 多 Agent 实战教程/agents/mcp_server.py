#!/usr/bin/env python3
"""
MCP Server for Finance Agent
将财经工具通过 Model Context Protocol 暴露给外部应用使用
"""

import asyncio
import json
import sys
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from mcp.server import Server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from mcp.server.stdio import stdio_server

# 导入财经工具
import yfinance as yf
from duckduckgo_search import DDGS

# 线程池，用于运行同步的 yfinance 调用，避免阻塞事件循环
_executor = ThreadPoolExecutor(max_workers=4)


# ============================================================
# 工具实现函数
# ============================================================


def get_stock_info_impl(ticker: str) -> dict[str, Any]:
    """获取股票基本信息"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

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
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_stock_history_impl(ticker: str, period: str = "1mo") -> dict[str, Any]:
    """获取股票历史价格数据"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            return {"success": False, "error": f"未找到 {ticker} 的历史数据"}

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
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_financial_news_impl(query: str, max_results: int = 6) -> dict[str, Any]:
    """搜索财经新闻"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return {"success": False, "error": f"未找到关于 '{query}' 的搜索结果"}

        news_list = []
        for r in results:
            news_list.append({
                "标题": r.get("title", ""),
                "摘要": r.get("body", "")[:200],
                "来源": r.get("href", ""),
            })

        return {"success": True, "data": news_list}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_stock_news_impl(ticker: str) -> dict[str, Any]:
    """获取股票相关新闻"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return {"success": False, "error": f"未找到 {ticker} 的相关新闻"}

        news_list = []
        for item in news[:8]:
            content = item.get("content", {})
            news_list.append({
                "标题": content.get("title", item.get("title", "N/A")),
                "发布者": content.get("provider", {}).get("displayName", "N/A"),
                "链接": content.get("canonicalUrl", {}).get("url")
                        or item.get("link", "N/A"),
            })

        return {"success": True, "data": news_list}
    except Exception as e:
        return {"success": False, "error": str(e)}


def compare_stocks_impl(tickers: str) -> dict[str, Any]:
    """对比多只股票"""
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
            })

        return {"success": True, "data": comparison}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_recommendations_impl(ticker: str) -> dict[str, Any]:
    """获取分析师推荐"""
    try:
        stock = yf.Ticker(ticker)

        # 获取推荐摘要
        info = stock.info
        rec_summary = {
            "推荐评级": info.get("recommendationKey", "N/A"),
            "目标均价": info.get("targetMeanPrice", "N/A"),
            "目标最高价": info.get("targetHighPrice", "N/A"),
            "目标最低价": info.get("targetLowPrice", "N/A"),
            "分析师数量": info.get("numberOfAnalystOpinions", "N/A"),
        }

        # 获取推荐记录
        rec = stock.recommendations
        rec_data = []
        if rec is not None and not rec.empty:
            recent_rec = rec.tail(10)
            for date, row in recent_rec.iterrows():
                entry = {}
                for col in recent_rec.columns:
                    entry[col] = str(row[col])
                entry["日期"] = str(date)
                rec_data.append(entry)

        result = {
            "股票代码": ticker.upper(),
            "推荐摘要": rec_summary,
            "最近推荐记录": rec_data if rec_data else "暂无分析师推荐数据",
        }
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
# MCP Server 实现
# ============================================================


async def serve() -> None:
    """启动 MCP Server"""

    server = Server("finance-agent-mcp")

    # ========================================
    # 注册工具列表
    # ========================================

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """返回可用工具列表"""
        return [
            Tool(
                name="get_stock_info",
                description="获取股票的基本信息，包括当前价格、市值、市盈率等关键指标",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "股票代码，如 AAPL, MSFT, 600519.SS (贵州茅台), 000001.SZ (平安银行)",
                        }
                    },
                    "required": ["ticker"],
                },
            ),
            Tool(
                name="get_stock_history",
                description="获取股票的历史价格数据，包括汇总统计和近期交易数据",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "股票代码",
                        },
                        "period": {
                            "type": "string",
                            "description": "时间范围，可选值: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max",
                            "default": "1mo",
                        },
                    },
                    "required": ["ticker"],
                },
            ),
            Tool(
                name="search_financial_news",
                description="搜索财经新闻和信息，返回相关新闻列表",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词，如'苹果公司财报'、'A股市场行情'",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "返回结果数量",
                            "default": 6,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_stock_news",
                description="获取与特定股票相关的最新新闻",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "股票代码",
                        }
                    },
                    "required": ["ticker"],
                },
            ),
            Tool(
                name="compare_stocks",
                description="对比多只股票的关键指标，帮助分析投资价值",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "string",
                            "description": "逗号分隔的股票代码列表，如 'AAPL,MSFT,GOOGL'",
                        }
                    },
                    "required": ["tickers"],
                },
            ),
            Tool(
                name="get_recommendations",
                description="获取分析师对股票的评级和推荐",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "股票代码",
                        }
                    },
                    "required": ["ticker"],
                },
            ),
        ]

    # ========================================
    # 处理工具调用
    # ========================================

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
        """执行工具调用（在线程池中运行同步函数，避免阻塞事件循环）"""

        loop = asyncio.get_event_loop()
        timeout = 30  # 超时时间（秒）

        # 根据工具名称构建对应的同步调用
        if name == "get_stock_info":
            func = partial(get_stock_info_impl, arguments["ticker"])
        elif name == "get_stock_history":
            period = arguments.get("period", "1mo")
            func = partial(get_stock_history_impl, arguments["ticker"], period)
        elif name == "search_financial_news":
            max_results = arguments.get("max_results", 6)
            func = partial(search_financial_news_impl, arguments["query"], max_results)
        elif name == "get_stock_news":
            func = partial(get_stock_news_impl, arguments["ticker"])
        elif name == "compare_stocks":
            func = partial(compare_stocks_impl, arguments["tickers"])
        elif name == "get_recommendations":
            func = partial(get_recommendations_impl, arguments["ticker"])
        else:
            result = {"success": False, "error": f"未知工具: {name}"}
            return [
                TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, indent=2),
                )
            ]

        # 在线程池中执行同步函数，并设置超时
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(_executor, func),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            result = {"success": False, "error": f"工具 {name} 执行超时（{timeout}秒），请稍后重试"}
        except Exception as e:
            result = {"success": False, "error": f"工具 {name} 执行异常: {str(e)}"}

        # 返回结果
        return [
            TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2),
            )
        ]

    # ========================================
    # 启动服务器
    # ========================================

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)


def main():
    """主入口"""
    asyncio.run(serve())


if __name__ == "__main__":
    main()
