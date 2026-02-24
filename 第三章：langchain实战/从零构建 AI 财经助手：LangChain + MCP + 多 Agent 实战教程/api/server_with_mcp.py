"""
FastAPI 服务器 - 使用 MCP Client 连接本地 MCP Server
演示完整的 MCP 工作流程
"""

import json
import asyncio
from typing import AsyncGenerator
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# MCP Client 相关
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 导入配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from finance_agent import ZHIPU_API_KEY, ZHIPU_BASE_URL, MODEL_NAME, SYSTEM_PROMPT


# ============================================================
# 全局变量
# ============================================================

mcp_client = None
agent = None
tools = []


# ============================================================
# FastAPI 应用初始化
# ============================================================

app = FastAPI(
    title="财经研究 Agent API (MCP 模式)",
    description="基于 MCP Client + LangGraph 的智能财经分析助手",
    version="2.0.0-mcp",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MCP Client 初始化
# ============================================================

@app.on_event("startup")
async def startup_event():
    """
    应用启动时初始化 MCP Client
    """
    global mcp_client, agent, tools

    print("🚀 启动 MCP Client...")

    # MCP Server 脚本路径
    mcp_server_path = str(Path(__file__).parent.parent / "agents" / "mcp_server.py")

    # 获取虚拟环境的 Python 路径
    venv_python = str(Path(__file__).parent.parent / ".venv" / "bin" / "python")

    # 创建 MCP 客户端（连接本地 MCP Server）
    mcp_client = MultiServerMCPClient({
        "finance": {
            "transport": "stdio",
            "command": venv_python,
            "args": [mcp_server_path],
        },
    })

    try:
        # 获取 MCP 工具
        print("📡 连接 MCP Server...")
        tools = await mcp_client.get_tools()
        print(f"✅ 成功获取 {len(tools)} 个工具：")
        for tool in tools:
            print(f"   - {tool.name}")

        # 创建 LLM
        llm = ChatOpenAI(
            model=MODEL_NAME,
            openai_api_key=ZHIPU_API_KEY,
            openai_api_base=ZHIPU_BASE_URL,
            temperature=0.3,
        )

        # 创建 Agent（使用 MCP 提供的工具）
        agent = create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)
        print("✅ Agent 创建成功（使用 MCP 工具）")

    except Exception as e:
        print(f"❌ MCP Client 初始化失败: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    应用关闭时清理 MCP Client
    """
    global mcp_client
    if mcp_client:
        print("🔄 关闭 MCP Client...")
        try:
            # MCP Client 会自动清理，不需要手动调用
            pass
        except Exception as e:
            print(f"⚠️ 清理时出错: {e}")
        print("✅ MCP Client 已关闭")


# ============================================================
# 请求/响应模型
# ============================================================

class ChatRequest(BaseModel):
    """对话请求模型"""
    message: str = Field(..., description="用户输入的问题")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "查询苹果公司的股票信息"
            }
        }


class ChatResponse(BaseModel):
    """对话响应模型"""
    response: str = Field(..., description="Agent 的回复")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "苹果公司（AAPL）当前股价..."
            }
        }


class ToolInfo(BaseModel):
    """工具信息模型"""
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    args_schema: dict = Field(..., description="参数结构")


# ============================================================
# API 端点
# ============================================================

@app.get("/api/health", tags=["系统"])
async def health_check():
    """
    健康检查端点
    """
    return {
        "status": "healthy",
        "service": "finance-agent-api-mcp",
        "version": "2.0.0-mcp",
        "mcp_enabled": mcp_client is not None,
        "tools_count": len(tools),
    }


@app.get("/api/tools", response_model=list[ToolInfo], tags=["工具"])
async def list_tools():
    """
    列出所有可用工具（来自 MCP Server）
    """
    tool_list = []
    for tool in tools:
        tool_list.append({
            "name": tool.name,
            "description": tool.description,
            "args_schema": tool.args_schema if hasattr(tool, "args_schema") else {},
        })
    return tool_list


@app.post("/api/chat", response_model=ChatResponse, tags=["对话"])
async def chat(request: ChatRequest):
    """
    非流式对话端点
    """
    if not agent:
        raise HTTPException(status_code=500, detail="Agent 未初始化")

    try:
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=request.message)]
        })
        response = result["messages"][-1].content
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@app.get("/api/chat/stream", tags=["对话"])
async def stream_chat_get(message: str):
    """
    SSE 流式对话端点（GET 方式，用于 EventSource）
    """
    if not agent:
        raise HTTPException(status_code=500, detail="Agent 未初始化")

    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            # 使用 agent.astream 异步流式输出（MCP 工具需要异步）
            events = agent.astream(
                {"messages": [HumanMessage(content=message)]},
                stream_mode="messages",
            )

            async for msg, metadata in events:
                # 处理工具调用事件
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        yield {
                            "event": "tool_call",
                            "data": json.dumps({
                                "name": tc["name"],
                                "args": tc["args"],
                            }, ensure_ascii=False)
                        }

                # 处理 agent 输出的文本（节点名为 "model"）
                elif hasattr(msg, "content") and msg.content and metadata.get("langgraph_node") == "model":
                    yield {
                        "event": "message",
                        "data": msg.content
                    }

            # 发送完成标志
            yield {
                "event": "done",
                "data": "[DONE]"
            }

        except Exception as e:
            # 发送错误信息
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}, ensure_ascii=False)
            }

    return EventSourceResponse(event_generator())


# ============================================================
# 静态文件服务
# ============================================================

static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", response_class=HTMLResponse, tags=["前端"])
async def root():
    """
    根路径 - 返回前端 HTML 页面
    """
    index_file = static_path / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    else:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>财经研究 Agent API (MCP 模式)</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .success { color: green; }
                .info { color: blue; }
            </style>
        </head>
        <body>
            <h1>💡 财经研究 Agent API (MCP 模式)</h1>
            <p class="success">✅ 服务运行中</p>
            <p class="info">🔌 使用 MCP Client 连接本地 MCP Server</p>
            <ul>
                <li><a href="/docs">API 文档（Swagger UI）</a></li>
                <li><a href="/redoc">API 文档（ReDoc）</a></li>
                <li><a href="/api/health">健康检查</a></li>
                <li><a href="/api/tools">查看工具列表</a></li>
            </ul>
            <h3>架构说明</h3>
            <pre>
浏览器 → FastAPI (MCP Client)
           ↓
       MCP Protocol (stdio)
           ↓
       MCP Server (agents/mcp_server.py)
           ↓
       财经工具 (yfinance, DuckDuckGo)
            </pre>
        </body>
        </html>
        """


# ============================================================
# 启动配置
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("""
╔══════════════════════════════════════════════════════════╗
║     Finance Agent - MCP 模式                              ║
║  Web 服务作为 MCP Client 连接本地 MCP Server              ║
╚══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "server_with_mcp:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
