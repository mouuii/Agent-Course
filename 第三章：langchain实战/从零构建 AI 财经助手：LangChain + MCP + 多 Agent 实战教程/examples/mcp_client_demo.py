"""
MCP Client 示例
演示如何在 LangChain Agent 中作为 Client 调用 MCP Server
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI

async def demo_mcp_client():
    """
    演示作为 MCP Client 使用

    注意：需要先安装 langchain-mcp-adapters
    pip install langchain-mcp-adapters
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain.agents import create_agent
    except ImportError:
        print("❌ 请先安装依赖：")
        print("   pip install langchain-mcp-adapters")
        return

    print("🚀 创建 MCP Client，连接到我们的财经 MCP Server...\n")

    # 创建 MCP 客户端
    client = MultiServerMCPClient({
        "finance": {
            "transport": "stdio",
            "command": "python",
            "args": [str(Path(__file__).parent / "mcp_server.py")],
        },
    })

    try:
        # 获取所有工具
        tools = await client.get_tools()
        print(f"✅ 成功连接！获取到 {len(tools)} 个工具：")
        for tool in tools:
            print(f"   - {tool.name}: {tool.description[:50]}...")

        # 创建 LLM
        print("\n📝 创建 Agent...")
        from finance_agent import ZHIPU_API_KEY, ZHIPU_BASE_URL, MODEL_NAME

        model = ChatOpenAI(
            model=MODEL_NAME,
            openai_api_key=ZHIPU_API_KEY,
            openai_api_base=ZHIPU_BASE_URL,
            temperature=0.3,
        )

        # 创建 Agent（使用 MCP 提供的工具）
        agent = create_agent(model, tools=tools)

        # 测试查询
        print("\n💬 测试查询：查询苹果公司的股票信息\n")
        print("=" * 60)

        result = await agent.ainvoke({
            "messages": [{
                "role": "user",
                "content": "查询苹果公司（AAPL）的股票信息"
            }]
        })

        print(result["messages"][-1].content)
        print("=" * 60)

        print("\n✅ MCP Client 演示完成！")
        print("\n💡 这个例子展示了：")
        print("   1. 如何连接 MCP Server（stdio 模式）")
        print("   2. 如何获取 MCP Server 提供的工具")
        print("   3. 如何在 LangChain Agent 中使用这些工具")
        print("\n📌 实际应用场景：")
        print("   - 整合多个 MCP Server（天气、数据库、API 等）")
        print("   - 复用社区提供的 MCP 工具")
        print("   - 构建复杂的多数据源 Agent")

    finally:
        # 清理资源
        await client.cleanup()


async def demo_multiple_servers():
    """
    演示连接多个 MCP Server
    """
    print("\n" + "=" * 60)
    print("🌐 高级示例：连接多个 MCP Server")
    print("=" * 60)

    print("""
这个示例展示了如何同时连接多个 MCP Server：

```python
client = MultiServerMCPClient({
    # 财经数据
    "finance": {
        "transport": "stdio",
        "command": "python",
        "args": ["agents/mcp_server.py"],
    },

    # 天气服务（假设有这个服务）
    "weather": {
        "transport": "http",
        "url": "http://weather-api.com/mcp",
    },

    # 新闻服务（假设有这个服务）
    "news": {
        "transport": "http",
        "url": "http://news-api.com/mcp",
    },
})

# Agent 可以同时使用所有服务的工具
tools = await client.get_tools()
agent = create_agent(model, tools=tools)

# 复杂查询：结合多个数据源
result = await agent.ainvoke({
    "messages": [{
        "role": "user",
        "content": "分析特斯拉股价，考虑今天的天气和最新新闻"
    }]
})
```

🎯 这样做的好处：
1. 一个 Agent，多个数据源
2. 复用社区工具，不重复造轮子
3. 模块化架构，易于扩展
""")


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════╗
║         MCP Client 使用示例                               ║
║  演示如何在 LangChain 中调用 MCP Server                   ║
╚══════════════════════════════════════════════════════════╝
""")

    # 运行基础示例
    asyncio.run(demo_mcp_client())

    # 运行高级示例说明
    asyncio.run(demo_multiple_servers())


if __name__ == "__main__":
    main()
