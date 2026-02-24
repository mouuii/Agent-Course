"""
LangGraph 入门：计算器 Agent - 使用智谱 GLM-5 模型
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
from typing import Literal
from langgraph.graph import StateGraph, START, END
import operator

# ========== 1. 模型 ==========
ZHIPU_API_KEY = "87d066b707514d128dd6929ebce7959e.DjjZdsvdQ1ockUnN"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

llm = ChatOpenAI(
    temperature=0,
    model="glm-5",
    openai_api_key=ZHIPU_API_KEY,
    openai_api_base=ZHIPU_BASE_URL,
)

print("✅ 模型初始化完成")

# ========== 2. 工具 ==========
@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: First int
        b: Second int
    """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: First int
        b: Second int
    """
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """Divide a by b.

    Args:
        a: First int
        b: Second int
    """
    return a / b

tools = [add, multiply, divide]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

print("✅ 工具定义完成")

# ========== 3. 状态 ==========
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# ========== 4. 节点 ==========
def llm_call(state: MessagesState):
    """大模型决定：直接回答还是调工具。"""
    return {
        "messages": [
            llm_with_tools.invoke(
                [SystemMessage(content="你是一个计算助手，使用提供的工具来完成计算任务。")]
                + state["messages"]
            )
        ]
    }

def tool_node(state: MessagesState):
    """执行工具调用。"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        t = tools_by_name[tool_call["name"]]
        observation = t.invoke(tool_call["args"])
        result.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )
        print(f"  🔧 调用 {tool_call['name']}({tool_call['args']}) = {observation}")
    return {"messages": result}

# ========== 5. 路由 ==========
def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END

# ========== 6. 组装图 ==========
graph_builder = StateGraph(MessagesState)

graph_builder.add_node("llm_call", llm_call)
graph_builder.add_node("tool_node", tool_node)

graph_builder.add_edge(START, "llm_call")
graph_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
graph_builder.add_edge("tool_node", "llm_call")

agent = graph_builder.compile()
print("✅ Agent 编译完成\n")

# ========== 7. 测试 ==========

print("=" * 50)
print("测试 1：简单加法")
print("=" * 50)
response = agent.invoke(
    {"messages": [HumanMessage(content="3 加 4 等于多少？")]}
)
print()
for msg in response["messages"]:
    msg.pretty_print()

print("\n")
print("=" * 50)
print("测试 2：多步计算")
print("=" * 50)
response = agent.invoke(
    {"messages": [HumanMessage(content="先算 3 加 4，再把结果乘以 2")]}
)
print()
for msg in response["messages"]:
    msg.pretty_print()

print("\n")
print("=" * 50)
print("测试 3：不需要工具的问题")
print("=" * 50)
response = agent.invoke(
    {"messages": [HumanMessage(content="你好，你是谁？")]}
)
print()
for msg in response["messages"]:
    msg.pretty_print()

print("\n")
print("=" * 50)
print("测试 4：复杂计算")
print("=" * 50)
response = agent.invoke(
    {"messages": [HumanMessage(content="100 除以 5，再加上 30，最后乘以 3")]}
)
print()
for msg in response["messages"]:
    msg.pretty_print()
