"""
LangSmith 追踪教程：给你的 AI Agent 装个"行车记录仪"

用 LangGraph + LangSmith 搭一个维基百科研究助手，
顺便演示怎么用 LangSmith 看清 Agent 每一步在干嘛。

使用智谱 GLM-5 模型
"""

import os
import time
import requests
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# ========== 1. 配置 ==========

# 智谱模型
ZHIPU_API_KEY = "87d066b707514d128dd6929ebce7959e.DjjZdsvdQ1ockUnN"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

# LangSmith 追踪（填你自己的 key，没有就设 false 先跑通）
os.environ["LANGCHAIN_TRACING_V2"] = os.environ.get("LANGCHAIN_API_KEY", "") and "true" or "false"
os.environ["LANGCHAIN_PROJECT"] = "langsmith-demo-zhipu"
# os.environ["LANGCHAIN_API_KEY"] = "你的 LangSmith API Key"

llm = ChatOpenAI(
    temperature=0,
    model="glm-5",
    openai_api_key=ZHIPU_API_KEY,
    openai_api_base=ZHIPU_BASE_URL,
)

print("✅ 模型初始化完成")


# ========== 2. 状态 ==========

class AgentState(TypedDict):
    user_question: str      # 用户问题
    needs_search: bool      # 是否需要搜索
    search_result: str      # 搜索结果
    final_answer: str       # 最终回答
    reasoning: str          # 决策理由


# ========== 3. 搜索工具 ==========

@tool
def wikipedia_search(query: str) -> str:
    """搜索维基百科获取信息。"""
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3,
        }
        resp = requests.get(search_url, params=params, timeout=10)

        if resp.status_code != 200:
            return f"搜索失败，状态码：{resp.status_code}"

        results = resp.json().get("query", {}).get("search", [])
        if not results:
            return f"没找到关于 '{query}' 的内容"

        # 拿第一条结果的摘要
        title = results[0]["title"]
        summary_url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/"
            f"{title.replace(' ', '_')}"
        )
        summary_resp = requests.get(summary_url, timeout=10)

        if summary_resp.status_code == 200:
            extract = summary_resp.json().get("extract", "无摘要")
            return f"关于 '{title}'：{extract[:500]}"
        else:
            return f"找到了 '{title}'，但获取摘要失败"

    except Exception as e:
        return f"搜索出错：{e}"


print("✅ 搜索工具准备完成")


# ========== 4. 节点 ==========

def decide_search_need(state: AgentState) -> AgentState:
    """第一步：判断是否需要搜索。"""
    question = state["user_question"]

    prompt = f"""分析这个问题，判断是否需要搜索最新信息：

问题："{question}"

判断标准：
- 如果问的是最近的新闻、实时数据、当前价格 → 需要搜索
- 如果问的是常识、历史知识、概念解释 → 不需要搜索

只回复 SEARCH 或 DIRECT，然后换行写理由。"""

    response = llm.invoke([
        SystemMessage(content="你是一个智能助手，帮助判断用户的问题是否需要搜索。"),
        HumanMessage(content=prompt),
    ])
    text = response.content.strip()

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    decision = lines[0].upper() if lines else "DIRECT"
    reasoning = lines[1] if len(lines) > 1 else "无"

    # 只要包含 SEARCH 就认为需要搜索
    needs_search = "SEARCH" in decision
    state["needs_search"] = needs_search
    state["reasoning"] = f"{'需要搜索' if needs_search else '直接回答'}。理由：{reasoning}"

    print(f"🤔 判断：{'需要搜索' if needs_search else '直接回答'} — {reasoning}")
    return state


def execute_search(state: AgentState) -> AgentState:
    """第二步：如果需要就搜索，不需要就跳过。"""
    if not state["needs_search"]:
        print("⏭️  跳过搜索")
        state["search_result"] = ""
        return state

    print(f"🔍 搜索中：{state['user_question']}")
    result = wikipedia_search.invoke({"query": state["user_question"]})
    state["search_result"] = result
    print(f"📄 搜索完成，返回 {len(result)} 字符")
    return state


def generate_response(state: AgentState) -> AgentState:
    """第三步：生成最终回答。"""
    question = state["user_question"]
    search_result = state.get("search_result", "")

    if state["needs_search"] and search_result and "搜索出错" not in search_result:
        prompt = f"""根据搜索结果回答用户问题，用中文回答。

问题：{question}

搜索结果：{search_result}

要求：结合搜索结果给出准确、有帮助的回答。"""
        messages = [
            SystemMessage(content="你是一个知识渊博的研究助手。"),
            HumanMessage(content=prompt),
        ]
    else:
        prompt = f"""用你已有的知识回答这个问题，用中文回答。

问题：{question}

要求：给出准确、有帮助的回答。"""
        messages = [
            SystemMessage(content="你是一个知识渊博的助手。"),
            HumanMessage(content=prompt),
        ]

    response = llm.invoke(messages)
    state["final_answer"] = response.content

    print(f"💬 回答生成完成（{len(response.content)} 字符）")
    return state


# ========== 5. 组装图 ==========

builder = StateGraph(AgentState)

builder.add_node("decide", decide_search_need)
builder.add_node("search", execute_search)
builder.add_node("respond", generate_response)

builder.set_entry_point("decide")
builder.add_edge("decide", "search")
builder.add_edge("search", "respond")
builder.add_edge("respond", END)

agent = builder.compile()
print("✅ Agent 编译完成\n")


# ========== 6. 测试 ==========

def ask(question: str, test_type: str = "general"):
    """运行一次测试。"""
    print("=" * 55)
    print(f"❓ {question}")
    print("-" * 55)

    start = time.time()

    initial_state = {
        "user_question": question,
        "needs_search": False,
        "search_result": "",
        "final_answer": "",
        "reasoning": "",
    }

    config = {
        "metadata": {"test_type": test_type},
        "tags": ["tutorial", test_type],
    }

    result = agent.invoke(initial_state, config=config)
    elapsed = time.time() - start

    print(f"\n💡 回答：\n{result['final_answer'][:300]}")
    print(f"\n⏱️  耗时：{elapsed:.1f}s | 搜索：{'是' if result['needs_search'] else '否'}")
    print()
    return result


# 测试 1：常识题（不需要搜索）
ask("法国的首都是哪里？", "direct_answer")

# 测试 2：需要搜索的题
ask("2024年美国总统大选结果是什么？", "current_info")

# 测试 3：概念解释（不需要搜索）
ask("简单介绍一下人工智能", "factual_lookup")

print("\n🎉 全部测试完成！")
print("去 https://smith.langchain.com 查看完整的追踪记录")
