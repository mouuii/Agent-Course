"""
LangGraph 进阶：客服邮件 Agent - 使用智谱 GLM-5 模型

演示 Command 路由、interrupt 人工审核、状态设计
"""

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
import json

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

# ========== 2. 状态 ==========
class EmailClassification(TypedDict):
    intent: str       # question / bug / billing / feature / complex
    urgency: str      # low / medium / high / critical
    topic: str
    summary: str

class EmailAgentState(TypedDict):
    email_content: str
    sender_email: str
    email_id: str
    classification: EmailClassification | None
    search_results: list[str] | None
    draft_response: str | None

# ========== 3. 节点 ==========

def read_email(state: EmailAgentState):
    """读取邮件。"""
    print(f"📧 收到邮件：{state['email_content'][:50]}...")
    return {}


def classify_intent(state: EmailAgentState) -> Command:
    """用 LLM 分类邮件，然后用 Command 路由到下一步。"""

    prompt = (
        "分析下面这封客户邮件，返回 JSON 格式的分类结果。\n"
        "字段：intent(question/bug/billing/feature/complex), "
        "urgency(low/medium/high/critical), topic(话题), summary(总结)\n"
        "只返回 JSON，不要其他内容。\n\n"
        f"邮件：{state['email_content']}\n发件人：{state['sender_email']}"
    )
    response = llm.invoke([("human", prompt)])

    # 解析 JSON（智谱有时会用 ```json 包裹）
    try:
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        classification = json.loads(text.strip())
    except (json.JSONDecodeError, IndexError):
        classification = {
            "intent": "complex", "urgency": "medium",
            "topic": "unknown", "summary": state["email_content"][:100]
        }

    print(f"📋 分类结果：intent={classification.get('intent')}, urgency={classification.get('urgency')}")

    # 根据意图路由
    intent = classification.get("intent", "complex")
    if intent in ["question", "feature", "billing"]:
        goto = "search_documentation"
    elif intent == "bug":
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    return Command(update={"classification": classification}, goto=goto)


def search_documentation(state: EmailAgentState) -> Command:
    """搜索知识库（这里用 mock 数据模拟）。"""
    topic = state.get("classification", {}).get("topic", "")
    email = state.get("email_content", "").lower()

    mock_docs = {
        "password": ["重置密码：设置 → 安全 → 修改密码", "密码要求：至少12位，包含大小写和数字"],
        "密码": ["重置密码：设置 → 安全 → 修改密码", "密码要求：至少12位，包含大小写和数字"],
        "export": ["导出支持 PDF/CSV/Excel 三种格式", "大文件导出可能需要几分钟"],
        "导出": ["导出支持 PDF/CSV/Excel 三种格式", "大文件导出可能需要几分钟"],
        "api": ["API 限流：每秒最多100次请求", "504 错误通常是网关超时"],
        "dark": ["深色模式已在开发计划中", "预计下个版本发布"],
        "深色": ["深色模式已在开发计划中", "预计下个版本发布"],
        "billing": ["退款流程：提交工单后3-5个工作日处理", "重复扣款：系统自动检测并退还多余费用"],
        "扣款": ["退款流程：提交工单后3-5个工作日处理", "重复扣款：系统自动检测并退还多余费用"],
        "订阅": ["退款流程：提交工单后3-5个工作日处理", "重复扣款：系统自动检测并退还多余费用"],
    }

    results = []
    for key, docs in mock_docs.items():
        if key in topic.lower() or key in email:
            results.extend(docs)
    # 去重
    results = list(dict.fromkeys(results))

    if not results:
        results = ["暂未找到相关文档"]

    print(f"🔍 搜索到 {len(results)} 条相关文档")
    return Command(update={"search_results": results}, goto="draft_response")


def bug_tracking(state: EmailAgentState) -> Command:
    """创建 Bug 工单。"""
    ticket_id = "BUG-" + state.get("email_id", "000")[-3:]
    print(f"🐛 创建了 Bug 工单：{ticket_id}")
    return Command(
        update={"search_results": [f"已创建 Bug 工单 {ticket_id}，技术团队会尽快处理"]},
        goto="draft_response"
    )


def draft_response(state: EmailAgentState) -> Command:
    """生成回复草稿，根据紧急程度决定走人工审核还是直接发送。"""
    classification = state.get("classification", {})

    context = ""
    if state.get("search_results"):
        context = "参考资料：\n" + "\n".join(f"- {doc}" for doc in state["search_results"])

    prompt = (
        f"为这封客户邮件写回复：\n{state['email_content']}\n"
        f"类型：{classification.get('intent', '未知')}\n"
        f"紧急程度：{classification.get('urgency', '中')}\n"
        f"{context}\n"
        "要求：专业友好，针对具体问题，用中文，直接输出回复内容"
    )
    response = llm.invoke([("human", prompt)])

    needs_review = (
        classification.get("urgency") == "critical"
        or classification.get("intent") == "billing"
    )
    goto = "human_review" if needs_review else "send_reply"
    print(f"✍️  回复草稿已生成，{'需要人工审核' if needs_review else '直接发送'}")

    return Command(update={"draft_response": response.content}, goto=goto)


def human_review(state: EmailAgentState) -> Command:
    """人工审核节点 - interrupt() 会暂停图的执行，等待人工输入。"""
    classification = state.get("classification", {})

    # interrupt() 暂停执行，把信息展示给审核人
    human_decision = interrupt({
        "email_id": state.get("email_id", ""),
        "original_email": state.get("email_content", ""),
        "draft_response": state.get("draft_response", ""),
        "urgency": classification.get("urgency"),
        "action": "请审核这封回复，approved=True 表示通过"
    })

    # 人工传回 Command(resume={...}) 后，从这里继续
    if human_decision.get("approved"):
        edited = human_decision.get("edited_response", state.get("draft_response", ""))
        print("✅ 人工审核通过")
        return Command(update={"draft_response": edited}, goto="send_reply")
    else:
        print("❌ 人工拒绝，由人工自行处理")
        return Command(update={}, goto="__end__")


def send_reply(state: EmailAgentState):
    """发送邮件回复。"""
    print(f"📤 回复已发送给 {state['sender_email']}")
    print(f"   内容预览：{state.get('draft_response', '')[:100]}...")
    return {}


# ========== 4. 组装图 ==========
graph_builder = StateGraph(EmailAgentState)

graph_builder.add_node("read_email", read_email)
graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("search_documentation", search_documentation)
graph_builder.add_node("bug_tracking", bug_tracking)
graph_builder.add_node("draft_response", draft_response)
graph_builder.add_node("human_review", human_review)
graph_builder.add_node("send_reply", send_reply)

# 只需要三条固定边，其余路由全靠 Command
graph_builder.add_edge(START, "read_email")
graph_builder.add_edge("read_email", "classify_intent")
graph_builder.add_edge("send_reply", END)

# MemorySaver 做 checkpointer，interrupt() 需要它来保存状态
memory = MemorySaver()
app = graph_builder.compile(checkpointer=memory)
print("✅ 图编译完成\n")


# ========== 5. 测试 ==========

# --- 测试 1：简单问题（全自动流程）---
print("=" * 50)
print("测试 1：简单问题（全自动）")
print("=" * 50)
result = app.invoke(
    {
        "email_content": "你好，请问怎么重置密码？谢谢",
        "sender_email": "zhangsan@example.com",
        "email_id": "email_001",
    },
    config={"configurable": {"thread_id": "test_001"}}
)
print(f"\n最终回复：\n{result.get('draft_response', '')}")


# --- 测试 2：紧急账单（需要人工审核）---
print("\n\n")
print("=" * 50)
print("测试 2：紧急账单问题（人工审核流程）")
print("=" * 50)
result = app.invoke(
    {
        "email_content": "我被扣了两次订阅费！这很紧急，请立刻处理！",
        "sender_email": "lisi@example.com",
        "email_id": "email_002",
    },
    config={"configurable": {"thread_id": "test_002"}}
)
print("\n--- 图在 human_review 节点暂停了 ---")
print(f"当前草稿预览：{result.get('draft_response', '')[:100]}...")

# 模拟人工审核通过
print("\n模拟人工审核通过：")
final = app.invoke(
    Command(resume={
        "approved": True,
        "edited_response": "非常抱歉给您造成不便！我们已确认您的账户存在重复扣款，退款将在3个工作日内到账。如有其他问题，请随时联系我们。"
    }),
    config={"configurable": {"thread_id": "test_002"}}
)
print(f"\n最终回复：\n{final.get('draft_response', '')}")


# --- 测试 3：Bug 报告（自动创建工单）---
print("\n\n")
print("=" * 50)
print("测试 3：Bug 报告（自动创建工单）")
print("=" * 50)
result = app.invoke(
    {
        "email_content": "导出 PDF 的时候页面直接崩溃了，每次都这样",
        "sender_email": "wangwu@example.com",
        "email_id": "email_003",
    },
    config={"configurable": {"thread_id": "test_003"}}
)
print(f"\n最终回复：\n{result.get('draft_response', '')}")


# --- 测试 4：功能建议（全自动）---
print("\n\n")
print("=" * 50)
print("测试 4：功能建议（全自动）")
print("=" * 50)
result = app.invoke(
    {
        "email_content": "能不能加个深色模式？晚上用太刺眼了",
        "sender_email": "zhaoliu@example.com",
        "email_id": "email_004",
    },
    config={"configurable": {"thread_id": "test_004"}}
)
print(f"\n最终回复：\n{result.get('draft_response', '')}")
