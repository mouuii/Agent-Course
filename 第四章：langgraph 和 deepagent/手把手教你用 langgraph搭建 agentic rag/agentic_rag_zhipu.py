"""
Agentic RAG with LangGraph - 使用智谱(Zhipu) GLM 模型
"""

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import SystemMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

# ========== 1. 初始化智谱模型 ==========

ZHIPU_API_KEY = "87d066b707514d128dd6929ebce7959e.DjjZdsvdQ1ockUnN"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

llm = ChatOpenAI(
    temperature=0,
    model="glm-5",
    openai_api_key=ZHIPU_API_KEY,
    openai_api_base=ZHIPU_BASE_URL,
)

# Embedding 也用智谱的
zhipu_embeddings = OpenAIEmbeddings(
    model="embedding-3",
    openai_api_key=ZHIPU_API_KEY,
    openai_api_base=ZHIPU_BASE_URL,
)

print("✅ 模型初始化完成")

# ========== 2. 建知识库 ==========

print("⏳ 正在加载网页...")
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content")
    )},
)
docs = loader.load()
print(f"   加载了 {len(docs)} 个文档，共 {len(docs[0].page_content)} 个字符")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
all_splits = text_splitter.split_documents(docs)
print(f"   切成了 {len(all_splits)} 个文档块")

print("⏳ 正在向量化...")
vector_store = InMemoryVectorStore(zhipu_embeddings)
vector_store.add_documents(all_splits)
print("✅ 知识库建好了")

# ========== 3. 检索工具 ==========

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Search blog posts for relevant information to answer user questions."""
    retrieved_docs = vector_store.similarity_search(query, k=5)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# ========== 4. 图节点 ==========

def generate_query_or_respond(state: MessagesState):
    """Agent 决策：搜索还是直接回答。"""
    llm_with_tools = llm.bind_tools([retrieve])
    system = SystemMessage(
        content="你是一个博客文章问答助手。当用户的问题涉及 AI Agent、LLM、任务分解、"
                "记忆机制等技术话题时，你必须先使用 retrieve 工具搜索文档来获取准确信息。"
                "只有当问题明显与技术无关（如闲聊、简单数学）时，才直接回答。"
                "重要：知识库中的文档是英文的，所以你在调用 retrieve 工具时，query 参数必须用英文。"
    )
    response = llm_with_tools.invoke([system] + state["messages"])
    if response.tool_calls:
        print(f"--- Agent 决定搜索：{response.tool_calls[0]['args']['query']} ---")
    else:
        print("--- Agent 决定直接回答 ---")
    return {"messages": [response]}


retrieve_node = ToolNode([retrieve])


def grade_documents(state: MessagesState):
    """评估检索文档的相关性。

    智谱模型的 with_structured_output 可能不稳定，
    这里用普通文本解析代替。
    """
    messages = state["messages"]

    # 拿到用户原始问题和检索结果
    question = messages[0].content
    tool_content = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "tool":
            tool_content = msg.content
            break

    grading_prompt = (
        "你是一个评估检索文档与用户问题相关性的评分员。\n"
        "如果文档包含与问题相关的关键词或语义内容，回复 yes。\n"
        "如果文档与问题完全无关，回复 no。\n"
        "只回复 yes 或 no，不要解释。\n\n"
        f"用户问题：{question}\n\n"
        f"检索到的文档：{tool_content[:2000]}"
    )

    response = llm.invoke([("human", grading_prompt)])
    score = response.content.strip().lower()

    if "yes" in score:
        print("--- 评估结果：文档相关 ---")
        return {"messages": state["messages"]}
    else:
        print("--- 评估结果：文档不相关 ---")
        return {"messages": [messages[0]]}


def rewrite_question(state: MessagesState):
    """改写问题以优化检索效果。"""
    question = state["messages"][0].content
    msg = llm.invoke(
        [SystemMessage(content="优化下面的问题，使其更适合语义搜索。只输出改写后的问题，不要解释。")]
        + [("human", question)]
    )
    print(f"--- 问题改写：{question} → {msg.content} ---")
    return {"messages": [("human", msg.content)]}


def generate_answer(state: MessagesState):
    """基于检索到的文档生成回答。"""
    messages = state["messages"]

    docs_content = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "tool":
            docs_content = msg.content
            break

    question = messages[0].content
    response = llm.invoke(
        [SystemMessage(content="你是一个问答助手。根据检索到的文档回答问题。用中文回答，简洁准确。")]
        + [("human", f"问题：{question}\n\n参考文档：\n{docs_content}")]
    )
    return {"messages": [response]}


# ========== 5. 连边 ==========

graph_builder = StateGraph(MessagesState)

graph_builder.add_node("generate_query_or_respond", generate_query_or_respond)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("grade_documents", grade_documents)
graph_builder.add_node("rewrite_question", rewrite_question)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "generate_query_or_respond")
graph_builder.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {"tools": "retrieve", END: END},
)
graph_builder.add_edge("retrieve", "grade_documents")


def route_after_grading(state: MessagesState):
    if len(state["messages"]) == 1:
        return "rewrite_question"
    return "generate_answer"


graph_builder.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {"rewrite_question": "rewrite_question", "generate_answer": "generate_answer"},
)
graph_builder.add_edge("rewrite_question", "generate_query_or_respond")
graph_builder.add_edge("generate_answer", END)

graph = graph_builder.compile()
print("✅ 图编译完成\n")

# ========== 6. 测试 ==========

print("=" * 50)
print("测试 1：需要检索的问题")
print("=" * 50)
response = graph.invoke(
    {"messages": [("human", "Agent 有哪些类型的记忆？")]}
)
print()
for msg in response["messages"]:
    msg.pretty_print()

print("\n")
print("=" * 50)
print("测试 2：不需要检索的问题")
print("=" * 50)
response = graph.invoke(
    {"messages": [("human", "你好，1+1等于几？")]}
)
print()
for msg in response["messages"]:
    msg.pretty_print()

print("\n")
print("=" * 50)
print("测试 3：需要多步推理的问题")
print("=" * 50)
response = graph.invoke(
    {"messages": [("human", "什么是 Chain of Thought prompting？它和 Tree of Thoughts 有什么区别？")]}
)
print()
for msg in response["messages"]:
    msg.pretty_print()
