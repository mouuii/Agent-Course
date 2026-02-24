"""
用 LangGraph 搭一个 SQL Agent - 使用智谱 GLM-5 模型

让 AI 帮你查数据库：自然语言 → SQL → 结果 → 回答
"""

import requests
import pathlib
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

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

# ========== 2. 数据库 ==========
db_path = pathlib.Path(__file__).parent / "Chinook.db"

if not db_path.exists():
    print("⏳ 正在下载 Chinook 示例数据库...")
    url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
    resp = requests.get(url)
    db_path.write_bytes(resp.content)
    print(f"✅ 下载完成：{db_path}")
else:
    print(f"✅ 数据库已存在：{db_path}")

db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
print(f"   方言：{db.dialect}")
print(f"   可用表：{db.get_usable_table_names()}")

# ========== 3. 工具 ==========
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

for t in tools:
    print(f"   🔧 {t.name}")

get_schema_tool = next(t for t in tools if t.name == "sql_db_schema")
run_query_tool = next(t for t in tools if t.name == "sql_db_query")
list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")

get_schema_node = ToolNode([get_schema_tool], name="get_schema")
run_query_node = ToolNode([run_query_tool], name="run_query")

print("✅ 工具准备完成")


# ========== 4. 节点 ==========

def list_tables(state: MessagesState):
    """第一步：列出所有可用的表。硬编码调用，不需要 LLM。"""
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "list_tables_001",
        "type": "tool_call",
    }
    tool_call_msg = AIMessage(content="", tool_calls=[tool_call])
    tool_msg = list_tables_tool.invoke(tool_call)
    response = AIMessage(content=f"Available tables: {tool_msg.content}")
    print(f"📋 列出所有表：{tool_msg.content}")
    return {"messages": [tool_call_msg, tool_msg, response]}


def call_get_schema(state: MessagesState):
    """第二步：让 LLM 选择要查看哪些表的结构，强制调用工具。"""
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
    if response.tool_calls:
        tables = response.tool_calls[0]["args"].get("table_names", "")
        print(f"📊 查看表结构：{tables}")
    return {"messages": [response]}


generate_query_prompt = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain,
always limit your query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

回答用户时请使用中文。""".format(dialect=db.dialect, top_k=5)


def generate_query(state: MessagesState):
    """第三步：根据用户问题和表结构，生成 SQL 查询或直接回答。"""
    system_msg = {"role": "system", "content": generate_query_prompt}
    llm_with_tools = llm.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_msg] + state["messages"])
    if response.tool_calls:
        query = response.tool_calls[0]["args"].get("query", "")
        print(f"📝 生成 SQL：{query}")
    else:
        print("💬 模型直接回答（不需要查询）")
    return {"messages": [response]}


check_query_prompt = """You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query.
If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.""".format(
    dialect=db.dialect
)


def check_query(state: MessagesState):
    """第四步：SQL 专家检查查询，没问题就执行。"""
    system_msg = {"role": "system", "content": check_query_prompt}
    tool_call = state["messages"][-1].tool_calls[0]
    query = tool_call["args"]["query"]
    user_msg = {"role": "user", "content": query}

    llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_msg, user_msg])
    # 关键：用相同 ID 替换上一条消息，让 ToolNode 能正确匹配
    response.id = state["messages"][-1].id

    checked_query = ""
    if response.tool_calls:
        checked_query = response.tool_calls[0]["args"].get("query", "")

    if checked_query and checked_query != query:
        print(f"🔍 查询已修正：{checked_query}")
    else:
        print("✅ 查询检查通过")

    return {"messages": [response]}


# ========== 5. 路由 ==========

def should_continue(state: MessagesState) -> Literal["check_query", "__end__"]:
    """generate_query 之后：有工具调用就检查查询，没有就结束。"""
    if state["messages"][-1].tool_calls:
        return "check_query"
    return END


# ========== 6. 组装图 ==========

builder = StateGraph(MessagesState)

builder.add_node("list_tables", list_tables)
builder.add_node("call_get_schema", call_get_schema)
builder.add_node("get_schema", get_schema_node)
builder.add_node("generate_query", generate_query)
builder.add_node("check_query", check_query)
builder.add_node("run_query", run_query_node)

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges("generate_query", should_continue)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")

agent = builder.compile()
print("✅ Agent 编译完成\n")


# ========== 7. 测试 ==========

def ask(question: str):
    """向 SQL Agent 提问。"""
    print("=" * 60)
    print(f"❓ {question}")
    print("-" * 60)

    result = agent.invoke({"messages": [("user", question)]})

    # 找最后一条有内容的 AI 回复
    for msg in reversed(result["messages"]):
        if (hasattr(msg, "content") and msg.content
                and not getattr(msg, "tool_call_id", None)):
            print(f"\n💡 回答：\n{msg.content}")
            break

    print()


# 测试 1：聚合查询
ask("哪个音乐流派的歌曲平均时长最长？前 5 名是哪些？")

# 测试 2：多表联查
ask("消费金额最高的前 5 个客户是谁？分别花了多少钱？")

# 测试 3：简单计数
ask("一共有多少首歌？")
