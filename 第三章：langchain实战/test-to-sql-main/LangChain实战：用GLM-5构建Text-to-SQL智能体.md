# LangChain 实战：用 GLM-5 构建 Text-to-SQL 智能体

> 让大模型听懂人话、写出 SQL、查出结果——从零到跑通只需 50 行代码。

## 你将学到什么

读完本教程，你将掌握：

1. LangChain Agent 的核心概念（Tool、Toolkit、System Prompt）
2. 如何用 `SQLDatabaseToolkit` 让 LLM 自主探索和查询数据库
3. 如何通过 `ChatOpenAI` 接入国产大模型（智谱 GLM-5）
4. 如何用 LangSmith 追踪 Agent 的每一步推理过程

**前置知识**：了解 Python 基础语法、知道 SQL 是什么即可。

---

## 一、项目全貌

### 1.1 我们要做什么

用户输入一句自然语言（比如"加拿大有多少客户？"），Agent 自动完成以下流程：

```
用户提问 → 探索数据库表结构 → 生成 SQL → 执行查询 → 返回自然语言答案
```

关键点：**Agent 不是一次性生成 SQL，而是像一个真正的数据分析师一样，先看有什么表、再看表结构、然后才写查询，写错了还会自己修改重试**。

### 1.2 技术栈

| 组件 | 作用 |
|------|------|
| **LangChain** | Agent 框架，编排 LLM 与工具的交互 |
| **LangGraph** | LangChain 的底层运行时，驱动 Agent 的状态机循环 |
| **ChatOpenAI** | LLM 接口，通过 OpenAI 兼容协议接入智谱 GLM-5 |
| **SQLDatabaseToolkit** | 提供数据库交互工具集（列表、查 schema、执行 SQL 等） |
| **SQLAlchemy** | 数据库连接层 |
| **Chinook DB** | 示例数据库——一个数字音乐商店，含 11 张表、3500+ 条音轨数据 |

### 1.3 项目结构

```
text-to-sql-agent/
├── agent.py              # 核心代码，~50 行
├── tutorial.ipynb         # Jupyter 交互式教程
├── chinook.db            # SQLite 示例数据库
├── pyproject.toml        # 依赖配置
├── .env                  # API 密钥（不入 git）
└── .env.example          # 密钥模板
```

---

## 二、环境搭建

### 2.1 安装依赖

```bash
git clone https://github.com/kevinbfrank/text-to-sql-agent.git
cd text-to-sql-agent

# 创建虚拟环境
uv venv --python 3.11
source .venv/bin/activate

# 安装项目
uv pip install -e .
```

### 2.2 准备数据库

下载 Chinook 示例数据库：

```bash
curl -L -o Chinook_Sqlite.sql \
  "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
sqlite3 chinook.db < Chinook_Sqlite.sql
rm Chinook_Sqlite.sql
```

Chinook 是一个经典的教学数据库，模拟数字音乐商店，包含以下数据：

| 表名 | 说明 | 行数 |
|------|------|------|
| Track | 音轨 | 3503 |
| InvoiceLine | 发票明细 | 2240 |
| Invoice | 发票 | 412 |
| Album | 专辑 | 347 |
| Artist | 艺术家 | 275 |
| Customer | 客户 | 59 |
| Genre | 流派 | 25 |
| Employee | 员工 | 8 |

表之间通过外键关联，例如 `Track → Album → Artist`，`InvoiceLine → Invoice → Customer`，非常适合练习多表 JOIN 查询。

### 2.3 配置 API 密钥

```bash
cp .env.example .env
```

编辑 `.env`：

```env
# 智谱 AI（GLM-5）
ZHIPU_API_KEY=你的智谱API密钥
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/

# LangSmith（可选，用于追踪调试）
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=你的LangSmith密钥
LANGSMITH_PROJECT="text2sql-agent"
```

> **为什么用 GLM-5？** 智谱的 GLM-5 兼容 OpenAI API 协议，可以直接通过 LangChain 的 `ChatOpenAI` 接入，无需额外适配。国内访问速度快、注册即有免费额度，非常适合学习和实验。

---

## 三、逐行拆解核心代码

下面我们逐步拆解 `agent.py`，理解每一个组件的作用。

### 3.1 连接数据库

```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri(
    "sqlite:///chinook.db",
    sample_rows_in_table_info=3   # 获取表结构时，附带 3 行样本数据
)
```

`SQLDatabase` 是 LangChain 对 SQLAlchemy 的封装。`sample_rows_in_table_info=3` 是个关键参数——当 Agent 查看表结构时，除了列名和类型，还会看到 3 行真实数据，这能帮助 LLM 理解每个字段的实际含义（比如看到 `Country` 列的值是 "Canada"、"Brazil" 就能理解这是国家名）。

### 3.2 初始化大模型

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="glm-5",
    openai_api_key=os.getenv("ZHIPU_API_KEY"),
    openai_api_base=os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"),
    temperature=0.3,
)
```

**要点解读**：

- **`ChatOpenAI` 不只是 OpenAI 专用的**。任何兼容 OpenAI API 协议的大模型（智谱、通义、Moonshot、DeepSeek 等）都可以通过 `openai_api_base` 参数接入。这是 LangChain 灵活性的体现。
- **`temperature=0.3`**：SQL 生成需要确定性，温度设低一些减少随机性。生产环境中甚至可以设为 0。

### 3.3 创建工具集（Toolkit）

```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
```

这一步是核心。`SQLDatabaseToolkit` 会自动生成以下工具供 Agent 调用：

| 工具名 | 作用 | Agent 使用场景 |
|--------|------|---------------|
| `sql_db_list_tables` | 列出所有表名 | 第一步：了解数据库有哪些表 |
| `sql_db_schema` | 查看指定表的 DDL + 样本数据 | 第二步：了解表结构和字段含义 |
| `sql_db_query` | 执行 SQL 并返回结果 | 第三步：执行生成的查询 |
| `sql_db_query_checker` | 让 LLM 检查 SQL 是否正确 | 执行前自检，减少错误 |

**这就是 Agent 和普通 Chain 的本质区别**：Agent 拥有工具，能自主决定什么时候用什么工具。它不是按照固定流程执行，而是根据当前情境做出判断。

### 3.4 编写 System Prompt

```python
SYSTEM_PROMPT = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
"""
```

**System Prompt 的设计要点**（这是调优 Agent 最关键的部分）：

1. **安全约束**：`DO NOT make any DML statements` — 禁止增删改，只允许查询
2. **行为规范**：`ALWAYS look at the tables... Do NOT skip this step` — 强制 Agent 先探索再查询
3. **质量保证**：`MUST double check your query` — 执行前自检
4. **结果控制**：`limit your query to at most {top_k} results` — 防止返回太多数据
5. **性能优化**：`Never query for all the columns` — 只查需要的列

注意 `{dialect}` 和 `{top_k}` 是动态填入的，分别代表数据库方言（sqlite）和最大结果数（5）。

### 3.5 组装 Agent

```python
from langchain.agents import create_agent

agent = create_agent(
    model,                  # 大模型
    tools,                  # 工具集
    system_prompt=SYSTEM_PROMPT.format(dialect=db.dialect, top_k=5)
)
```

`create_agent` 将模型、工具、提示词组装成一个完整的 Agent。底层基于 LangGraph 的 `ReAct` 模式实现：

```
Think → Act → Observe → Think → Act → Observe → ... → Final Answer
```

即：思考该做什么 → 调用工具 → 观察结果 → 继续思考 → ... → 给出最终答案。

### 3.6 调用 Agent

```python
result = agent.invoke({
    "messages": [{"role": "user", "content": "加拿大有多少客户？"}]
})

answer = result["messages"][-1].content
```

Agent 接收消息列表，返回的也是消息列表。最后一条消息就是 Agent 的最终回答。

---

## 四、运行效果

### 4.1 命令行运行

```bash
python agent.py "How many customers are from Canada?"
```

输出：

```
╭──────────────────────────────────────────────────╮
│ Question: How many customers are from Canada?    │
╰──────────────────────────────────────────────────╯

Creating SQL Agent...
Processing query...

╭──────────────────────────────────────────────────╮
│ Answer:                                          │
│                                                  │
│ There are **8 customers** from Canada.           │
╰──────────────────────────────────────────────────╯
```

### 4.2 Agent 在背后做了什么？

当你问"加拿大有多少客户"时，Agent 的完整推理链路如下：

```
Step 1: 调用 sql_db_list_tables
        → 返回 11 张表名

Step 2: 调用 sql_db_schema("Customer")
        → 返回 Customer 表结构 + 3 行样本数据
        → Agent 发现有 Country 字段

Step 3: 调用 sql_db_query_checker
        → 检查 SQL: SELECT COUNT(*) FROM Customer WHERE Country = 'Canada'
        → 确认语法正确

Step 4: 调用 sql_db_query
        → 执行查询，返回结果: [(8,)]

Step 5: 生成自然语言回答
        → "There are 8 customers from Canada."
```

这就是 Agent 的强大之处——**它不需要你告诉它查哪张表、用什么字段，它自己探索、自己推理、自己执行**。

### 4.3 更多查询示例

```bash
# 聚合查询：按国家统计收入
python agent.py "What is the total revenue by country?"

# 多表 JOIN：最畅销的 5 首歌
python agent.py "What are the top 5 best-selling tracks?"

# 复杂分析：哪个员工创造的收入最多
python agent.py "Which employee generated the most revenue?"
```

---

## 五、用 LangSmith 追踪 Agent 推理过程

LangSmith 是 LangChain 官方的可观测性平台，可以记录 Agent 的每一步调用、每一次 LLM 请求的完整输入输出。

### 5.1 为什么需要 LangSmith？

开发 Agent 时，你经常会遇到：
- Agent 给出了错误答案，但不知道哪一步出了问题
- Agent 调用了不必要的工具，浪费了 token
- SQL 查询出错后，Agent 有没有成功重试？

LangSmith 让整个推理过程透明可见。

### 5.2 配置方式

在 `.env` 中添加：

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=你的密钥
LANGSMITH_PROJECT="text2sql-agent"
```

配置后无需修改任何代码，LangChain 会自动上报 trace。

### 5.3 在 Trace 中你能看到什么

打开 [smith.langchain.com](https://smith.langchain.com)，你可以看到：

- **完整的调用链**：Agent 调用了哪些工具、顺序是什么
- **每一步的输入输出**：LLM 收到了什么 prompt、返回了什么
- **Token 用量**：每次 LLM 调用消耗了多少 token
- **耗时分析**：每一步花了多长时间
- **错误重试**：如果 SQL 执行失败，Agent 的重试过程

---

## 六、核心概念回顾

### 6.1 Agent vs Chain

| | Chain | Agent |
|--|-------|-------|
| 执行方式 | 固定流水线，按预设顺序执行 | 自主决策，根据情况选择下一步 |
| 工具使用 | 不使用工具（或固定使用） | 动态选择工具 |
| 错误处理 | 需要预设错误处理逻辑 | 能自主修改和重试 |
| 适用场景 | 简单、确定的任务 | 需要探索和推理的复杂任务 |

Text-to-SQL 天然适合 Agent 模式，因为：
- 不同问题需要查不同的表（需要先探索）
- SQL 可能写错（需要重试能力）
- 需要多步推理（看表 → 看结构 → 写 SQL → 执行）

### 6.2 Toolkit 设计模式

`SQLDatabaseToolkit` 是 LangChain 的 Toolkit 设计模式的典型代表。一个 Toolkit 将相关的多个 Tool 打包在一起，提供一个完整的能力集。

```python
# Toolkit 模式的好处：一行代码获得整套能力
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()   # 自动创建 4 个工具
```

LangChain 还提供了其他 Toolkit，比如 `GmailToolkit`、`GitHubToolkit` 等，模式是一样的。

### 6.3 OpenAI 兼容协议的威力

通过 `ChatOpenAI` + `openai_api_base`，你可以接入任何兼容 OpenAI 协议的模型：

```python
# 智谱 GLM
ChatOpenAI(model="glm-5", openai_api_base="https://open.bigmodel.cn/api/paas/v4/")

# DeepSeek
ChatOpenAI(model="deepseek-chat", openai_api_base="https://api.deepseek.com/v1")

# 本地 Ollama
ChatOpenAI(model="qwen2.5", openai_api_base="http://localhost:11434/v1")
```

换模型只需改两个参数，业务代码不用动。

---

## 七、进阶方向

掌握了基础之后，你可以继续探索：

1. **换数据库**：把 SQLite 换成 MySQL/PostgreSQL，只需修改连接字符串
2. **加上 Few-Shot**：在 System Prompt 中加入 SQL 示例，提升复杂查询的准确率
3. **限制表范围**：使用 `SQLDatabase(include_tables=["Customer", "Invoice"])` 只暴露部分表
4. **流式输出**：使用 `agent.stream()` 替代 `agent.invoke()`，实时看到 Agent 思考过程
5. **接入 Web UI**：用 Streamlit 或 Gradio 包装成交互式应用

---

## 八、完整代码

以下是 `agent.py` 的完整代码，仅约 50 行核心逻辑：

```python
import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

SYSTEM_PROMPT = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.
Then you should query the schema of the most relevant tables.
"""

# 1. 连接数据库
db = SQLDatabase.from_uri("sqlite:///chinook.db", sample_rows_in_table_info=3)

# 2. 初始化大模型
model = ChatOpenAI(
    model="glm-5",
    openai_api_key=os.getenv("ZHIPU_API_KEY"),
    openai_api_base=os.getenv("ZHIPU_BASE_URL"),
    temperature=0.3,
)

# 3. 创建工具集
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# 4. 组装 Agent
agent = create_agent(
    model, tools,
    system_prompt=SYSTEM_PROMPT.format(dialect=db.dialect, top_k=5)
)

# 5. 提问
result = agent.invoke({
    "messages": [{"role": "user", "content": "加拿大有多少客户？"}]
})
print(result["messages"][-1].content)
```

---

## 总结

本教程通过一个 Text-to-SQL 项目，覆盖了 LangChain Agent 开发的核心知识点：

| 知识点 | 本项目中的体现 |
|--------|---------------|
| **Agent 概念** | LLM 自主决定探索数据库的步骤和顺序 |
| **Tool & Toolkit** | `SQLDatabaseToolkit` 提供 4 个数据库操作工具 |
| **System Prompt 工程** | 安全约束、行为规范、质量保证的 prompt 设计 |
| **模型接入** | 通过 OpenAI 兼容协议接入智谱 GLM-5 |
| **可观测性** | LangSmith 追踪完整推理链路 |

核心理念：**Agent = LLM + Tools + Prompt**。LLM 提供推理能力，Tools 提供行动能力，Prompt 定义行为边界。理解了这三者的关系，你就理解了 LangChain Agent 的本质。
