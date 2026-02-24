"""
多 Agent 协作系统
使用 LangGraph 的 StateGraph 协调研究和分析 Agents
"""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import operator

from agents.research_agent import create_research_agent
from agents.analysis_agent import create_analysis_agent

# ============================================================
# 状态定义
# ============================================================


class AgentState(TypedDict):
    """多 Agent 系统的共享状态

    Attributes:
        messages: 消息历史（累加）
        query: 用户原始查询
        research_result: 研究 Agent 的结果
        analysis_result: 分析 Agent 的结果
        final_report: 最终综合报告
        next_step: 下一步要执行的节点
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    research_result: str
    analysis_result: str
    final_report: str
    next_step: str


# ============================================================
# Agent 节点函数
# ============================================================


def route_query(state: AgentState) -> AgentState:
    """路由节点：决定查询类型和执行策略

    分析用户查询，决定需要调用哪些 Agent：
    - 只需要数据分析 -> 仅调用分析 Agent
    - 只需要新闻信息 -> 仅调用研究 Agent
    - 需要综合分析 -> 调用两个 Agent
    """
    query = state["query"].lower()

    # 判断是否需要新闻/市场动态
    needs_research = any(kw in query for kw in [
        "新闻", "消息", "动态", "报道", "评级",
        "分析师", "市场", "情绪", "舆情"
    ])

    # 判断是否需要数据分析
    needs_analysis = any(kw in query for kw in [
        "财报", "财务", "估值", "市盈率", "市值",
        "对比", "比较", "分析", "指标", "数据",
        "股价", "价格", "涨跌"
    ]) or not needs_research  # 默认需要分析

    # 决定下一步
    if needs_research and needs_analysis:
        next_step = "parallel"  # 并行执行
    elif needs_research:
        next_step = "research_only"
    else:
        next_step = "analysis_only"

    state["next_step"] = next_step
    state["messages"] = [HumanMessage(content=f"[路由决策] 执行策略: {next_step}")]

    return state


def research_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """研究节点：收集新闻和市场信息"""
    agent = create_research_agent(llm)

    # 构造研究任务
    research_query = f"""
请收集以下内容的最新信息：
{state['query']}

重点关注：
1. 最新新闻和媒体报道
2. 分析师评级和目标价
3. 市场情绪和舆论
4. 行业动态

请提供详细的信息来源。
"""

    result = agent.invoke({"messages": [HumanMessage(content=research_query)]})
    research_result = result["messages"][-1].content

    state["research_result"] = research_result
    state["messages"] = [AIMessage(content=f"[研究完成]\n{research_result}")]

    return state


def analysis_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """分析节点：进行数据分析"""
    agent = create_analysis_agent(llm)

    # 构造分析任务
    analysis_query = f"""
请对以下内容进行深度数据分析：
{state['query']}

重点分析：
1. 核心财务指标和估值
2. 盈利能力和成长性
3. 财务健康度和风险
4. 横向对比（如果涉及多只股票）

请提供量化数据支撑。
"""

    result = agent.invoke({"messages": [HumanMessage(content=analysis_query)]})
    analysis_result = result["messages"][-1].content

    state["analysis_result"] = analysis_result
    state["messages"] = [AIMessage(content=f"[分析完成]\n{analysis_result}")]

    return state


def synthesize_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """综合节点：整合研究和分析结果，生成最终报告"""

    # 构建综合提示
    synthesis_parts = [
        f"用户原始查询：{state['query']}\n",
    ]

    if state.get("research_result"):
        synthesis_parts.append(f"## 研究结果\n{state['research_result']}\n")

    if state.get("analysis_result"):
        synthesis_parts.append(f"## 分析结果\n{state['analysis_result']}\n")

    synthesis_prompt = "\n".join(synthesis_parts) + """

请基于以上研究和分析结果，生成一份专业的综合报告。

报告要求：
1. **清晰的结构**：使用标题和小标题组织内容
2. **数据支撑**：引用具体的数字和指标
3. **信息来源**：标注关键信息的来源
4. **综合观点**：结合市场动态和财务数据给出整体评估
5. **风险提示**：指出关键风险和不确定性

请用专业且易懂的中文撰写。
"""

    messages = [HumanMessage(content=synthesis_prompt)]
    response = llm.invoke(messages)
    final_report = response.content

    state["final_report"] = final_report
    state["messages"] = [AIMessage(content=final_report)]

    return state


# ============================================================
# 构建工作流
# ============================================================


def create_multi_agent_system(llm: ChatOpenAI):
    """创建多 Agent 协作系统

    工作流程：
    1. route_query: 分析查询类型
    2. research/analysis: 并行或串行执行
    3. synthesize: 综合结果

    Args:
        llm: 语言模型实例

    Returns:
        编译好的工作流
    """

    # 创建状态图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("route", route_query)
    workflow.add_node("research", lambda state: research_node(state, llm))
    workflow.add_node("analysis", lambda state: analysis_node(state, llm))
    workflow.add_node("synthesize", lambda state: synthesize_node(state, llm))

    # 设置入口
    workflow.set_entry_point("route")

    # 条件路由：根据 route 结果决定执行路径
    def route_decision(state: AgentState) -> str:
        """根据路由结果决定下一步"""
        next_step = state.get("next_step", "parallel")

        if next_step == "research_only":
            return "research_only"
        elif next_step == "analysis_only":
            return "analysis_only"
        else:
            return "parallel"

    workflow.add_conditional_edges(
        "route",
        route_decision,
        {
            "research_only": "research",
            "analysis_only": "analysis",
            "parallel": "research",  # 先执行研究
        }
    )

    # 连接边
    # 研究完成后的路径
    def after_research(state: AgentState) -> str:
        if state.get("next_step") == "parallel":
            return "to_analysis"
        else:
            return "to_synthesize"

    workflow.add_conditional_edges(
        "research",
        after_research,
        {
            "to_analysis": "analysis",
            "to_synthesize": "synthesize",
        }
    )

    # 分析完成后的路径
    workflow.add_edge("analysis", "synthesize")

    # 综合完成后结束
    workflow.add_edge("synthesize", END)

    # 编译工作流
    return workflow.compile()


# ============================================================
# 简化的运行接口
# ============================================================


def run_multi_agent(llm: ChatOpenAI, query: str) -> dict:
    """运行多 Agent 系统

    Args:
        llm: 语言模型实例
        query: 用户查询

    Returns:
        包含所有结果的字典
    """
    system = create_multi_agent_system(llm)

    # 初始状态
    initial_state = {
        "messages": [],
        "query": query,
        "research_result": "",
        "analysis_result": "",
        "final_report": "",
        "next_step": "",
    }

    # 运行工作流
    result = system.invoke(initial_state)

    return {
        "query": query,
        "research_result": result.get("research_result", ""),
        "analysis_result": result.get("analysis_result", ""),
        "final_report": result.get("final_report", ""),
        "execution_path": result.get("next_step", ""),
    }


def stream_multi_agent(llm: ChatOpenAI, query: str):
    """流式运行多 Agent 系统

    Args:
        llm: 语言模型实例
        query: 用户查询

    Yields:
        工作流执行过程中的事件
    """
    system = create_multi_agent_system(llm)

    initial_state = {
        "messages": [],
        "query": query,
        "research_result": "",
        "analysis_result": "",
        "final_report": "",
        "next_step": "",
    }

    # 流式执行
    for event in system.stream(initial_state):
        yield event
