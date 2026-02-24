"""
多 Agent 系统使用示例
演示如何使用研究 Agent、分析 Agent 和多 Agent 协作系统
"""

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI

from agents.research_agent import run_research
from agents.analysis_agent import run_analysis
from agents.multi_agent_system import run_multi_agent, stream_multi_agent

# ============================================================
# 配置
# ============================================================

# 智谱 GLM 配置
ZHIPU_API_KEY = "ccb9818e987149dd8cc2541ff7c9df57.AMe31g7tJHwgPo9q"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL_NAME = "glm-4-flash"

llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=ZHIPU_API_KEY,
    openai_api_base=ZHIPU_BASE_URL,
    temperature=0.3,
)


# ============================================================
# 示例 1：单独使用研究 Agent
# ============================================================


def example_research_agent():
    """示例：使用研究 Agent 收集信息"""
    print("=" * 60)
    print("示例 1：研究 Agent - 收集市场信息")
    print("=" * 60)

    query = "苹果公司最近有什么新闻？分析师怎么看？"
    print(f"\n查询：{query}\n")

    result = run_research(llm, query)
    print(result)
    print()


# ============================================================
# 示例 2：单独使用分析 Agent
# ============================================================


def example_analysis_agent():
    """示例：使用分析 Agent 进行数据分析"""
    print("=" * 60)
    print("示例 2：分析 Agent - 财务数据分析")
    print("=" * 60)

    query = "分析 AAPL 的估值水平和财务指标"
    print(f"\n查询：{query}\n")

    result = run_analysis(llm, query)
    print(result)
    print()


# ============================================================
# 示例 3：多 Agent 协作 - 综合分析
# ============================================================


def example_multi_agent_comprehensive():
    """示例：多 Agent 协作进行综合分析"""
    print("=" * 60)
    print("示例 3：多 Agent 系统 - 综合分析")
    print("=" * 60)

    query = "全面分析苹果公司(AAPL)的投资价值，包括市场动态和财务状况"
    print(f"\n查询：{query}\n")

    result = run_multi_agent(llm, query)

    print("\n" + "─" * 60)
    print("执行路径:", result["execution_path"])
    print("─" * 60)

    if result["research_result"]:
        print("\n【研究结果】")
        print(result["research_result"])

    if result["analysis_result"]:
        print("\n【分析结果】")
        print(result["analysis_result"])

    print("\n【综合报告】")
    print(result["final_report"])
    print()


# ============================================================
# 示例 4：多 Agent 协作 - 股票对比
# ============================================================


def example_multi_agent_comparison():
    """示例：多 Agent 协作进行股票对比"""
    print("=" * 60)
    print("示例 4：多 Agent 系统 - 股票对比分析")
    print("=" * 60)

    query = "对比苹果(AAPL)和微软(MSFT)，哪个更值得投资？"
    print(f"\n查询：{query}\n")

    result = run_multi_agent(llm, query)

    print("\n【执行路径】", result["execution_path"])
    print("\n【最终报告】")
    print(result["final_report"])
    print()


# ============================================================
# 示例 5：流式执行多 Agent 系统
# ============================================================


def example_streaming():
    """示例：流式运行多 Agent 系统"""
    print("=" * 60)
    print("示例 5：流式执行多 Agent 系统")
    print("=" * 60)

    query = "特斯拉(TSLA)最近表现如何？"
    print(f"\n查询：{query}\n")

    print("执行过程：")
    print("-" * 60)

    for event in stream_multi_agent(llm, query):
        # 打印每个节点的执行情况
        for node_name, node_state in event.items():
            print(f"\n[{node_name}] 执行完成")
            if "final_report" in node_state and node_state["final_report"]:
                print("\n【最终报告】")
                print(node_state["final_report"])

    print()


# ============================================================
# 示例 6：交互式多 Agent 系统
# ============================================================


def interactive_mode():
    """交互式模式：持续接受用户输入"""
    print("=" * 60)
    print("  多 Agent 财经分析系统")
    print("  - 研究 Agent：收集新闻、评级、市场动态")
    print("  - 分析 Agent：分析财务数据、估值指标")
    print("  - 协作系统：智能路由，综合分析")
    print()
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    print()

    while True:
        try:
            query = input("请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        print("\n" + "─" * 60)

        try:
            result = run_multi_agent(llm, query)

            print(f"执行策略: {result['execution_path']}")
            print("─" * 60)

            # 显示最终报告
            print("\n" + result["final_report"])

        except Exception as e:
            print(f"\n处理出错: {e}")

        print("\n" + "=" * 60 + "\n")


# ============================================================
# 主程序
# ============================================================


def main():
    """运行所有示例"""
    print("\n🚀 多 Agent 系统示例演示\n")

    # 选择运行模式
    print("请选择运行模式：")
    print("1. 示例 1：研究 Agent")
    print("2. 示例 2：分析 Agent")
    print("3. 示例 3：多 Agent 综合分析")
    print("4. 示例 4：多 Agent 股票对比")
    print("5. 示例 5：流式执行")
    print("6. 交互式模式")
    print("7. 运行所有示例")
    print()

    choice = input("请输入选项 (1-7): ").strip()

    if choice == "1":
        example_research_agent()
    elif choice == "2":
        example_analysis_agent()
    elif choice == "3":
        example_multi_agent_comprehensive()
    elif choice == "4":
        example_multi_agent_comparison()
    elif choice == "5":
        example_streaming()
    elif choice == "6":
        interactive_mode()
    elif choice == "7":
        example_research_agent()
        example_analysis_agent()
        example_multi_agent_comprehensive()
        example_multi_agent_comparison()
        example_streaming()
    else:
        print("无效选项，退出")


if __name__ == "__main__":
    main()
