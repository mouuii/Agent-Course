"""
Agents 模块 - 多 Agent 协作系统

包含：
- research_agent: 研究 Agent，专注于信息收集
- analysis_agent: 分析 Agent，专注于数据分析
- multi_agent_system: 多 Agent 协作系统
"""

from .research_agent import create_research_agent, run_research
from .analysis_agent import create_analysis_agent, run_analysis
from .multi_agent_system import (
    create_multi_agent_system,
    run_multi_agent,
    stream_multi_agent,
)

__all__ = [
    "create_research_agent",
    "run_research",
    "create_analysis_agent",
    "run_analysis",
    "create_multi_agent_system",
    "run_multi_agent",
    "stream_multi_agent",
]
