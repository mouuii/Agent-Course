.PHONY: help install dev run test clean docker

help:  ## 显示帮助信息
	@echo "可用命令:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## 安装依赖
	uv sync

dev:  ## 开发模式运行 Web 服务
	.venv/bin/python -m uvicorn api.server_with_mcp:app --reload --port 8000

run:  ## 生产模式运行 Web 服务
	.venv/bin/python -m uvicorn api.server_with_mcp:app --host 0.0.0.0 --port 8000

cli:  ## 运行命令行 Agent
	python finance_agent.py

multi-agent:  ## 运行多 Agent 示例
	python agents/example_multi_agent.py

mcp:  ## 启动 MCP Server
	python agents/mcp_server.py

test:  ## 运行测试
	python api/test_api.py
	python agents/test_system.py

test-api:  ## 测试 API
	python api/test_api.py

test-mcp:  ## 测试 MCP
	python test_mcp_tools.py

clean:  ## 清理缓存文件
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:  ## 构建 Docker 镜像
	docker build -t finance-agent .

docker-run:  ## 运行 Docker 容器
	docker run -p 8000:8000 -e ZHIPU_API_KEY=$$ZHIPU_API_KEY finance-agent

docker-stop:  ## 停止 Docker 容器
	docker stop $$(docker ps -q --filter ancestor=finance-agent)
