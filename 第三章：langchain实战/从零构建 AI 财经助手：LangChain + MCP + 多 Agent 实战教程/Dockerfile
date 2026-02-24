FROM python:3.13-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY pyproject.toml uv.lock ./
COPY finance_agent.py main.py ./
COPY api/ ./api/
COPY agents/ ./agents/
COPY static/ ./static/

# 安装 Python 依赖
RUN pip install --no-cache-dir -e .

# 暴露端口
EXPOSE 8000

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 启动命令
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
