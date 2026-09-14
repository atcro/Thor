FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 build-essential && apt-get clean
WORKDIR /app
COPY pyproject.toml README.md ./
COPY apps ./apps
COPY agents ./agents
COPY data ./data
COPY streaming ./streaming
COPY edge ./edge
RUN pip install . && pip install "pytest>=8"
EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
