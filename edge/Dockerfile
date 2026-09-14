FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install "fastapi>=0.115" "uvicorn[standard]>=0.30" "onnxruntime>=1.18" "paho-mqtt>=2.0" "numpy>=1.26" "httpx>=0.27" "pydantic>=2.7"
COPY edge ./edge
EXPOSE 8001
CMD ["uvicorn", "edge.main:app", "--host", "0.0.0.0", "--port", "8001"]
