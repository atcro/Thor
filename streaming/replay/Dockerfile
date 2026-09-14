FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install "paho-mqtt>=2.0" "pandas>=2.2" "numpy>=1.26" "pyarrow>=15" "httpx>=0.27" "pydantic>=2.7"
COPY data ./data
COPY streaming ./streaming
CMD ["python", "-m", "streaming.replay.replay"]
