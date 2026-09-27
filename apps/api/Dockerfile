FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 build-essential && apt-get clean
WORKDIR /app
# Dependency layer: install everything pyproject declares against a stub source tree, so this
# (slow, ~10 min) layer is cached until pyproject.toml changes. The app runs from /app on
# sys.path, so the real sources only need to be copied in afterwards.
COPY pyproject.toml README.md ./
RUN mkdir -p apps agents edge streaming data && pip install . && pip install "pytest>=8"
COPY apps ./apps
COPY agents ./agents
COPY data ./data
COPY streaming ./streaming
COPY edge ./edge
EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
