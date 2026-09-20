FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OPTICS_DB=/data/optics_paths.db

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# 仅经 HTTP 对外提供追迹与成像求解，无前端、无账户体系
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
