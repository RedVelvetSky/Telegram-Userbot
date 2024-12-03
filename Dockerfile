FROM python:3.10-slim
LABEL authors="Space Dimension"

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app

RUN mkdir -p /app/generated

ENTRYPOINT ["python", "app/main.py"]