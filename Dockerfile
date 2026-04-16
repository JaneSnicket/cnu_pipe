# 로컬이3.13이라. 이전버전이면거기에맞춰서수정3.11가가장안정되었다고들함
FROM python:3.13-slim
WORKDIR /app
# 의존성설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# 소스복사
COPY . .
# Render에서PORT를주고, 로컬에서는10000 기본값
ENV PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]