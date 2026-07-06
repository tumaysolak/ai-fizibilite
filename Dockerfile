FROM python:3.11-slim

WORKDIR /app

# Bağımlılıklar (scipy için sistem kütüphaneleri slim imajda hazır gelir)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bulut sağlayıcı PORT verir; verilmezse 8000
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
