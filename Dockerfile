# Dockerfile
FROM python:3.12-slim

# Install supervisor
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose both service ports
EXPOSE 8000 8502

CMD ["supervisord", "-c", "supervisord.conf"]
