# Multi-stage Dockerfile for ST-GCN Crime Prediction System
# Build stage e production stage separados

# Stage 1: Builder - Instala dependências e prepara ambiente
FROM python:3.9-slim as builder

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python em diretório separado
RUN pip install --user --no-cache-dir -r requirements.txt


# Stage 2: Runtime - Imagem final leve
FROM python:3.9-slim

WORKDIR /app

# Criar usuário não-root por segurança
RUN useradd -m -u 1000 appuser

# Instalar dependências de runtime (não-dev)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar pacotes do builder
COPY --from=builder /root/.local /home/appuser/.local

# Copiar código da aplicação
COPY --chown=appuser:appuser . .

# Criar diretórios necessários
RUN mkdir -p /app/logs /app/models /app/data && \
    chown -R appuser:appuser /app

# Configurar PATH
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app.py

# Trocar para usuário não-root
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expor porta
EXPOSE 5000

# Comando de inicialização
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
