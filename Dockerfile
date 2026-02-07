# Dockerfile for ST-GCN Crime Prediction System

FROM python:3.9-slim

WORKDIR /app

# Criar usuário não-root por segurança
RUN useradd -m -u 1000 appuser

# Instalar dependências de runtime (não-dev)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements para instalar dependências Python
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY --chown=appuser:appuser . .

# Criar diretórios necessários
RUN mkdir -p /app/logs /app/models /app/data && \
    chown -R appuser:appuser /app

# Configurar variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
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
CMD ["python", "app.py"]
