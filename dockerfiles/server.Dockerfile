FROM python:3.10-slim

WORKDIR /app

# Copiar apenas o arquivo requirements.txt primeiro (aproveita cache)
COPY requirements.txt .

# Combinar instalação de dependências e limpeza em uma única camada
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libgtk2.0-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copiar o código do projeto
COPY . .

# Porta exposta pelo FastAPI
EXPOSE 8000

# Configurações de ambiente
ENV PYTHONPATH=/app

# Executar o servidor
CMD ["python", "run_server.py"]