FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
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
    && rm -rf /var/lib/apt/lists/*

RUN cd ..

# Copiar requirements
COPY requirements.txt .

# Instalar dependências
RUN pip install -r requirements.txt

# Copiar o código do projeto
COPY . .

# Porta exposta pelo FastAPI
EXPOSE 8000

# Configurações de ambiente
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/config/default.yaml

# Executar o servidor
CMD ["python", "main_server.py"]