# Dockerfile seguro para Render: usa python:3.10-slim y instala PyTorch CPU desde los wheels oficiales
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Copiar requirements (sin torch)
COPY ./requirements.txt /app/requirements.txt

# Dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 ffmpeg wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Actualizar pip
RUN pip install --upgrade pip

# Instalar PyTorch + torchvision CPU desde los wheels oficiales (evita compilar)
RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Instalar el resto de dependencias (facenet-pytorch, scikit-image, etc.)
RUN pip install -r /app/requirements.txt

# Copiar el resto de la app
COPY . /app

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
