# Dockerfile alternativo: usa imagen base con PyTorch CPU ya instalado
FROM pytorch/pytorch:1.13.1-cpu

WORKDIR /app

# Copiamos requirements para instalar dependencias Python (torch ya viene en la imagen)
COPY ./requirements.txt /app/requirements.txt

# Instalar dependencias del sistema necesarias y paquetes Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 ffmpeg wget && \
    pip install --upgrade pip && \
    pip install -r /app/requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copiamos el resto del proyecto
COPY . /app

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
