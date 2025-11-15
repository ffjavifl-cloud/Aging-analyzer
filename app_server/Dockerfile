FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y build-essential libgl1-mesa-glx libglib2.0-0 && \
    pip install --upgrade pip && pip install -r /app/requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /app

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
