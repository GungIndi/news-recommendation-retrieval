FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    libopenblas-dev \
    gfortran \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
