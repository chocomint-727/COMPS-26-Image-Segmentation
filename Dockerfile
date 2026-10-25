FROM nvcr.io/nvidia/pytorch:26.09-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws

COPY requirements.txt .
RUN pip install --no-cache-dir \
    -r requirements.txt \
    -c /etc/pip/constraint.txt

COPY . .