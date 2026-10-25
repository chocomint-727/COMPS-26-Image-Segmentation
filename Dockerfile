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

ARG REPO_URL=https://github.com/chocomint-727/COMPS-26-Image-Segmentation.git
ARG REPO_BRANCH=main
ARG GITHUB_USER
ARG GITHUB_PAT

RUN git clone https://${GITHUB_USER}:${GITHUB_PAT}@github.com/chocomint-727/COMPS-26-Image-Segmentation.git . \
    && git remote set-url origin https://github.com/chocomint-727/COMPS-26-Image-Segmentation.git \
    && git config --global credential.helper store \
    && printf 'https://%s:%s@github.com\n' "$GITHUB_USER" "$GITHUB_PAT" > /root/.git-credentials \
    && chmod 600 /root/.git-credentials

WORKDIR /ws

RUN git clone --depth 1 --branch "${REPO_BRANCH}" "${REPO_URL}" .

RUN pip install --no-cache-dir \
    -r requirements.txt \
    -c /etc/pip/constraint.txt
