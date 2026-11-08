# Sandbox image: everything that runs INSIDE gVisor. Build from the repo root:
#   docker build -f worker/Dockerfile -t autoresearch-worker:latest .
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY worker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r /tmp/requirements.txt PyOpenGL
WORKDIR /app
COPY control/__init__.py control/models.py /app/control/
COPY worker /app/worker
RUN useradd -u 1000 -m runner && python -c "import gymnasium, mujoco, torch, sklearn; print('ok')"
USER 1000
ENV PYTHONUNBUFFERED=1 HOME=/tmp MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
