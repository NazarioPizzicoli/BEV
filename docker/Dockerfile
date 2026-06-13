FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Installiamo le librerie di sistema essenziali (incluso curl per uv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libgl1-mesa-glx libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 2. Copiamo i binari ufficiali di UV all'interno del container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Creiamo l'ambiente virtuale usando Python 3.11 gestito da UV
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv venv --python 3.11
ENV PATH="/opt/venv/bin:$PATH"

# 4. Copiamo il tuo requirements.txt (grazie al context superiore, Docker lo troverà)
COPY docker/requirements.txt .

# 5. UV installa PyTorch (ottimizzato CUDA) e poi il resto del tuo file
RUN uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install -r requirements.txt

EXPOSE 8888

# Comando per avviare Jupyter Lab
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
