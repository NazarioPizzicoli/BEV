# docker/Dockerfile
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. Installiamo le librerie di sistema essenziali
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libgl1-mesa-glx libglib2.0-0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Installiamo UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# 3. Creiamo l'ambiente virtuale usando Python 3.11
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv venv --python 3.11
ENV PATH="/opt/venv/bin:$PATH"

# 4. Installiamo PyTorch (Layer pesante)
RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Installiamo dipendenze aggiuntive richieste (scikit-learn, seaborn)
RUN uv pip install scikit-learn seaborn

# 6. Copiamo i file minimi necessari per l'installazione delle dipendenze
COPY pyproject.toml uv.lock README.md ./

# 6. Sincronizziamo l'ambiente con tutte le extras (notebooks, dev)
# Questo crea un ambiente deterministico basato sul lockfile
RUN uv sync --all-extras --no-install-project

# 7. Copiamo il codice sorgente
COPY src/ ./src/

# 8. Installiamo il progetto in modalità editable
RUN uv sync --all-extras

# Aggiungiamo questa variabile per ignorare il file .python-version locale se presente
ENV UV_IGNORE_DOT_PYTHON_VERSION=1

EXPOSE 8888 6006

# Mantieni il container attivo per sviluppo interattivo
CMD ["tail", "-f", "/dev/null"]
