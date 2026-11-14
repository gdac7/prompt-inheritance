FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && pip check

# O juiz do artigo (Gemma-7B) carrega com flash-attention 2; wheel pronta para torch 2.9 + CUDA 12 + Python 3.12
RUN pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# test/prompt_inheritance.py importa o juiz e o AttackGenerator de ../../autodan-itau
ARG AUTODAN_REPO=https://github.com/gdac7/autodan-turbo-implementation.git
ARG AUTODAN_REF=14679e69b34d29e2d1994d1b9a65891377ec90c4
RUN git clone "$AUTODAN_REPO" /workspace/autodan-itau \
    && git -C /workspace/autodan-itau checkout "$AUTODAN_REF" \
    && rm -rf /workspace/autodan-itau/.git

# Servidor do juiz instrumentado com o PerfomanceMonitor (branch scorer-stable), o mesmo que mediu a VRAM do scorer no artigo
ARG SCORER_REF=d6437b45c7ae191140e2c3da4f94c34863169186
RUN git clone "$AUTODAN_REPO" /workspace/autodan-scorer \
    && git -C /workspace/autodan-scorer checkout "$SCORER_REF" \
    && rm -rf /workspace/autodan-scorer/.git /workspace/autodan-scorer/server_metrics

# O container roda com o UID do host, que não existe no /etc/passwd: HOME gravável e USER
# definido (getpass.getuser, usado pelo torch, consulta USER antes do passwd)
ENV HOME=/tmp \
    USER=pi \
    HF_HOME=/hf-cache \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/prompt-inheritance/test
CMD ["bash"]
