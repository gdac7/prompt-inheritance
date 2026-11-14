FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && pip check

# test/prompt_inheritance.py importa o juiz e o AttackGenerator de ../../autodan-itau
ARG AUTODAN_REPO=https://github.com/gdac7/autodan-turbo-implementation.git
ARG AUTODAN_REF=14679e69b34d29e2d1994d1b9a65891377ec90c4
RUN git clone "$AUTODAN_REPO" /workspace/autodan-itau \
    && git -C /workspace/autodan-itau checkout "$AUTODAN_REF" \
    && rm -rf /workspace/autodan-itau/.git

# O container roda com o UID do host, que não existe no /etc/passwd: HOME gravável e USER
# definido (getpass.getuser, usado pelo torch, consulta USER antes do passwd)
ENV HOME=/tmp \
    USER=pi \
    HF_HOME=/hf-cache \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/prompt-inheritance/test
CMD ["bash"]
