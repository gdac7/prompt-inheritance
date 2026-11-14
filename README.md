# Prompt Inheritance

## Execução com Docker

A imagem traz só as dependências, incluindo dois checkouts do `autodan-turbo-implementation`: o `AttackGenerator` (rubrica do juiz e leitura da nota), importado por `test/prompt_inheritance.py`, e o servidor do juiz (Gemma-7B) instrumentado com o `PerfomanceMonitor` (branch `scorer-stable`), o mesmo que mediu a VRAM do scorer no artigo. O repositório entra por bind mount: o código e os resultados ficam no host, e mudanças no código não exigem rebuild.

Pré-requisitos no host: Docker com Compose v2, driver NVIDIA e [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) e Git LFS.

```bash
git lfs pull                       # test/results_100_requests/get_approaches_results.json está no LFS
cp .env.example .env               # HF_TOKEN, HF_CACHE, HOST_UID/HOST_GID (0 e 0 no Docker rootless) e GPUs
mkdir -p ~/.cache/huggingface      # pasta do HF_CACHE (evita que o Docker a crie como root)
docker compose build
```

### Juiz

O serviço `judge` sobe o juiz em `127.0.0.1:8001`, na GPU `JUDGE_GPU`. Os serviços `pi` e `gapi` o acessam por `SCORER_ADDRESS` (padrão `127.0.0.1`; para um juiz em outra máquina, troque o IP no `.env`).

```bash
docker compose up -d --wait judge  # espera o Gemma carregar (o primeiro start baixa ~17 GB)
# ... execuções que pontuam prompts ...
docker compose stop judge          # ao parar, grava test/results_judge/scorer_performance_metrics.json
```

A VRAM do scorer só é gravada quando o juiz para. Ela acumula todas as chamadas desde que ele subiu (operação `scorer_generation`).

### Execuções

```bash
docker compose run --rm pi         # shell dentro do container, já em test/
```

Dentro do container:

```bash
# respostas do alvo + notas do juiz para as gerações dos 100 pedidos (precisa do juiz no ar)
python prompt_inheritance.py                   # alvo: Llama-3.1-8B-Instruct
python prompt_inheritance.py --multipletargets # alvos da lista em __main__

# TSA e TSP com o classificador do HarmBench (Llama-2-13B, 4 bits)
python harmbench_eval.py \
    --results results_100_requests/get_new_scores_results.json \
    --output_dir results_100_requests/harmbench
```

`harmbench_eval.py` grava, por método, `<prefixo>_classifications.json`, `<prefixo>_tsa.txt` e `<prefixo>_tsp.txt` (mesmo formato de `test/results-sbrc/`) em `<output_dir>/<arquivo>/<alvo>/`. Use `--target` para outro alvo e `--methods` para restringir os métodos.

O serviço `pi` usa as GPUs `GPU_0` e `GPU_1` (duas, como na medição de custo do artigo).
