# Prompt Inheritance

## Execução com Docker

A imagem traz só as dependências (incluindo o `autodan-turbo-implementation`, de onde vêm o juiz remoto e o `AttackGenerator`). O repositório entra por bind mount: o código e os resultados ficam no host, e mudanças no código não exigem rebuild.

Pré-requisitos no host: Docker com Compose v2, driver NVIDIA e [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), Git LFS e acesso de rede ao juiz (`SCORER_ADDRESS` em `test/prompt_inheritance.py`).

```bash
git lfs pull                       # test/results_100_requests/get_approaches_results.json está no LFS
cp .env.example .env               # preencha HF_TOKEN e confira HF_CACHE, HOST_UID e HOST_GID
mkdir -p ~/.cache/huggingface      # pasta do HF_CACHE (evita que o Docker a crie como root)
docker compose build
docker compose run --rm pi         # shell dentro do container, já em test/
```

Dentro do container:

```bash
# respostas do alvo + notas do juiz para as gerações dos 100 pedidos
python prompt_inheritance.py                   # alvo: Llama-3.1-8B-Instruct
python prompt_inheritance.py --multipletargets # alvos da lista em __main__

# TSA e TSP com o classificador do HarmBench (Llama-2-13B, 4 bits)
python harmbench_eval.py \
    --results results_100_requests/get_new_scores_results.json \
    --output_dir results_100_requests/harmbench
```

`harmbench_eval.py` grava, por método, `<prefixo>_classifications.json`, `<prefixo>_tsa.txt` e `<prefixo>_tsp.txt` (mesmo formato de `test/results-sbrc/`) em `<output_dir>/<arquivo>/<alvo>/`. Use `--target` para outro alvo e `--methods` para restringir os métodos.
