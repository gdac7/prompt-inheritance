# GAPI — Genetic Algorithm Prompt Inheritance

Substitui a etapa IEA do Prompt Inheritance (PCA/ICA/SWPCA) por um algoritmo genético sobre as sentenças dos 20 prompts similares ao pedido, guiado por uma fitness kNN calculada na base pontuada (`data/data.json`). Não há treino e o modelo alvo não é consultado durante a geração.

| Arquivo | Conteúdo |
|---|---|
| `knn_fitness.py` | F(x) = ŝ(x)/10 + λ₂·cos(x, pedido) − λ₃·max(0, cos(x, vizinho) − τ), com ŝ(x) = média das notas dos k vizinhos ponderada pelo cosseno |
| `genetic_prompt.py` | indivíduo = tupla de sentenças; torneio, elitismo, crossover de 1 ou 2 pontos, mutações de troca, remoção e pedido (GAPI-L) |
| `validate_fitness.py` | leave-one-out na base e Spearman ρ entre nota prevista e real |
| `report.py` | TSA/TSP, juiz, custos por fase, consultas, ρ fitness × realidade, deriva, diversidade, convergência, ablação e projeção 2D |

A integração com o pipeline está em `test/prompt_inheritance.py` (`--approach gapi_s|gapi_l`) e o template do sanitizador em `llm_code/prompt_manager.py` (`GAPISanitizerPrompt`).

## Configurações

| Método (chave nos resultados) | Saída |
|---|---|
| `gapi_s_top5` | Llama-3.1-8B reescreve os 5 melhores indivíduos distintos (1 prompt cada, em um único lote) |
| `gapi_s_top1x5` | Llama-3.1-8B gera 5 amostras a partir do melhor indivíduo (como o PI) |
| `gapi_l` | os 5 melhores indivíduos, com o pedido antigo trocado pelo atual; nenhum LLM |
| `gapi_l_norel`, `gapi_l_nonov` | ablação: GAPI-L com λ₂ = 0 ou λ₃ = 0 |

Hiperparâmetros (`GAConfig`): população 20, até 20 gerações (para após 5 sem melhora), torneio 3, elite 2, `p_cx` 0.8, `p_mut` 0.2, k = 10, λ₂ = 0.5, λ₃ = 1.0, τ = 0.95.

Decisões além do README do projeto:
- As notas da base são cortadas em [1, 10] (há uma nota 19.0 em `data.json`).
- Indivíduos têm no máximo 30 sentenças (o p90 da base é 23). O MiniLM só lê ~256 tokens, então sentenças além disso não mudariam F(x) e o indivíduo incharia.
- Os 5 indivíduos finais são os melhores **distintos entre todos os avaliados** (hall of fame), não só os da última população.
- No GAPI-L, uma passada final garante o pedido atual em cada um dos 5 indivíduos finais.

## Custo como no artigo

Além do pico de memória por fase, o `report.py` reproduz a métrica de custo do artigo (`test/results-sbrc/plots.ipynb`): VRAM média **reservada** (snapshots a cada 0,1 s, somada entre as GPUs) e tempo por prompt na fase que gera o prompt final, com IC de 95% sobre os pedidos. Essa fase é a sanitização no PI e no GAPI-S e a evolução no GAPI-L. A tabela inclui o AutoDAN-Turbo, recalculado com a fórmula do artigo a partir de `test/results-sbrc/` (36.891 MB = ataque + scorer), e a VRAM do juiz medida pelo serviço `judge` (`test/results_judge/scorer_performance_metrics.json`, gravado em `docker compose stop judge`). Para comparar com o artigo, rode com 2 GPUs (`GPU_0`/`GPU_1` no `.env`), como na medição original.

## Execução (de dentro de `test/`, no container)

Pontuar prompts exige o juiz no ar: `docker compose up -d --wait judge` (o serviço `gapi` já sobe o juiz sozinho).

```bash
python ../gapi/validate_fitness.py --data ../data/data.json --k 10
python prompt_inheritance.py --approach gapi_s --gapi_mode top5   --n_requests 100 --seed 0
python prompt_inheritance.py --approach gapi_s --gapi_mode top1x5 --n_requests 100 --seed 0
python prompt_inheritance.py --approach gapi_l                    --n_requests 100 --seed 0
python harmbench_eval.py --results results_gapi/*_scores.json --output_dir results_gapi/harmbench
python ../gapi/report.py --gapi_dir results_gapi --pi_results <PI> --pi_classifications <pasta PI>
```

Cada execução grava em `results_gapi/`:
- `<método>_seed<n>_generation.json`: prompts e campos do GAPI (indivíduos, fitness, convergência, custos);
- `<método>_seed<n>_scores.json`: o mesmo arquivo com as respostas do alvo e as notas do juiz (`get_new_scores`);
- `<método>_seed<n>_monitor.json`: `PerfomanceMonitor` das fases `gapi_evolution`, `gapi_sanitization` e `evaluation`.

`--skip_scoring` para depois da geração. `test/run_gapi_experiments.sh` roda o protocolo completo: 3 sementes × 5 configurações, HarmBench e relatório.
