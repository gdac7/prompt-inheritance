#!/usr/bin/env bash
# Protocolo completo do GAPI (rodar de test/, dentro do container).
# Execuções com *_scores.json já existente são puladas, então o script pode ser retomado.
# Opcional: PI_RESULTS e PI_CLASSIFICATIONS (caminhos a partir de test/) para comparar com o PI.
set -euo pipefail

SEEDS="${SEEDS:-0 1 2}"
N_REQUESTS="${N_REQUESTS:-100}"
OUT="${OUT:-results_gapi}"

# $1 = nome do método (como em gapi_method_name); demais argumentos vão para prompt_inheritance.py
run_gapi() {
    local method="$1"
    shift
    if [[ -f "$OUT/${method}_seed${seed}_scores.json" ]]; then
        echo "Pulando ${method}_seed${seed} (já pontuado)"
        return
    fi
    python prompt_inheritance.py "$@" --n_requests "$N_REQUESTS" --seed "$seed" --output_dir "$OUT"
}

mkdir -p "$OUT"
python ../gapi/validate_fitness.py --data ../data/data.json --k 10 --output "$OUT/fitness_validation.json"

for seed in $SEEDS; do
    run_gapi gapi_s_top5   --approach gapi_s --gapi_mode top5
    run_gapi gapi_s_top1x5 --approach gapi_s --gapi_mode top1x5
    run_gapi gapi_l        --approach gapi_l
    run_gapi gapi_l_norel  --approach gapi_l --lambda_rel 0
    run_gapi gapi_l_nonov  --approach gapi_l --lambda_nov 0
done

python harmbench_eval.py --results "$OUT"/*_scores.json --output_dir "$OUT/harmbench"

report_args=(--gapi_dir "$OUT" --output_dir "$OUT/report")
[[ -n "${PI_RESULTS:-}" ]] && report_args+=(--pi_results "$PI_RESULTS")
[[ -n "${PI_CLASSIFICATIONS:-}" ]] && report_args+=(--pi_classifications "$PI_CLASSIFICATIONS")
python ../gapi/report.py "${report_args[@]}"
