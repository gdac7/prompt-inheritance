"""Valida a fitness kNN antes do GA: prevê a nota de cada prompt da base pelos seus k
vizinhos (excluindo ele mesmo) e mede a correlação de Spearman com a nota real.

Uso (de dentro de test/): python ../gapi/validate_fitness.py --data ../data/data.json --k 10
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

from gapi.knn_fitness import KnnFitness, knn_predict
from utils import load_config


def evaluate_predictions(similarities, scores, k, excluded):
    """Prevê as notas com os vizinhos permitidos e compara com as notas reais.

    Args:
        similarities: matriz (n, n) de cossenos entre os prompts da base.
        scores: notas reais da base, shape (n,).
        k: número de vizinhos.
        excluded: máscara booleana (n, n) dos vizinhos proibidos para cada prompt.

    Returns:
        Dict com Spearman ρ, p-valor, MAE e número de prompts.
    """
    predicted, _ = knn_predict(np.where(excluded, -np.inf, similarities), scores, k)
    rho, p_value = spearmanr(predicted, scores)
    return {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "mae": float(np.abs(predicted - scores).mean()),
        "n": int(len(scores)),
    }


def parse_args():
    """Lê os argumentos da linha de comando.

    Returns:
        argparse.Namespace com data, k e output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="../data/data.json", help="Base pontuada (data.json)")
    parser.add_argument("--k", type=int, default=10, help="Vizinhos da regressão kNN")
    parser.add_argument("--output", default="results_gapi/fitness_validation.json", help="JSON de saída")
    return parser.parse_args()


def main():
    """Roda o leave-one-out e o leave-one-request-out e salva as métricas."""
    args = parse_args()
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)
    config = load_config(os.path.join(REPO_ROOT, "config", "models.yaml"))
    fitness = KnnFitness(SentenceTransformer(config["models"]["embedding"]), data, k=args.k)
    similarities = fitness.base_embeddings @ fitness.base_embeddings.T
    requests = np.array([item["malicious_request"] for item in data])
    results = {
        "k": args.k,
        # LOO do README: só o próprio prompt é excluído (outros prompts do mesmo pedido ficam)
        "leave_one_out": evaluate_predictions(similarities, fitness.base_scores, args.k, np.eye(len(data), dtype=bool)),
        # Variante mais rígida: nenhum prompt do mesmo pedido pode ser vizinho
        "leave_one_request_out": evaluate_predictions(
            similarities, fitness.base_scores, args.k, requests[:, None] == requests[None, :]
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
