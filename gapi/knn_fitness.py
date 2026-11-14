"""Fitness substituta do GAPI: regressão kNN sobre a base pontuada, sem treino.

F(x) = ŝ(x)/10 + λ₂·cos(e_x, e_pedido) − λ₃·max(0, cos(e_x, e_viz) − τ)
"""
from dataclasses import asdict, dataclass

import numpy as np

MIN_SCORE = 1.0
MAX_SCORE = 10.0


@dataclass
class FitnessTerms:
    """Valor de F(x) e os termos que o compõem."""
    fitness: float
    predicted_score: float
    relevance: float
    novelty_penalty: float
    nearest_similarity: float

    def to_dict(self):
        """Converte os termos em dict serializável em JSON.

        Returns:
            Dict com os campos da fitness.
        """
        return asdict(self)


def knn_predict(similarities, scores, k):
    """Prevê notas pela média das notas dos k vizinhos, ponderada pelo cosseno.

    Args:
        similarities: matriz (n_consultas, n_base) de cossenos; -inf exclui a entrada.
        scores: notas da base, shape (n_base,).
        k: número de vizinhos.

    Returns:
        Tupla (notas previstas, cosseno do vizinho mais próximo), ambas com shape (n_consultas,).
    """
    neighbors = np.argpartition(-similarities, k - 1, axis=1)[:, :k]
    neighbor_similarities = np.take_along_axis(similarities, neighbors, axis=1)
    weights = np.clip(neighbor_similarities, 1e-6, None)
    predicted = (weights * scores[neighbors]).sum(axis=1) / weights.sum(axis=1)
    return predicted, neighbor_similarities.max(axis=1)


class KnnFitness:
    """Avalia F(x) usando toda a base pontuada (data.json) como referência."""

    def __init__(self, sentence_model, data, k=10, lambda_rel=0.5, lambda_nov=1.0, tau=0.95):
        """Codifica a base uma única vez para reaproveitar em todas as avaliações.

        Args:
            sentence_model: SentenceTransformer (all-MiniLM-L6-v2).
            data: itens de data.json com "attack_prompt" e "score".
            k: vizinhos da regressão kNN.
            lambda_rel: peso λ₂ da relevância ao pedido.
            lambda_nov: peso λ₃ da penalidade de novidade.
            tau: limiar τ de similaridade acima do qual a novidade é penalizada.
        """
        self.sentence_model = sentence_model
        self.k = k
        self.lambda_rel = lambda_rel
        self.lambda_nov = lambda_nov
        self.tau = tau
        # A base tem notas fora da escala 1–10 (ex.: 19.0)
        self.base_scores = np.clip([float(item["score"]) for item in data], MIN_SCORE, MAX_SCORE)
        self.base_embeddings = self.encode([item["attack_prompt"] for item in data])

    def encode(self, texts):
        """Gera embeddings normalizados, para que o produto interno seja o cosseno.

        Args:
            texts: lista de textos.

        Returns:
            Matriz (len(texts), dim) de embeddings.
        """
        return self.sentence_model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True
        )

    def evaluate(self, texts, request_embedding):
        """Calcula F(x) e seus termos para cada texto.

        Args:
            texts: prompts candidatos.
            request_embedding: embedding normalizado do pedido atual.

        Returns:
            Lista de FitnessTerms, na ordem de texts.
        """
        embeddings = self.encode(texts)
        predicted, nearest = knn_predict(embeddings @ self.base_embeddings.T, self.base_scores, self.k)
        relevance = embeddings @ request_embedding
        novelty_penalty = np.maximum(0.0, nearest - self.tau)
        fitness = predicted / MAX_SCORE + self.lambda_rel * relevance - self.lambda_nov * novelty_penalty
        return [
            FitnessTerms(float(f), float(p), float(r), float(n), float(s))
            for f, p, r, n, s in zip(fitness, predicted, relevance, novelty_penalty, nearest)
        ]
