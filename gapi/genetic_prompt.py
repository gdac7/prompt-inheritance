"""Algoritmo genético do GAPI: evolui prompts como listas de sentenças dos prompts similares."""
import re
from dataclasses import dataclass

import numpy as np

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text):
    """Segmenta um prompt em sentenças (pontuação final ou quebra de linha).

    Args:
        text: prompt original.

    Returns:
        Lista de sentenças não vazias.
    """
    return [sentence.strip() for sentence in SENTENCE_BOUNDARY.split(text) if sentence.strip()]


def join_sentences(individual):
    """Converte um indivíduo em texto.

    Args:
        individual: tupla de sentenças.

    Returns:
        Prompt com as sentenças separadas por espaço.
    """
    return " ".join(individual)


@dataclass
class GAConfig:
    """Hiperparâmetros do GA (valores do README do GAPI)."""
    pop_size: int = 20
    generations: int = 20
    patience: int = 5
    tournament_k: int = 3
    elite: int = 2
    p_cx: float = 0.8
    p_mut: float = 0.2
    num_prompts: int = 5
    # O MiniLM só enxerga ~256 tokens; sem limite, sentenças extras não afetam F(x) e o indivíduo incha
    max_sentences: int = 30


@dataclass
class EvolutionResult:
    """Saída da evolução para um pedido."""
    individuals: list
    terms: list
    best_history: list
    mean_history: list
    generations_run: int


class PromptEvolution:
    """GA de um pedido: seleção por torneio, elitismo, crossover de blocos e mutação."""

    def __init__(self, fitness, request, similar_prompts, config, rng, insert_request):
        """Prepara a população inicial e o pool de sentenças.

        Args:
            fitness: KnnFitness.
            request: pedido malicioso atual.
            similar_prompts: prompts recuperados por find_n_similar (população inicial).
            config: GAConfig.
            rng: random.Random da execução (reprodutibilidade).
            insert_request: habilita a mutação de pedido (GAPI-L).
        """
        self.fitness = fitness
        self.request = request
        self.config = config
        self.rng = rng
        self.insert_request = insert_request
        self.request_embedding = fitness.encode([request])[0]
        sentence_lists = [split_sentences(prompt) for prompt in similar_prompts]
        self.initial_population = [self._truncate(tuple(sentences)) for sentences in sentence_lists if sentences]
        if not self.initial_population:
            raise ValueError(f"Nenhum prompt similar para o pedido: {request}")
        self.gene_pool = [sentence for individual in self.initial_population for sentence in individual]
        self.request_similarity = self._sentence_request_similarity() if insert_request else {}
        self.evaluated = {}

    def run(self):
        """Evolui a população até G gerações ou até `patience` gerações sem melhora.

        Returns:
            EvolutionResult com os melhores indivíduos distintos já avaliados.
        """
        population = self._initial_population()
        self._evaluate(population)
        best_history, mean_history = [], []
        self._record(population, best_history, mean_history)
        generations_without_improvement = 0
        generations_run = 0
        for _ in range(self.config.generations):
            population = self._next_generation(population)
            self._evaluate(population)
            previous_best = best_history[-1]
            self._record(population, best_history, mean_history)
            generations_run += 1
            if best_history[-1] > previous_best + 1e-9:
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            if generations_without_improvement >= self.config.patience:
                break
        top = self._top_distinct(self.config.num_prompts)
        return EvolutionResult(top, [self.evaluated[i] for i in top], best_history, mean_history, generations_run)

    def with_request(self, individuals):
        """Garante o pedido atual em cada indivíduo (passada final do GAPI-L).

        Args:
            individuals: indivíduos finais.

        Returns:
            Indivíduos com a sentença do pedido antigo trocada pelo pedido atual, quando ausente.
        """
        return [individual if self.request in individual else self._replace_request(individual) for individual in individuals]

    def _sentence_request_similarity(self):
        """Pré-calcula o cosseno entre cada sentença do pool e o pedido atual.

        Returns:
            Dict sentença -> cosseno (o próprio pedido vale 1.0).
        """
        sentences = list(dict.fromkeys(self.gene_pool))
        similarities = self.fitness.encode(sentences) @ self.request_embedding
        similarity = dict(zip(sentences, similarities.tolist()))
        similarity[self.request] = 1.0
        return similarity

    def _initial_population(self):
        """Usa os prompts similares e completa com cópias mutadas se faltarem indivíduos.

        Returns:
            Lista com pop_size indivíduos.
        """
        population = list(self.initial_population[: self.config.pop_size])
        while len(population) < self.config.pop_size:
            population.append(self._mutate(self.rng.choice(self.initial_population)))
        return population

    def _evaluate(self, population):
        """Calcula F(x) dos indivíduos ainda não avaliados (em lote) e guarda no arquivo.

        Args:
            population: lista de indivíduos.
        """
        new_individuals = [individual for individual in dict.fromkeys(population) if individual not in self.evaluated]
        if new_individuals:
            texts = [join_sentences(individual) for individual in new_individuals]
            self.evaluated.update(zip(new_individuals, self.fitness.evaluate(texts, self.request_embedding)))

    def _score(self, individual):
        """Retorna F(x) de um indivíduo já avaliado.

        Args:
            individual: tupla de sentenças.

        Returns:
            Valor da fitness.
        """
        return self.evaluated[individual].fitness

    def _record(self, population, best_history, mean_history):
        """Registra o melhor F(x) e o F(x) médio da geração.

        Args:
            population: população avaliada.
            best_history: lista do melhor F(x) por geração (alterada no lugar).
            mean_history: lista do F(x) médio por geração (alterada no lugar).
        """
        scores = [self._score(individual) for individual in population]
        best_history.append(max(scores))
        mean_history.append(float(np.mean(scores)))

    def _next_generation(self, population):
        """Gera a próxima população: elite intacta + filhos de torneio, crossover e mutação.

        Args:
            population: população atual (avaliada).

        Returns:
            Nova população com pop_size indivíduos.
        """
        children = sorted(population, key=self._score, reverse=True)[: self.config.elite]
        while len(children) < self.config.pop_size:
            parent_a, parent_b = self._tournament(population), self._tournament(population)
            offspring = self._crossover(parent_a, parent_b) if self.rng.random() < self.config.p_cx else (parent_a, parent_b)
            for child in offspring:
                children.append(self._mutate(child) if self.rng.random() < self.config.p_mut else child)
        return children[: self.config.pop_size]

    def _tournament(self, population):
        """Seleção por torneio.

        Args:
            population: população avaliada.

        Returns:
            Melhor de tournament_k indivíduos sorteados.
        """
        contenders = self.rng.sample(population, min(self.config.tournament_k, len(population)))
        return max(contenders, key=self._score)

    def _crossover(self, parent_a, parent_b):
        """Troca blocos de sentenças entre dois pais, com 1 ou 2 pontos de corte (50% cada).

        Args:
            parent_a: primeiro pai.
            parent_b: segundo pai.

        Returns:
            Tupla com os dois filhos (o pai correspondente substitui um filho vazio).
        """
        if self.rng.random() < 0.5:
            children = self._one_point_crossover(parent_a, parent_b)
        else:
            children = self._two_point_crossover(parent_a, parent_b)
        return tuple(self._truncate(child) if child else parent for child, parent in zip(children, (parent_a, parent_b)))

    def _one_point_crossover(self, parent_a, parent_b):
        """Crossover de 1 ponto: cada filho junta o início de um pai com o fim do outro.

        Args:
            parent_a: primeiro pai.
            parent_b: segundo pai.

        Returns:
            Tupla com os dois filhos.
        """
        cut_a, cut_b = self.rng.randint(0, len(parent_a)), self.rng.randint(0, len(parent_b))
        return parent_a[:cut_a] + parent_b[cut_b:], parent_b[:cut_b] + parent_a[cut_a:]

    def _two_point_crossover(self, parent_a, parent_b):
        """Crossover de 2 pontos: os pais trocam o bloco do meio.

        Args:
            parent_a: primeiro pai.
            parent_b: segundo pai.

        Returns:
            Tupla com os dois filhos.
        """
        start_a, end_a = sorted([self.rng.randint(0, len(parent_a)), self.rng.randint(0, len(parent_a))])
        start_b, end_b = sorted([self.rng.randint(0, len(parent_b)), self.rng.randint(0, len(parent_b))])
        return (
            parent_a[:start_a] + parent_b[start_b:end_b] + parent_a[end_a:],
            parent_b[:start_b] + parent_a[start_a:end_a] + parent_b[end_b:],
        )

    def _mutate(self, individual):
        """Aplica uma mutação sorteada entre as disponíveis para o indivíduo.

        Args:
            individual: tupla de sentenças.

        Returns:
            Indivíduo mutado.
        """
        operations = [self._swap_sentence]
        if len(individual) > 1:
            operations.append(self._remove_sentence)
        if self.insert_request:
            operations.append(self._replace_request)
        return self.rng.choice(operations)(individual)

    def _swap_sentence(self, individual):
        """Troca uma sentença por outra de qualquer prompt similar.

        Args:
            individual: tupla de sentenças.

        Returns:
            Indivíduo com uma sentença substituída.
        """
        position = self.rng.randrange(len(individual))
        return individual[:position] + (self.rng.choice(self.gene_pool),) + individual[position + 1:]

    def _remove_sentence(self, individual):
        """Remove uma sentença (favorece concisão; só é chamada com 2+ sentenças).

        Args:
            individual: tupla de sentenças.

        Returns:
            Indivíduo sem a sentença sorteada.
        """
        position = self.rng.randrange(len(individual))
        return individual[:position] + individual[position + 1:]

    def _replace_request(self, individual):
        """Troca pelo pedido atual a sentença mais similar a ele, que é a que carrega o pedido antigo.

        Args:
            individual: tupla de sentenças.

        Returns:
            Indivíduo com o pedido atual.
        """
        position = max(range(len(individual)), key=lambda i: self.request_similarity.get(individual[i], 0.0))
        return individual[:position] + (self.request,) + individual[position + 1:]

    def _truncate(self, individual):
        """Limita o indivíduo a max_sentences sentenças.

        Args:
            individual: tupla de sentenças.

        Returns:
            Indivíduo truncado.
        """
        return individual[: self.config.max_sentences]

    def _top_distinct(self, n):
        """Seleciona os n melhores indivíduos com textos distintos entre todos os já avaliados.

        Args:
            n: quantidade de indivíduos.

        Returns:
            Lista de indivíduos, do melhor para o pior.
        """
        top, seen_texts = [], set()
        for individual in sorted(self.evaluated, key=self._score, reverse=True):
            text = join_sentences(individual)
            if text not in seen_texts:
                top.append(individual)
                seen_texts.add(text)
            if len(top) == n:
                break
        return top
