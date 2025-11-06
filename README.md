### Proposta

- 1: Gerar um conjunto de N prompts semelhantes baseado na entrada de um usuário
- 2: Calcular o embedding de cada prompt inteirou usando um modelo (ex: SentenceTransformer)
- 3: Calcular o vetor médio (centroide) desses embeddings $C = \frac{1}{N}\sum_{i=1}^{n}E_{i}$
- 4: Análise de Componentes Principais (PCA) ou 
- 5: Transformar o novo vetor (centroide mofificado) de volta em tokens
- 6: Sanitização: Pedir a um LLM para "fazer sentido"

### Quebra de passos
- No passo 4:
  - Assumindo que completamos 1, 2 e 3:
    - Temos N prompts semelhantes
    - Temos N embeddings
    - Temos o centroide C
  - Fazemos:
    - Centralizamos os dados
      - Subtrair a centroide de cada um dos seus embeddings: $E_{i}^{'} = E_{i} - C$
    - Aplicar Principal Component Analysis (PCA)/ Singular Value Decomposition (SVD)
      - Teremos os componentes principais: $v_{1}, v_{2}, ..., v_{k}$
      - $v_{1}$ aponta para onde os prompts mais variam
      - $v_{2}$ aponta para a segunda maior direção...
- No passo 5
  - Geramos um $C_{novo}$ navegando a partir do centro
    - $C_{novo} = C + (\alpha v_{1}) + (\beta v_{2})$
      - $\alpha$ é o peso que escolhemos
        - Exemplo: 1.5 um prompt muito focado no que o v1 representa, -1.5 estamos indo na direção oposta
- Passo 6
  - Usar distância de cosseno para transformar $C_{novo}$ em palavras (tokens)
    - Teremos um **Bag-of-Words** como resultado
- Passo 7
  - Passa o saco de palavras para o LLM e pede para criar um prompt usando as palavras-chave 
  
### Resultado
- Um prompt novo, coerente e semanticamente relacionado ao cluster original, mas que explora uma variação específica que controlamos com alpha

### Próximos passos
- Testar com ICA
  - Irá encontrar componentes independentes
- Testar com LCA + PCA
- Testar com filtro de perplexidade





### Resultados para colher
- Verificar se o score do novo prompt está próximo da média de score dos prompts base -> gráfico
- Verificar se, se nos basearmos em prompts que eram jailbreak, o novo prompt ainda é jailbreak
- Verificar nova taxa de jailbreak para comparar com a taxa base do ADT
- Pertubação no alpha testando por um filtro de perplexidade
- Alteração em score e jailbreak de acordo com pertubações no alpha e no k do top_k_scores
- SGD usando função de custo a perplexidade
- Simulated Annealing com custo de perplexidade