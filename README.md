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
  - Irá fazer uma espécie de feature engineering antes
  - Para cada um dos $N_{i}$ scores nos N prompts, pegaremos o score original e binarizamos (> 8.0 -> 1) senão 0
  - Treinaremos o LCA nessas features 0/1. O LCA dividirá nossos N prompts em 2 classes
  - Identificaremos qual é a classe referente ao sucesso, que tem maioria 1 e filtramos nosso cluster embeddings mantendo apenas os que pertencem a classe elite
  - Rodaremos a PIPELINE do PCA original apenas no novo cluster de elite
- Testar com simulated annealing usando o custo uma função de perplexidade
  - Partindo de prompts com um score bom, se deixarmos mais legivel melhora? E se deixarmos menos legivel?
    - Basicamente, se andarmos no eixo da perplexidade o que acontece no eixo de Attack Sucess Rate?
  - Pegaremos o prompt que o LCA + PCA gerou e usaremos SA para maximizar a perplexidade
- Testar com SGD também usando uma função de perplexidade
  - Single Token Optimizaiton (STO)
  - Preliminary Selection
  - Fine Selection
- Testar pertubações no alpha
  - Se melhora ou piora os resultados
- Testar mudanças no K (tamanho da bag of words)
  - Se melhora ou piora os resultados 





### Resultados para colher
- Verificar se o score do novo prompt está próximo da média de score dos prompts base -> gráfico
- Verificar se, se nos basearmos em prompts que eram jailbreak, o novo prompt ainda é jailbreak
- Verificar nova taxa de jailbreak para comparar com a taxa base do ADT
- Alteração em score e jailbreak de acordo com pertubações no alpha e no k do top_k_scores
- Simulated Annealing com custo de perplexidade
- SGD usando função de custo a perplexidade
