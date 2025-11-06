from typing import Dict, List
import matplotlib as plt
import seaborn as sns

def plot_score_distribution(analysis_data: Dict[str, Any], base_scores: List[float], filename: str):
    """
    Plota a distribuição dos scores base e a posição do novo score.
    
    Args:
        analysis_data: O dicionário JSON retornado por analyze_score_proximity.
        base_scores: A lista bruta de scores base.
        filename: O nome do arquivo para salvar o gráfico.
    """
    
    # 1. Extrair os dados estatísticos do dicionário de análise
    new_score = analysis_data['new_score']
    stats = analysis_data['base_stats']
    mean = stats['mean']
    std_dev = stats['std_dev']
    z_score = analysis_data['comparison']['z_score']
    
    # 2. Configurar o estilo do gráfico
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7)) # Tamanho (largura, altura) em polegadas
    
    # 3. Plotar o Histograma e a Curva de Densidade (KDE)
    # kde=True desenha a linha de curva de densidade
    # stat="density" normaliza o histograma para que a área sob a curva seja 1
    # element="step" e alpha=0.3 dão o visual mais limpo
    sns.histplot(base_scores, kde=True, stat="density", 
                 label='Distribuição Scores Base', 
                 color='skyblue', element='step', alpha=0.3)
    
    # 4. Linha da Média da Base (Linha vertical tracejada)
    plt.axvline(mean, color='black', linestyle='--', linewidth=2, 
                label=f'Média Base ({mean:.2f})')
    
    # 5. Área de Desvio Padrão (+1 e -1)
    # axvspan cria um retângulo vertical (uma "faixa" sombreada)
    plt.axvspan(max(0, mean - std_dev), mean + std_dev, color='gray', alpha=0.15,
                label=f'Área de 1 Desvio Padrão (±{std_dev:.2f})')
    
    # 6. Linha do Novo Score (A linha vermelha, sólida e mais grossa)
    plt.axvline(new_score, color='red', linestyle='-', linewidth=2.5, 
                label=f'Novo Score ({new_score:.2f})\nEscore-Z: {z_score:.2f}')
    
    # 7. Adicionar Títulos e Labels
    # Usamos o "proximity_summary" do dicionário de análise para criar um título dinâmico
    title = (f'Análise de Proximidade: Novo Score ({new_score:.2f}) vs. Base\n'
             f'{analysis_data["comparison"]["proximity_summary"]}')
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Pontuação (Score)', fontsize=12)
    plt.ylabel('Densidade', fontsize=12)
    
    # 8. Legenda
    # Colocar a legenda fora do gráfico para não sobrepor
    # bbox_to_anchor=(1.05, 1) -> Posição (x, y) fora do eixo
    # loc='upper left' -> Qual "canto" da legenda vai naquela posição
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # 9. Ajustar layout para caber a legenda
    # Isso encolhe a área do gráfico para que a legenda (que está fora) caiba na imagem salva
    # rect=[left, bottom, right, top]
    plt.tight_layout(rect=[0, 0, 0.82, 1]) 
    
    # 10. Salvar o arquivo
    plt.savefig(filename)
    print(f"Gráfico salvo como '{filename}'")
    plt.close() # Fechar a figura para liberar memória

def plot_jailbreak_comparison(
    results_data: List[Dict], 
    filename: str, 
    requests_para_plotar: int = 20 # Parâmetro renomeado
):
    """
    Plota um gráfico de barras agrupadas comparando 'base_rate' e 'new_rate'
    para cada 'request_id'.
    
    Args:
        results_data: Lista de dicionários. Cada dict deve ter as chaves
                      'request_id', 'base_rate', e 'new_rate'.
        filename: Nome do arquivo para salvar (ex: 'grafico.png' ou 'grafico.pdf').
        requests_para_plotar: Número de requests a exibir no eixo X.
    """
    
    print("Iniciando a plotagem...")
    
    if not results_data:
        print("Não há dados para plotar.")
        return
        
    df_wide = pd.DataFrame(results_data)
    
    # [MODIFICAÇÃO 1: Validação das novas chaves]
    if not {'request_id', 'base_rate', 'new_rate'}.issubset(df_wide.columns):
        print("Erro: Os dados devem conter as colunas 'request_id', 'base_rate' e 'new_rate'.")
        return

    # [MODIFICAÇÃO 2: Melt com as novas chaves]
    df_long = df_wide.melt(
        id_vars=['request_id'],       # ATUALIZADO
        value_vars=['base_rate', 'new_rate'], # ATUALIZADO
        var_name='tipo_prompt',
        value_name='taxa_jailbreak'
    )
    
    # [MODIFICAÇÃO 3: Mapeamento das novas chaves]
    df_long['tipo_prompt'] = df_long['tipo_prompt'].map({
        'base_rate': 'Prompts Base', # ATUALIZADO
        'new_rate': 'Prompts Novos'  # ATUALIZADO
    })
    
    # [MODIFICAÇÃO 4: Filtro com a nova chave]
    df_wide_sorted = df_wide.sort_values(by='request_id') # ATUALIZADO
    ids_para_plotar = df_wide_sorted['request_id'].unique()[:requests_para_plotar] # ATUALIZADO
    
    if len(ids_para_plotar) == 0:
        print("Não há IDs de request válidos nos dados.")
        return
        
    df_plot = df_long[df_long['request_id'].isin(ids_para_plotar)] # ATUALIZADO
    
    num_requests_plotados = len(ids_para_plotar)
    print(f"Plotando {num_requests_plotados} requests (de {ids_para_plotar[0]} a {ids_para_plotar[-1]})...")

    # [MODIFICAÇÃO 5: Plotagem]
    largura_grafico = max(15, num_requests_plotados * 0.8) 
    plt.figure(figsize=(largura_grafico, 8))
    sns.set_style("whitegrid")

    ax = sns.barplot(
        data=df_plot,
        x='request_id', # ATUALIZADO
        y='taxa_jailbreak',
        hue='tipo_prompt',
        palette=['cornflowerblue', 'darkorange']
    )

    # [MODIFICAÇÃO 6: Títulos e Labels]
    ax.set_title(f'Comparação da Taxa de Jailbreak: Base vs. Novos (Requests {ids_para_plotar[0]}-{ids_para_plotar[-1]})', fontsize=20, pad=20)
    ax.set_xlabel('ID do Request', fontsize=14) # ATUALIZADO
    ax.set_ylabel('Taxa de Jailbreak (0.0 = 0%, 1.0 = 100%)', fontsize=14)
    ax.set_ylim(0, 1.1)

    # (Loop de anotação não precisa de mudanças)
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.0%}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points',
                        fontsize=9,
                        color='black')

    ax.legend(title='Tipo de Prompt', fontsize=12, title_fontsize=14, loc='upper right')
    
    plt.savefig(filename, bbox_inches='tight')
    print(f"Gráfico de barras agrupadas salvo como '{filename}'")
    plt.close()


def plot_total_rate_comparison(old_rate: float, new_rate: float, filename: str):
    """
    Cria um gráfico de barras simples para comparar duas taxas (antiga vs. nova).
    
    Args:
        old_rate: O valor da taxa antiga (float).
        new_rate: O valor da taxa nova (float).
        filename: Nome do arquivo para salvar (ex: 'comparacao.png').
    """
    
    print(f"Plotando comparação de taxas e salvando em '{filename}'...")
    
    # 1. Estruturar os dados para o Seaborn
    data = {
        'Tipo de Taxa': ['Taxa Antiga (Base)', 'Taxa Nova (Gerada)'],
        'Taxa': [old_rate, new_rate]
    }
    df_plot = pd.DataFrame(data)

    # 2. Configurar o gráfico
    plt.figure(figsize=(8, 6)) # Tamanho (largura, altura)
    sns.set_style("whitegrid")
    
    # 3. Criar o gráfico de barras
    ax = sns.barplot(
        data=df_plot,
        x='Tipo de Taxa', # Categorias no eixo X
        y='Taxa',         # Valores no eixo Y
        palette=['#FF9999', '#99FF99'] # Cores (Vermelho claro, Verde claro)
    )
    
    # 4. Melhorar o gráfico (Títulos e Limites)
    ax.set_title('Comparação da Taxa Total de Jailbreak', fontsize=16, pad=20)
    ax.set_xlabel('Origem do Resultado', fontsize=12)
    ax.set_ylabel('Taxa de Jailbreak (0.0 = 0%, 1.0 = 100%)', fontsize=12)
    
    # Definir limite do eixo Y para ir de 0% a 100% (ou 110% para dar espaço)
    ax.set_ylim(0, 1.1) 

    # 5. Adicionar anotações de porcentagem em cima das barras
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1%}', # Formato de porcentagem com 1 casa decimal
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=12, weight='bold')
    
    # 6. Salvar o arquivo
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo como '{filename}'")
