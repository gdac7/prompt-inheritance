import numpy as np
import json
from typing import List
from generate_plots import *

def analyze_score_proximity(new_score: float, base_prompts):
    base_scores = np.array([prompt['score'] for prompt in base_prompts])
    mean_base = np.mean(base_scores)
    std_dev_base = np.std(base_scores)
    median_base = np.min(base_scores)
    min_base = np.min(base_scores)
    max_base = np.max(base_scores)
    count_base = len(base_scores)
    if std_dev_base == 0:
        if new_score == mean_base:
            z_score = 0.0
        else:
            z_score = float('inf') if new_score > mean_base else float('-inf')
    else:
        z_score = (new_score - mean_base) / std_dev_base
    
    result = {
        "new_score": new_score,
        "base_stats": {
            "count": count_base,
            "mean": mean_base,
            "std_dev": std_dev_base,
            "median": median_base,
            "min": min_base,
            "max": max_base,
        },
        "comparison": {
            "z_score": z_score,
            "difference_from_mean": new_score - mean_base
        }
    }

    with open("results/analyze_score_proximity.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    plot_score_distribution(result, base_scores, "results/score_proximity.pdf")
    

def get_jailbreak_rate(prompts_jailbreak_results: List):
    results = [r.lower() for r in prompts_jailbreak_results]
    yes_count = results.count("yes")
    total_count = len(results)
    rate = yes_count / total_count
    return rate

def analyze_jailbreak_persistence(base_prompt_results, new_prompt_results):
    result_list = []
    for i, (base_result, new_result) in enumerate(zip(base_prompt_results, new_prompt_results)):
        base_rate = get_jailbreak_rate(base_result)
        new_rate = get_jailbreak_rate(new_result)
        result = {
            "request_id": i,
            "base_rate": base_rate,
            "new_rate": new_rate
        }
        result_list.append(result)

    with open("results/analyze_jailbreak_persistence.json", "w", encoding="utf-8") as f:
        json.dump(result_list, f, indent=4)
    plot_jailbreak_comparison(result_list, "results/analyze_jailbrealk_persistence.pdf")

def analyze_total_jailbreak_rate(old_result: float, generations: Dict, len_behaviors: int, approach: str):
    all_classifications = generations.values()
    new_result = sum(1 if "yes" in classification else 0 for classification in all_classifications) / len_behaviors
    with open(f"results/analyze_total_jailbreak_rate_{approach}", "w", encoding="utf-8") as f:
        json.dump(f"Novo resultado: {new_result}\nResultado base: {old_result}", f, indent=4)
    plot_total_rate_comparison(old_result, new_result, "results/analyze_total_jailbreak_rate.pdf")




    
    

    



