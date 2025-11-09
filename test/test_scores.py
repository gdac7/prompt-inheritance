import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../autodan-itau")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from src.api_models.remote_model import RemoteModelAPI
from src.core.attack_generator import AttackGenerator as ag
from llm_code.prompt_manager import SanitizerPrompt
from utils import calc_perplexity
import json
import gc
import random
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, set_seed
import torch
from llm_code.llm import LocalModelTransformers
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from utils import load_config
import math
import time
import multiprocessing as mp


json_path = "../data/data.json"
config = load_config("../config/models.yaml")

model = SentenceTransformer(config["models"]["embedding"])
sanitizer_model_name = config["models"]["sanitizer"]

def load_data():
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_base_prompts_and_scores(sim_prompts, data):
    base_scores = [float(item["score"]) for item in data if item["attack_prompt"] in sim_prompts]
    return sim_prompts, base_scores


def find_n_similar(request, requests_embeddings, n, all_requests, data, top_m=20):
    query_embedding = model.encode(request).reshape(1, -1)
    cos_sim = cosine_similarity(query_embedding, requests_embeddings)
    top_n = np.argsort(cos_sim[0])[-n:][::-1]
    sim_requests = []
    for idx in top_n:
        sim_requests.append(all_requests[idx])
    sim_requests_set = set(sim_requests)
    sim_prompts = []
    original_data_idx = []
    for original_idx, item in enumerate(data):
        if item["malicious_request"] in sim_requests_set and len(sim_prompts) < top_m:
            sim_prompts.append(item["attack_prompt"])
            original_data_idx.append(original_idx)
    cluster_embeddings = np.array(model.encode(sim_prompts, show_progress_bar=False))
    return cluster_embeddings, sim_requests, sim_prompts, original_data_idx

def get_centroid(cluster_embeddings):
    centroid = np.mean(cluster_embeddings, axis=0)
    centered_embeddings = cluster_embeddings - centroid
    return centered_embeddings, centroid

def apply_pca(centered_embeddings, centroid, k=15):
    pca = PCA(n_components=1)
    pca.fit(centered_embeddings)
    v1 = pca.components_[0]
    ## Getting new vector
    alpha_scale = np.sqrt(pca.explained_variance_[0])
    alpha = 1.0 * alpha_scale
    c_new = centroid + (alpha * v1)
    transformer_model = model[0]
    tokenizer = transformer_model.tokenizer
    word_embedding_matrix = transformer_model.auto_model.get_input_embeddings().weight.detach().cpu().numpy()
    c_new = c_new.reshape(1, -1)
    all_cos_sim = cosine_similarity(c_new, word_embedding_matrix)
    top_k_index = np.argsort(all_cos_sim[0])[-k:][::-1]
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_index)
    top_k_scores = all_cos_sim[0][top_k_index]
    bow = []
    special_tokens = tokenizer.all_special_tokens
    for token, score in zip(top_k_tokens, top_k_scores):
        if token not in special_tokens and not token.startswith("##") and len(token) > 1:
            bow.append(token)
    return bow

def apply_ica(centered_embeddings, centroid, n_components=5, max_iter=1000, tol=1e-3, random_state=42, component_index_to_use=0, k=15):
    ica = FastICA(
        n_components=n_components,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    ica.fit(centered_embeddings)
    ica_components = ica.components_
    v1 = ica_components[component_index_to_use]
    transformed_data = ica.transform(centered_embeddings)
    component_scores = transformed_data[:, component_index_to_use]
    alpha_scale = np.std(component_scores)
    alpha = 1.0 * alpha_scale
    c_new = centroid + (alpha * v1)
    ### Getting tokens back
    transformer_model = model[0]
    tokenizer = transformer_model.tokenizer
    word_embedding_matrix = transformer_model.auto_model.get_input_embeddings().weight.detach().cpu().numpy()
    c_new = c_new.reshape(1, -1)
    all_cos_sim = cosine_similarity(c_new, word_embedding_matrix)
    top_k_index = np.argsort(all_cos_sim[0])[-k:][::-1]
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_index)
    top_k_scores = all_cos_sim[0][top_k_index]
    bow = []
    special_tokens = tokenizer.all_special_tokens
    for token, score in zip(top_k_tokens, top_k_scores):
        if token not in special_tokens and not token.startswith("##") and len(token) > 1:
            bow.append(token)
    return bow

def apply_lca_pca(data, top_n, cluster_embeddings, sucess_threshold=8.5, n_components=2, random_state=42):
    features_list = []
    embeddings_list = []
    for _, original_index in enumerate(top_n):
        score = data[original_index]['score']
        is_successful = 1 if score >= sucess_threshold else 0
        features_list.append([is_successful])
    
    lca_features = np.array(features_list)
    aligned_embeddings = cluster_embeddings

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(aligned_embeddings)
    prompt_classes = gmm.predict(aligned_embeddings)
    success_labels = lca_features.flatten()
    success_rate_class_0 = np.mean(success_labels[prompt_classes == 0]) if (prompt_classes == 0).any() else 0.0
    success_rate_class_1 = np.mean(success_labels[prompt_classes == 1]) if (prompt_classes == 1).any() else 0.0
    if success_rate_class_1 >= success_rate_class_0:
        elite_class_label = 1
    else:
        elite_class_label = 0
    elite_mask = (prompt_classes == elite_class_label)
    elite_cluster_embeddings = aligned_embeddings[elite_mask]
    if elite_cluster_embeddings.shape[0] == 0:
        ### elite cluster is empty fall back to using the original (non-elite) cluster
        elite_cluster_embeddings = aligned_embeddings

    elite_centered_embeddings, centroid = get_centroid(elite_cluster_embeddings)
    baw_lca_pca = apply_pca(elite_centered_embeddings, centroid)
    return baw_lca_pca, elite_mask

def get_new_prompts(sanitizer, malicious_request, pca_result, ica_result, lca_pca_result, base_prompts, base_scores, base_prompts_lca, base_scores_lca, num_prompts=5) -> Dict[str, Dict[str, any]]:
    # Chamar o LLM para gerar 5 samples com cada bag of words dos métodos
    # Nos argumentos
    # Teremos no final um dicionario com {pca: prompts: List[5 items], scores: List[float], mean_score: float, o mesmo para o resto}
    bow_map = {
        "pca": pca_result,
        "ica": ica_result,
        "lca_pca": lca_pca_result
    }

    base_templates = {}
    for key, bow in bow_map.items():
        base_templates[key] = SanitizerPrompt.get_sanitizer_prompt(malicious_request, bow)
    
    user_prompts_batch = []
    keys_order = ["pca", "ica", "lca_pca"]
    for key in keys_order:
        template = base_templates[key]
        user_prompts_batch.extend([template.user_prompt] * num_prompts)
    common_template = base_templates["pca"]
    all_generated_prompts = sanitizer.batch_generate(
        user_prompts=user_prompts_batch,
        system_prompt=common_template.system_prompt,
        condition=common_template.condition,
        temperature=common_template.temperature,
        max_tokens=common_template.max_tokens,
    )

    base_mean_score = np.mean(base_scores)
    base_mean_score_lca = np.mean(base_scores_lca)
    results_dict = {
        "pca": {
            "malicious_request": malicious_request,
            "prompts": all_generated_prompts[0:num_prompts],
            "target_responses": [],
            "scores": [],
            "best_prompt": "",
            "worst_prompt": "",
            "mean_score": 0.0,
            "base_prompts": base_prompts,
            "base_scores": base_scores,
            "base_mean_score": base_mean_score,
        },
        "ica": {
            "malicious_request": malicious_request,
            "prompts": all_generated_prompts[num_prompts:num_prompts*2],
            "target_responses": [],
            "scores": [],
            "best_prompt": "",
            "worst_prompt": "",
            "mean_score": 0.0,
            "base_prompts": base_prompts,
            "base_scores": base_scores,
            "base_mean_score": base_mean_score,
        },
        "lca_pca": {
            "malicious_request": malicious_request,
            "prompts": all_generated_prompts[num_prompts*2:],
            "target_responses": [],
            "scores": [],
            "best_prompt": "",
            "worst_prompt": "",
            "mean_score": 0.0,
            "base_prompts": base_prompts_lca,
            "base_scores": base_scores_lca,
            "base_mean_score": base_mean_score_lca,
        }
    }
    return results_dict



# def create_dict_result(new_prompts_and_scores, base_prompts, base_scores):
#     # Ideia: Ter no final: {pca_prompts: base_prompts: List[str], base_scores: List[float], base_mean_score: float, new_prompts: List[5 items], ica_prompts: idem, lca_pca_prompts: idem, sa_ica, sa_pca, sa_lca_pca}
#     # Com isso depois salvamos esses dados é só plotar os dados
#     final_result = new_prompts_and_scores.copy()
#     for key, value in new_prompts_and_scores.items():
#         final_result[key]["base_prompts"] = base_prompts
#         final_result[key]["base_scores"] = base_scores
#         final_result[key]["base_mean_score"] = np.mean(final_result[key]["base_scores"])

#     return final_result

def get_approaches_results(output_dir="results/get_approaches_results.json"):
    data = load_data()
    sanitizer = LocalModelTransformers(sanitizer_model_name)
    requests = list(set([item["malicious_request"] for item in data]))[:30]
    requests_embeddings = np.array(model.encode(requests, show_progress_bar=False))
    n = 10
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    all_new_prompts = []
    for request in tqdm(requests, desc="Getting approaches results..."):
        cluster_embeddings, sim_requests, sim_prompts, top_n = find_n_similar(
            request, 
            requests_embeddings, 
            n, 
            requests, 
            data
        )
        base_prompts, base_scores = get_base_prompts_and_scores(sim_prompts, data)
        centered_embeddings, centroid = get_centroid(cluster_embeddings)
        baw_pca = apply_pca(centered_embeddings, centroid)
        baw_ica = apply_ica(centered_embeddings, centroid, n_components=5)
        baw_lca_pca, elite_mask = apply_lca_pca(data, top_n, cluster_embeddings)
        base_prompts_np = np.array(base_prompts)
        base_scores_np = np.array(base_scores)
        base_prompts_lca = base_prompts_np[elite_mask].tolist()
        base_scores_lca = base_scores_np[elite_mask].tolist()
        new_prompts = get_new_prompts(
            sanitizer=sanitizer,
            malicious_request=request,
            pca_result=baw_pca, 
            ica_result=baw_ica, 
            lca_pca_result=baw_lca_pca,
            base_prompts=base_prompts,
            base_scores=base_scores,
            base_prompts_lca=base_prompts_lca,
            base_scores_lca=base_scores_lca,
        )
        all_new_prompts.append(new_prompts)

    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(all_new_prompts, f, ensure_ascii=False, indent=4)

    del sanitizer
    gc.collect()
    torch.cuda.empty_cache()

    return all_new_prompts

def get_new_scores(new_prompts, output_dir="results/get_new_scores_results.json"):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    scorer = RemoteModelAPI("http://localhost:8001/generate_score")
    attack_generator = ag(None, None, scorer, None)
    target = LocalModelTransformers("meta-llama/Llama-3.1-8B-Instruct")
    for request_results in tqdm(new_prompts, desc="Scoring new prompts"):
        for method, results in request_results.items():
            best_score = -1
            worst_score = 11
            prompts = results["prompts"]   
            target_responses = target.batch_generate(user_prompts=prompts)  
            scores = []
            for prompt, response in tqdm(zip(prompts, target_responses), desc="Getting score"):
                score, _ = attack_generator._score_response(results["malicious_request"], prompt, response)
                if score is None:
                    continue
                if score < worst_score:
                    worst_score = score
                    worst_prompt = prompt
                if score > best_score:
                    best_score = score
                    best_prompt = prompt
                scores.append(score)
            results["target_responses"] = target_responses
            results["scores"] = scores
            results["mean_score"] = sum(scores) / len(scores)
            results["best_prompt"] = best_prompt
            results["worst_prompt"] = worst_prompt

    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(new_prompts, f, ensure_ascii=False, indent=4)
    del target
    gc.collect()
    torch.cuda.empty_cache()

    return new_prompts


#### Simulated Annealing
def simulated_annealing(prompts_list, iterations=1000, initial_temp=1.0, cooling_rate=0.995, maximize_ppl=True, output_dir="results/simmulated_annealing_results.json"):
    model = config["models"]["perplexed"]
    mask_model = config["models"]["mask-model"]
    ppl_tokenizer = AutoTokenizer.from_pretrained(model)
    ppl_model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype = torch.bfloat16,
        device_map="auto"
    )
    mutator = pipeline(
        "fill-mask",
        model=mask_model,
        device="cuda",
    )
    ppl_model.eval()
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    for request in tqdm(prompts_list, desc="Simulated Annealing"):
        for method, results in request.items():
            T = initial_temp
            curr_prompt = results["best_prompt"]
            curr_cost = calc_perplexity(ppl_tokenizer, ppl_model, curr_prompt)
            best_prompt = curr_prompt
            best_cost = curr_cost
            print(f"Iteration 0/{iterations}: Initial Cost = {curr_cost:.4f}")
            for i in range(1, iterations + 1):
                neighbor_prompt = get_neighbor(mutator, curr_prompt)
                neighbor_cost = calc_perplexity(ppl_tokenizer, ppl_model, neighbor_prompt)
                if maximize_ppl:
                    delta_cost = -1 * (neighbor_cost - curr_cost)
                else:
                    delta_cost = neighbor_cost - curr_cost
                if delta_cost < 0:
                    accept = True
                else:
                    accept_likelihood = math.exp(-delta_cost / T)
                    accept = random.random() < accept_likelihood
                if accept:
                    curr_prompt = neighbor_prompt
                    curr_cost = neighbor_cost
                if maximize_ppl:
                    if curr_cost > best_cost:
                        best_cost = curr_cost
                        best_prompt = curr_prompt
                else:
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_prompt = curr_prompt
                T = T * cooling_rate
                if i % 100 == 0:
                    print(f"iteration {i}/{iterations}: Temp={T:.4f}, Curr Cost={curr_cost:.4f}, Best={best_cost:.4f}")
            results["simulated_annealing_prompt"] = best_prompt
            results["simulated_annealing_ppl_cost"] = best_cost

    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(prompts_list, f, ensure_ascii=False, indent=4)
    del ppl_tokenizer
    del ppl_model
    del mutator
    gc.collect()
    torch.cuda.empty_cache()
    return prompts_list

def get_neighbor(mutator, curr_prompt, top_k = 5):
    tokens = curr_prompt.split()
    if len(tokens) < 3:
        return curr_prompt
    idx_to_mask = random.randint(1, len(tokens) - 2)
    original_token = tokens[idx_to_mask]
    tokens[idx_to_mask] = mutator.tokenizer.mask_token
    masked_prompt = " ".join(tokens)
    if len(masked_prompt) > 512:
        if "[MASK]" in masked_prompt:
            mask_index = masked_prompt.index("[MASK]")
            start = max(0, mask_index - 250)
            end = min(len(masked_prompt), mask_index + 250)
            masked_prompt = masked_prompt[start:end]
        else:
            masked_prompt = masked_prompt[:512]
    disturbances = mutator(masked_prompt, top_k=top_k)
    valid_dists = [
        s['token_str'] for s in disturbances
        if s['token_str'] != original_token and s['token_str'].strip()
    ]
    if not valid_dists:
        return curr_prompt
    new_token = random.choice(valid_dists)
    tokens[idx_to_mask] = new_token
    new_prompt = " ".join(tokens)
    return new_prompt


def get_simulated_annealing_scores(prompts_list, output_dir="results/simmulated_annealing_results_with_score.json"):
    scorer = RemoteModelAPI("http://localhost:8001/generate_score")
    target = LocalModelTransformers("meta-llama/Llama-3.1-8B-Instruct")
    attack_generator = ag(None, None, scorer, None)
    for request in tqdm(prompts_list, desc="Scoring simulated annealing"):
        for method, results in request.items():
            prompt = results["simulated_annealing_prompt"]
            target_response = target.generate(user_prompt=prompt)
            score, _ = attack_generator._score_response(results["malicious_request"], prompt, target_response)
            results["simulated_annealing_prompt_score"] = score
            results["simulated_annealing_target_response"] = target_response
    del target
    gc.collect()
    torch.cuda.empty_cache()
    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(prompts_list, f, ensure_ascii=False, indent=4)
    return prompts_list


            



    
        


    

if __name__ == "__main__":    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    new_prompts = get_approaches_results()
    scored_prompt_list = get_new_scores(new_prompts)
    simulated_annealing_results = simulated_annealing(scored_prompt_list)
    simulated_annealing_results_with_score = get_simulated_annealing_scores(simulated_annealing_results)
