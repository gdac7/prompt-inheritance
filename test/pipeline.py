import multiprocessing as mp
from test_scores import (
    get_approaches_results,
    get_new_scores,
    simulated_annealing,
    get_simulated_annealing_scores
)

def run_get_approaches_results(queue):
    new_prompts = get_approaches_results()
    queue.put(new_prompts)

def run_get_new_scores(queue_in, queue_out):
    new_prompts = queue_in.get()
    scored_prompt_list = get_new_scores(new_prompts)
    queue_out.put(scored_prompt_list)

def run_simulated_annealing(queue_in, queue_out):
    scored_prompt_list = queue_in.get()
    simulated_annealing_results = simulated_annealing(scored_prompt_list)
    queue_out.put(simulated_annealing_results)

def run_get_sa_scores(queue_in):
    simulated_annealing_results = queue_in.get()
    simulated_annealing_results_with_score = get_simulated_annealing_scores(simulated_annealing_results)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    q1 = mp.Queue()
    q2 = mp.Queue()
    q3 = mp.Queue()

    
    p1 = mp.Process(target=run_get_approaches_results, args=(q1,))
    p1.start()
    p1.join()

    
    p2 = mp.Process(target=run_get_new_scores, args=(q1, q2))
    p2.start()
    p2.join()

    
    p3 = mp.Process(target=run_simulated_annealing, args=(q2, q3))
    p3.start()
    p3.join()

    
    p4 = mp.Process(target=run_get_sa_scores, args=(q3,))
    p4.start()
    p4.join()