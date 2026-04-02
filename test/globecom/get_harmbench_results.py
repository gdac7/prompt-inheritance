import os
import sys


from get_harmbench_acc import get_results 

def get_harmbench_results(paths):
    generations_paths = []
    results_paths = []
    for path in paths:
        generations_file = [f for f in os.listdir(path) if f.endswith('.json') and f != "result.json"][0]
        generations_paths.append(os.path.join(path, generations_file))
        results_paths.append(os.path.join(path, "result.json"))

    get_results(
            generations_paths,
            results_paths,
    )

                


if __name__ == "__main__":
    paths = [
        "generations/autodan/",
        "generations/gw/",
        "generations/ica/",
        "generations/pca/",
        "generations/swpca/",
    ]
    get_harmbench_results(paths)
