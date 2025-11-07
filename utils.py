import yaml
import torch

def load_config(config_path: str = "../configs/base_config.yaml"):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found")
    except Exception as e:
        print(f"Error loading config file: {e}")


def calc_perplexity(ppl_tokenizer, ppl_model, prompt):
    inputs = ppl_tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs.input_ids
    with torch.no_grad():
        outputs = ppl_model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
    
    ppl = torch.exp(loss)
    return ppl.item()