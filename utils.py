import yaml
def load_config(config_path: str = "../configs/base_config.yaml"):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found")
    except Exception as e:
        print(f"Error loading config file: {e}")