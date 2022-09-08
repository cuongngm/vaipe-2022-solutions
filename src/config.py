from types import SimpleNamespace
import yaml


def load_config(path: str):
    with open(path, 'r') as fr:
        cfg = yaml.safe_load(fr)
        for k, v in cfg.items():
            if type(v) == dict:
                cfg[k] = SimpleNamespace(**v)
        cfg = SimpleNamespace(**cfg)
    return cfg

