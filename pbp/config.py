from dataclasses import dataclass

import yaml
from transformers import LlamaConfig


@dataclass
class PBPConfig(LlamaConfig):
    pretrained: str = None
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "./pbp_output"
    wandb: bool = False
    wandb_project: str = ""

    train_dataset: str = "./data/train.csv"
    eval_dataset: str = "./data/val.csv"

    compile: bool = False
    precision: str = "fp32"
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    use_adam: bool = False

    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0

    epochs: int = 5
    warmup: int = 500
    save_interval: int = 1000
    eval_interval: int = 100
    eval_steps: int = 100

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid argument: {key}")

def parse_yaml_to_config(yaml_path: str) -> PBPConfig:
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return PBPConfig(**config_dict)