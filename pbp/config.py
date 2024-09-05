from dataclasses import dataclass

import yaml
from transformers import LlamaConfig


@dataclass
class PBPConfig(LlamaConfig):
    """
    LlamaConfig args:
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,

    PBPConfig args:
        seed=42,
        device="cuda",
        output_dir="./pbp_output",
        wandb=False,
        wandb_project="",
        train_dataset="./data/train.csv",
        eval_dataset="./data/val.csv",
        compile=False,
        precision="fp32",
        batch_size=32,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        use_8bit_adam=False,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        epochs=5,
        warmup=500,
        save_interval=1000,
        eval_interval=100,
        eval_steps=100,
    """
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
    use_8bit_adam: bool = False

    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    epochs: int = 5
    warmup: int = 500
    save_interval: int = 1000
    eval_interval: int = 100
    eval_steps: int = 100

    team_embedding_dim: int = 256
    length_sample: bool = False

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