from dataclasses import dataclass


@dataclass
class P0QAEnvConfig:
    """Single-turn P0 question answering (the QA RL pilot): a rows file of data_gen.interiorgs_v10.qa_rl."""

    env_name: str = "P0QA"
    jsonl_path: str = ""
    image_placeholder: str = "<image>"
