import random

from collections import defaultdict
from datasets import load_dataset


def truncate_text(text: str, max_chars: int = 500) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "..."


N_PER_CLASS = 40


def load_ag_news(n_per_class: int = 40) -> dict:
    LABEL_MAP = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Technology",
    }

    LABEL_COLORS = {
        "World": "#3498DB",
        "Sports": "#E74C3C",
        "Business": "#2ECC71",
        "Technology": "#F39C12",
    }

    print("Loading ag_news dataset...")

    ALL_DOCUMENTS = load_dataset("ag_news", split="train")

    grouped = defaultdict(list)
    for item in ALL_DOCUMENTS:
        label_name = LABEL_MAP[item["label"]]
        grouped[label_name].append(item["text"])

    return {
        label: {
            "color": LABEL_COLORS[label],
            "items": [truncate_text(t) for t in random.sample(items, N_PER_CLASS)],
        }
        for label, items in grouped.items()
    }
