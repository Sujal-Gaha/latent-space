import os
import sys
import requests
import random

import numpy as np

from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# plt.style.use("dark_background")

load_dotenv()

OLLAMA_MODEl = os.getenv("OLLAMA_MODEL")
OLLAMA_URL = os.getenv("OLLAMA_URL") or "http://localhost:11434"


def get_random_colors(n: int = 2) -> list[str]:
    """
    Args:
        n is the total number of random colors to generate

    Returns:
        list of random colors in hex format
    """

    BLUE = "#1F77B4"
    ORANGE = "#FF7F0E"
    GREEN = "#2CA02C"
    RED = "#D62728"
    PURPLE = "#9467BD"
    BROWN = "#8C564B"
    PINK = "#E377C2"
    GRAY = "#7F7F7F"
    YELLOW = "#BCBD22"
    CYAN = "#17BECF"

    COLORS = [BLUE, ORANGE, GREEN, RED, PURPLE, BROWN, PINK, GRAY, YELLOW, CYAN]

    random_colors = [random.choice(COLORS) for _ in range(n)]

    return random_colors


# 1. Get embedding from Ollama
def get_embedding(text: str) -> np.ndarray:
    """
    Send a text to Ollama and get back its embedding vector.
    """
    payload = {"model": OLLAMA_MODEl, "prompt": text}
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return np.array(data["embedding"])
    except requests.exceptions.ConnectionError:
        print(f"\nCannot connect to Ollama at {OLLAMA_URL}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError getting embedding for '{text}': {e}")
        sys.exit(1)


def collect_all_embeddings(texts):
    """
    Loop through all texts, get their embeddings, and return everything as parallel lists.
    """

    all_texts = []
    all_vectors = []
    all_colors = []
    all_groups = []

    total = sum(len(g["items"]) for g in texts.values())

    print(f"Fetching embeddings from Ollama ({OLLAMA_MODEl})")
    print(f"{total} words\n")

    count = 0
    for group_name, group_data in texts.items():
        for text in group_data["items"]:
            count += 1
            print(f"[{count:3d}/{total}] Embedding: '{text}'")
            vec = get_embedding(text)
            all_texts.append(text)
            all_vectors.append(vec)
            all_colors.append(group_data["color"])
            all_groups.append(group_name)

    return all_texts, np.array(all_vectors), all_colors, all_groups


def collect_all_embeddings_V2(texts, default_group="texts"):
    """
    For both custom dataset and hugginface datasets
    """

    all_texts = []
    all_vectors = []
    all_colors = []
    all_groups = []

    # CASE 1: grouped dictionary dataset
    if isinstance(texts, dict):
        total = sum(len(g["items"]) for g in texts.values())

        print(f"Fetching embeddings from Ollama ({OLLAMA_MODEl})")
        print(f"{total} texts\n")

        count = 0
        for group_name, group_data in texts.items():
            for text in group_data["items"]:
                count += 1
                print(f"[{count:3d}/{total}] Emedding: '{text}'")

                vec = get_embedding(text)

                all_texts.append(text)
                all_vectors.append(vec)
                all_colors.append(group_data["color"])
                all_groups.append(group_name)

    # CASE 2: simple list dataset
    elif isinstance(texts, list):
        total = len(texts)

        print(f"Fetching embeddings from Ollama {OLLAMA_MODEl}")
        print(f"{total} texts\n")

        random_colors = get_random_colors(n=5)

        for i, text in enumerate(texts, 1):
            print(f"[{i:3d}/{total}] Embedding: '{text[:150]}'")

            vec = get_embedding(text)

            all_texts.append(text)
            all_vectors.append(vec)
            all_colors.append(random.choice(random_colors))
            all_groups.append(default_group)

    else:
        raise ValueError("Unsupported dataset format")

    return all_texts, np.array(all_vectors), all_colors, all_groups


# 2. Reduce with t-SNE
def reduce_with_tsne(vectors: np.ndarray, n_components: int = 2) -> np.ndarray:
    print("Running t-SNE")

    scalar = StandardScaler()
    vectors_scaled = scalar.fit_transform(vectors)

    tsne = TSNE(
        n_components=n_components,
        perplexity=min(10, len(vectors) // 3),
        n_iter_without_progress=2000,
        learning_rate="auto",
        init="pca",
        random_state=42,
        metric="cosine",
    )

    coords_2d = tsne.fit_transform(vectors_scaled)

    print("t-SNE complete!\n")

    return coords_2d
