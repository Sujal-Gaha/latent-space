import os
import sys
import requests

import numpy as np

from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# plt.style.use("dark_background")

load_dotenv()

OLLAMA_MODEl = os.getenv("OLLAMA_MODEL")
OLLAMA_URL = os.getenv("OLLAMA_URL") or "http://localhost:11434"


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


def collect_all_embeddings(WORDS):
    """
    Loop through all texts, get their embeddings, and return everything as parallel lists.
    """

    all_texts = []
    all_vectors = []
    all_colors = []
    all_groups = []

    total = sum(len(g["items"]) for g in WORDS.values())

    print(f"Fetching embeddings from Ollama ({OLLAMA_MODEl})")
    print(f"{total} words\n")

    count = 0
    for group_name, group_data in WORDS.items():
        for word in group_data["items"]:
            count += 1
            print(f"[{count:3d}/{total}] Embedding: '{word}'")
            vec = get_embedding(word)
            all_texts.append(word)
            all_vectors.append(vec)
            all_colors.append(group_data["color"])
            all_groups.append(group_name)

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
