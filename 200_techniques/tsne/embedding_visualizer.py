import os
import sys
import requests

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# plt.style.use("dark_background")

load_dotenv()

OLLAMA_MODEl = os.getenv("OLLAMA_MODEL")
OLLAMA_URL = os.getenv("OLLAMA_URL") or "http://localhost:11434"

WORD_GROUPS = {
    "Music": {
        "color": "#9B59B6",
        "items": [
            "guitar",
            "piano",
            "violin",
            "drums",
            "flute",
            "symphony",
            "melody",
            "rhythm",
            "orchestra",
            "concert",
        ],
    },
    "Animal": {
        "color": "#1ABC9C",
        "items": [
            "cat",
            "dog",
            "elephant",
            "lion",
            "tiger",
            "horse",
            "rabbit",
            "giraffe",
            "zebra",
            "monkey",
        ],
    },
    "Technology": {
        "color": "#F6515B",
        "items": [
            "laptop",
            "smartphone",
            "database",
            "cloud computing",
            "cybersecurity",
            "machine learning",
            "artificial intelligence",
            "programming",
            "operating system",
            "computer network",
        ],
    },
    "Food": {
        "color": "#480355",
        "items": [
            "pizza",
            "burger",
            "pasta",
            "french fries",
            "noodles",
            "sandwich",
            "fried rice",
            "sushi",
            "tacos",
            "salad",
        ],
    },
    "Mathematics": {
        "color": "#482728",
        "items": [
            "axiom",
            "theorem",
            "asymptote",
            "probability",
            "statistics",
            "calculus",
            "matrix",
            "derivative",
            "integral",
            "geometry",
        ],
    },
    "Biology": {
        "color": "#140D4F",
        "items": [
            "zoology",
            "botany",
            "cell",
            "mitochondria",
            "photosynthesis",
            "dna",
            "enzyme",
            "genetics",
            "evolution",
            "cell membrane",
        ],
    },
    "Chemistry": {
        "color": "#EE6352",
        "items": [
            "atom",
            "valence electron",
            "dilution",
            "citric acid",
            "sodium chloride",
            "sulfuric acid",
            "trichloromethane",
            "van der waals force",
            "molecule",
            "chemical reaction",
        ],
    },
    "Physics": {
        "color": "#00FF00",
        "items": [
            "velocity",
            "acceleration",
            "force",
            "mass",
            "density",
            "volume",
            "energy",
            "momentum",
            "gravity",
            "electromagnetism",
        ],
    },
    "Geography": {
        "color": "#3498DB",
        "items": [
            "continent",
            "ocean",
            "river",
            "mountain",
            "valley",
            "plateau",
            "desert",
            "island",
            "oasis",
            "latitude",
        ],
    },
    "Literature": {
        "color": "#E67E22",
        "items": [
            "novel",
            "poetry",
            "drama",
            "author",
            "metaphor",
            "narrative",
            "protagonist",
            "antagonist",
            "theme",
            "plot",
        ],
    },
}


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


def collect_all_embeddings():
    """
    Loop through all texts, get their embeddings, and return everything as parallel lists.
    """

    all_texts = []
    all_vectors = []
    all_colors = []
    all_groups = []

    total = sum(len(g["items"]) for g in WORD_GROUPS.values())

    print(f"Fetching embeddings from Ollama ({OLLAMA_MODEl})")
    print(f"{total} words\n")

    count = 0
    for group_name, group_data in WORD_GROUPS.items():
        for word in group_data["items"]:
            count += 1
            print(f"[{count:3d}/{total}] Embedding: '{word}'")
            vec = get_embedding(word)
            all_texts.append(word)
            all_vectors.append(vec)
            all_colors.append(group_data["color"])
            all_groups.append(group_name)

    return all_texts, np.array(all_vectors), all_colors, all_groups


# 2. Reduce to 2D with t-SNE
def reduce_with_tsne(vectors: np.ndarray) -> np.ndarray:
    print("Running t-SNE")

    scalar = StandardScaler()
    vectors_scaled = scalar.fit_transform(vectors)

    tsne = TSNE(
        n_components=2,
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


# 3. Plot
def plot_embedding_space(texts, coords, colors, groups):
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        f"Semantic Embedding Space - t-SNE Visualization\n"
        f"Model: {OLLAMA_MODEl} | Metric: cosine | {len(texts)}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        color="#2C3E50",
    )

    for ax_idx, ax in enumerate(axes):
        ax.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#DDDDDD")
        ax.spines["bottom"].set_color("#DDDDDD")
        ax.tick_params(colors="#AAAAAA")
        ax.set_xlabel("t-SNE Dimension 1", color="#888888", fontsize=10)
        ax.set_ylabel("t-SNE Dimension 2", color="#888888", fontsize=10)

        ax.axhline(y=0, color="#EEEEEE", linewidth=1, zorder=0)
        ax.axvline(x=0, color="#EEEEEE", linewidth=1, zorder=0)

        count = 0
        for i, (text, color) in enumerate(zip(texts, colors)):
            x, y = coords[i]
            count = i + 1

            print(
                f"[{count:3d}/{len(texts)}] Coordinate: ({x:.2f}, {y:.2f}) for text: '{text}'"
            )

            ax.scatter(
                x,
                y,
                c=color,
                s=180,
                zorder=3,
                edgecolors="white",
                linewidths=1.5,
                alpha=0.9,
            )

        label_indices = []

        if ax_idx == 0:
            ax.set_title(
                "Semantic Clusters\n(key labels shown)",
                fontsize=11,
                color="#555555",
                pad=10,
            )
            label_indices = list(range(len(texts)))

        for i in label_indices:
            text_str = texts[i]
            color = colors[i]
            x, y = coords[i]

            display = text_str if len(text_str) <= 20 else text_str[:18] + "..."

            ax.annotate(
                display,
                xy=(x, y),
                xytext=(8, 6),
                textcoords="offset points",
                fontsize=8.5,
                color="#2C3E50",
                fontweight="bold",
                fontstyle="normal",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=color,
                    alpha=0.15,
                    edgecolor=color,
                    linewidth=0.8,
                ),
                zorder=4,
            )

        legend_items = []
        seen_groups = []
        for group, color in {g: c for g, c in zip(groups, colors)}.items():
            if group not in seen_groups:
                patch = mpatches.Patch(color=color, label=group, alpha=0.8)
                legend_items.append(patch)
                seen_groups.append(group)

        legend1 = ax.legend(
            handles=legend_items,
            loc="upper left",
            fontsize=8,
            framealpha=0.9,
            edgecolor="#DDDDDD",
            title="Semantic Groups",
            title_fontsize=8,
        )
        ax.add_artist(legend1)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.savefig(
        "./600_outputs/tsne/embedding_space_tsne.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="#FAFAFA",
    )

    print("Plot saved as embedding_space_tsne.png")

    plt.show()


if __name__ == "__main__":
    texts, vectors, colors, groups = collect_all_embeddings()
    print(f"\nGot {len(texts)} embeddings, each of dimension {vectors.shape[1]}")

    coords_2d = reduce_with_tsne(vectors)

    plot_embedding_space(texts, coords_2d, colors, groups)
