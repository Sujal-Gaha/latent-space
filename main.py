import random

from datetime import datetime

from custom_datasets.WORDS import WORDS
from techniques.tsne.embedding_visualization.embedding_plotter import EmbeddingPlotter
from techniques.tsne.embedding_visualization.embedding_visualization import (
    collect_all_embeddings,
    collect_all_embeddings_V2,
    reduce_with_tsne,
)

from datasets import load_dataset

SENTENCES_DATASET = load_dataset(
    "agentlans/high-quality-english-sentences", split="train"
)

TEXTS = random.sample(SENTENCES_DATASET["text"], 300)

if __name__ == "__main__":
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    texts, vectors, colors, groups = collect_all_embeddings_V2(texts=TEXTS)
    print(f"\nGot {len(texts)} embeddings, each of dimension {vectors.shape[1]}")

    embedding_space = EmbeddingPlotter(session_id=session_id)

    coords_2d = reduce_with_tsne(vectors, n_components=2)
    embedding_space.plot(
        n_dim=2, texts=texts, coords=coords_2d, colors=colors, groups=groups
    )

    coords_3d = reduce_with_tsne(vectors, n_components=3)
    embedding_space.plot(
        n_dim=3, texts=texts, coords=coords_3d, colors=colors, groups=groups
    )
