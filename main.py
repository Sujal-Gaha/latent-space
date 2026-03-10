from custom_datasets.WORDS import WORDS
from techniques.tsne.embedding_visualization.embedding_space import EmbeddingSpace
from techniques.tsne.embedding_visualization.embedding_visualization import (
    collect_all_embeddings,
    reduce_with_tsne,
)

if __name__ == "__main__":
    texts, vectors, colors, groups = collect_all_embeddings(WORDS=WORDS)
    print(f"\nGot {len(texts)} embeddings, each of dimension {vectors.shape[1]}")

    embedding_space = EmbeddingSpace()

    coords_2d = reduce_with_tsne(vectors, n_components=2)
    embedding_space.plot(
        n_dim=2, texts=texts, coords=coords_2d, colors=colors, groups=groups
    )

    # coords_3d = reduce_with_tsne(vectors, n_components=3)
    # embedding_space.plot(
    #     n_dim=3, texts=texts, coords=coords_3d, colors=colors, groups=groups
    # )
