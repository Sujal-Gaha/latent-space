import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from numpy import ndarray


class EmbeddingSpace:
    def __init__(
        self,
    ):
        print("EmbeddingSpace class initialized")

    def plot(
        self,
        n_dim: int,
        texts: list[str],
        coords: ndarray,
        colors: list[str],
        groups: list[str],
    ):
        if n_dim == 2:
            for coord in coords:
                if len(coord) != 2:
                    raise ValueError(f"Coord is not 2D. Got: {coord}")
            self._plot_2d(texts, coords, colors, groups)
        elif n_dim == 3:
            self._plot_3d(texts, coords, colors, groups)
        else:
            raise ValueError(f"n_dim must be 2 or 3. Got: {n_dim}")

    def _plot_2d(
        self,
        texts: list[str],
        coords: ndarray,
        colors: list[str],
        groups: list[str],
    ):
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        fig.patch.set_facecolor("#FAFAFA")
        fig.suptitle(
            "Semantic Embedding Space - t-SNE 2D Visualization\n"
            f"Metric: cosine | {len(texts)}",
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
            "./outputs/tsne/embedding_space_tsne_2d.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="#FAFAFA",
        )

        print("Plot saved as embedding_space_tsne_2d.png")

        plt.show()

    def _plot_3d(
        self,
        texts: list[str],
        coords: ndarray,
        colors: list[str],
        groups: list[str],
    ):
        fig = plt.figure(figsize=(20, 9))
        fig.patch.set_facecolor("#FAFAFA")

        fig.suptitle(
            "Semantic Embedding Space - t-SNE 3D Visualization\n"
            f"Metric: cosine | {len(texts)}",
            fontsize=14,
            fontweight="bold",
            y=0.98,
            color="#2C3E50",
        )

        axes = [
            fig.add_subplot(1, 2, 1, projection="3d"),
            fig.add_subplot(1, 2, 2, projection="3d"),
        ]

        for ax_idx, ax in enumerate(axes):
            ax.set_facecolor("#FFFFFF")

            ax.set_xlabel("t-SNE Dimension 1", color="#888888", fontsize=10)
            ax.set_xlabel("t-SNE Dimension 2", color="#888888", fontsize=10)
            ax.set_zlabel("t-SNE Dimension 3", color="#888888", fontsize=10)

            ax.tick_params(colors="#AAAAAA")

            count = 0
            for i, (text, color) in enumerate(zip(texts, colors)):
                x, y, z = coords[i]
                count = i + 1

                print(
                    f"[{count:3d}/{len(texts)}] Coordinate: ({x:.2f}, {y:.2f}, {z:.2f}) for text: '{text}'"
                )

                ax.scatter(
                    x,
                    y,
                    z,
                    c=color,
                    s=180,
                    edgecolors="white",
                    linewidths=1.5,
                    alpha=0.9,
                    depthshade=True,
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
                x, y, z = coords[i]

                display = text_str if len(text_str) <= 20 else text_str[:18] + "..."

                ax.text(
                    x,
                    y,
                    z,
                    display,
                    fontsize=8,
                    color="#2C3E50",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor=color,
                        alpha=0.15,
                        edgecolor=color,
                        linewidth=0.8,
                    ),
                )

            legend_items = []
            seen_groups = []

            for group, color in {g: c for g, c in zip(groups, colors)}.items():
                if group not in seen_groups:
                    patch = mpatches.Patch(color=color, label=group, alpha=0.8)
                    legend_items.append(patch)
                    seen_groups.append(group)

            legend = ax.legend(
                handles=legend_items,
                loc="upper left",
                fontsize=8,
                framealpha=0.9,
                edgecolor="#DDDDDD",
                title="Semantic Groups",
                title_fontsize=8,
            )

            ax.add_artist(legend)

        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

        plt.savefig(
            "./outputs/tsne/embedding_space_tsne_3d.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="#FAFAFA",
        )

        print("Plot saved as embedding_space_tsne_3d.png")

        plt.show()
