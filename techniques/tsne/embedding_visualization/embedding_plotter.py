from datetime import datetime
from typing import TypedDict

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mpl_toolkits.mplot3d.axes3d import Axes3D

from numpy import ndarray

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class EmbeddingPlotterConfig(TypedDict):
    background_color: str
    text_color: str
    spine_color: str
    tick_color: str
    grid_color: str
    legend_edge_color: str
    output_dir: str
    dpi: int


class EmbeddingPlotter:
    def __init__(self) -> None:
        self.config = self._get_config()
        print("EmbeddingPlotter initialized")

    def _get_config(self) -> EmbeddingPlotterConfig:
        return {
            "background_color": "#FAFAFA",
            "text_color": "#2C3E50",
            "spine_color": "#DDDDDD",
            "tick_color": "#AAAAAA",
            "grid_color": "#EEEEEE",
            "legend_edge_color": "#DDDDDD",
            "output_dir": "./outputs/tsne/",
            "dpi": 150,
        }

    def plot(
        self,
        n_dim: int,
        texts: list[str],
        coords: ndarray,
        colors: list[str],
        groups: list[str],
    ) -> None:
        if n_dim not in (2, 3):
            raise ValueError(f"n_dim must be 2 or 3. Got: {n_dim}")
        if n_dim == 2:
            self._plot_2d(texts, coords, colors, groups)
        else:
            self._plot_3d(texts, coords, colors, groups)

    def _plot_2d(
        self,
        texts: list[str],
        coords: ndarray,
        colors: list[str],
        groups: list[str],
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))

        self._setup_figure(fig, len(texts), dim=2)

        self._configure_2d_axes(list(axes))

        for idx, ax in enumerate(axes):
            self._plot_2d_points(ax, texts, coords, colors)
            self._add_legend(ax, groups, colors)

            if idx == 0:
                self._plot_2d_labels(ax, texts, coords, colors)

        self._save_and_show("embedding_space_tsne_2d")

    def _plot_3d(
        self,
        texts: list[str],
        coords: ndarray,
        colors: list[str],
        groups: list[str],
    ):
        fig = plt.figure(figsize=(20, 9))

        self._setup_figure(fig, len(texts), dim=3)

        axes = [
            fig.add_subplot(1, 2, 1, projection="3d"),
            fig.add_subplot(1, 2, 2, projection="3d"),
        ]

        self._configure_3d_axes(axes)

        for idx, ax in enumerate(axes):
            self._plot_3d_points(ax, texts, coords, colors)
            self._add_legend(ax, groups, colors)

            if idx == 0:
                self._plot_3d_labels(ax, texts, coords, colors)

        self._save_and_show("embedding_space_tsne_3d")

    def _setup_figure(self, fig: Figure, n_items: int, dim: int) -> None:
        fig.patch.set_facecolor(self.config["background_color"])

        fig.suptitle(
            f"Semantic Embedding Space - t-SNE {dim}D Visualization\nMetric: cosine | {n_items}",
            fontsize=14,
            fontweight="bold",
            y=0.98,
            color=self.config["text_color"],
        )

    """
    AXIS CONFIGURATION
    """

    def _configure_2d_axes(self, axes: list[Axes]) -> None:
        for ax in axes:
            ax.set_facecolor("#FFFFFF")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.spines["left"].set_color(self.config["spine_color"])
            ax.spines["bottom"].set_color(self.config["spine_color"])

            ax.tick_params(colors=self.config["tick_color"])

            ax.set_xlabel("t-SNE Dimension 1", color=self.config["text_color"])
            ax.set_ylabel("t-SNE Dimension 2", color=self.config["text_color"])

            ax.axhline(y=0, color=self.config["grid_color"], linewidth=1)
            ax.axvline(x=0, color=self.config["grid_color"], linewidth=1)

    def _configure_3d_axes(self, axes: list[Axes3D]) -> None:
        for ax in axes:
            ax.set_facecolor("#FFFFFF")

            ax.set_xlabel("t-SNE Distribution 1", color=self.config["text_color"])
            ax.set_ylabel("t-SNE Distribution 2", color=self.config["text_color"])
            ax.set_zlabel("t-SNE Distribution 1", color=self.config["text_color"])

            ax.tick_params(colors=self.config["tick_color"])

    """
    Plotting helpers
    """

    def _plot_2d_points(
        self,
        ax: Axes,
        texts: list[str],
        coords: ndarray,
        colors: list[str],
    ) -> None:
        for idx, (text, color, coord) in enumerate(zip(texts, colors, coords), start=1):
            x, y = coord

            print(
                f"[{idx:3d}/{len(texts)}] Coordinate: ({x:.2f}, {y:.2f}) for text: '{text}'"
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

    def _plot_3d_points(
        self,
        ax: Axes3D,
        texts: list[str],
        coords: ndarray,
        colors: list[str],
    ) -> None:
        for idx, (text, color, coord) in enumerate(zip(texts, colors, coords), start=1):
            x, y, z = coord

            print(
                f"[{idx:3d}/{len(texts)}] Coordinate: ({x:.2f}, {y:.2f}, {z:.2f}) for text: '{text}'"
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

    def _plot_2d_labels(
        self,
        ax: Axes,
        texts: list[str],
        coords: ndarray,
        colors: list[str],
    ) -> None:
        for text, color, coord in zip(texts, colors, coords):
            label = text if len(text) <= 20 else text[:18] + "..."

            x, y = coord

            ax.annotate(
                label,
                xy=(x, y),
                xytext=(8, 6),
                textcoords="offset points",
                fontsize=8,
                color=self.config["text_color"],
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=color,
                    alpha=0.5,
                    edgecolor=color,
                    linewidth=0.8,
                ),
            )

    def _plot_3d_labels(
        self,
        ax: Axes3D,
        texts: list[str],
        coords: ndarray,
        colors: list[str],
    ) -> None:
        for text, color, coord in zip(texts, colors, coords):
            label = text if len(text) <= 20 else text[:18] + "..."

            x, y, z = coord

            ax.text(
                x,
                y,
                z,
                label,
                fontsize=8,
                color=self.config["text_color"],
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=color,
                    alpha=0.15,
                    edgecolor=color,
                    linewidth=0.8,
                ),
            )

    def _add_legend(
        self,
        ax: Axes | Axes3D,
        groups: list[str],
        colors: list[str],
    ) -> None:

        mapping = dict(zip(groups, colors))

        patches = [
            mpatches.Patch(color=color, label=group, alpha=0.8)
            for group, color in mapping.items()
        ]

        ax.legend(
            handles=patches,
            loc="upper left",
            fontsize=8,
            framealpha=0.9,
            edgecolor=self.config["legend_edge_color"],
            title="Semantic Groups",
            title_fontsize=8,
        )

    def _save_and_show(self, name: str) -> None:
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        path = f"{self.config["output_dir"]}{name}_{date_time}.png"

        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

        plt.savefig(
            path,
            dpi=self.config["dpi"],
            bbox_inches="tight",
            facecolor=self.config["background_color"],
        )

        print(f"Plot saved as {path}")

        plt.show()
