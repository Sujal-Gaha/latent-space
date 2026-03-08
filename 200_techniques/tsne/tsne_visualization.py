import time

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1. Load & Preprocess Data
digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_pca50 = PCA(n_components=30).fit_transform(X_scaled)
print(f"Pre-reduced from {X_scaled.shape[1]}D -> 30D with PCA (speeds up t-SNE)")

# 2. Run t-SNE with Different Perplexity Values
perplexities = [5, 30, 50]
results = {}

for perp in perplexities:
    print(f"Running t-SNE with perplexity={perp}...", end=" ", flush=True)
    t0 = time.time()
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        n_iter_without_progress=1000,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    results[perp] = tsne.fit_transform(X_pca50)
    print(f"done in {time.time()-t0:.1f}s")


# 3. Plot
colors = plt.cm.tab10(np.linspace(0, 1, 10))

fig = plt.figure(figsize=(17, 5))
fig.suptitle(
    "t-SNE - t-Distributed Stochastic Neighbor Embedding\n"
    "Comparing Perplexity Settings",
    fontsize=14,
    fontweight="bold",
)

axes = []
for idx, perp in enumerate(perplexities):
    ax = fig.add_subplot(1, 3, idx + 1)
    axes.append(ax)
    X_2d = results[perp]

    for digit in range(10):
        mask = y == digit
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=[colors[digit]],
            label=str(digit),
            alpha=0.65,
            s=12,
            edgecolors="none",
        )

    for digit in range(10):
        mask = y == digit
        cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
        ax.annotate(
            str(digit),
            (cx, cy),
            fontsize=10,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75),
        )

    title_notes = {
        5: "(Too local - fragmented)",
        30: "(Recommended - balanced)",
        50: "(More global)",
    }
    ax.set_title(
        f"Perplexity = {perp}\n{title_notes[perp]}", fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, alpha=0.25)

    if idx == 2:
        ax.legend(
            title="Digit",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            markerscale=2,
            fontsize=8,
        )


insight_text = (
    "IMPORTANT: In t-SNE plots, distances BETWEEN clusters\n"
    "are not interpretable. Only cluster shape and membership matter.\n"
    "Two clusters being far apart does NOT mean they are very different."
)
fig.text(
    0.5,
    -0.04,
    insight_text,
    ha="center",
    fontsize=9.5,
    bbox=dict(boxstyle="round", facecolor="#fff3cd", alpha=0.9, edgecolor="#ffc107"),
)

plt.tight_layout()
plt.savefig(
    "./600_outputs/tsne/tsne_visualization.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
print("\nPlot saved to: ./600_outputs/tsne/tsne_visualization.png")

# 4. Single Best t-SNE with richer styling
fig2, ax = plt.subplots(figsize=(9, 7))
X_best = results[30]

for digit in range(10):
    mask = y == digit
    ax.scatter(
        X_best[mask, 0],
        X_best[mask, 1],
        c=[colors[digit]],
        label=f"Digit {digit}",
        alpha=0.7,
        s=18,
        edgecolors="none",
    )


for digit in range(10):
    mask = y == digit
    cx, cy = X_best[mask, 0].mean(), X_best[mask, 1].mean()
    ax.annotate(
        str(digit),
        (cx, cy),
        fontsize=13,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[digit], alpha=0.9),
    )

ax.set_title(
    "t-SNE Latent Space - Handwritten Digits\n(Perplexity=30, 1000 iterations)",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax.set_facecolor("#1a1a2e")
ax.tick_params(colors="gray")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")

plt.tight_layout()
plt.savefig(
    "./600_outputs/tsne/tsne_best.png", dpi=150, bbox_inches="tight", facecolor="white"
)
print("Plot saved to: ./600_outputs/tsne/tsne_best.png")
plt.show()
