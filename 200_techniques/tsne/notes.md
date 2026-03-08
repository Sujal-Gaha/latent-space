## LATENT SPACE VISUALIZATION - t-SNE

t-Distributed Stochastic Neighbor Embedding

## What is t-SNE?

t-SNE is a NON-LINEAR dimensionality reduction technique designed specifically for visualization.

It works by:

1. Measuring SIMILARITY between points in high-D space (using a Gaussian probability distribution)

2. Trying to reproduce those similarities in 2D (using a t-distribution - hence the "t")

3. Minimizing the "mismatch" between the two distributions using gradient descent

The t-distribution has heavier tails than Gaussian, which helps push dissimilar clusters far apart -> cleaner separation.

## KEY PARAMETERS: PERPLEXITY

- Controls how many "neighbors" each point cares about
- Low (-5) -> very tight local clusters, may fragment real clusters
- Mid (-30) -> good balance (recommended default)
- High (-100) -> more global structure, but clusters may merge

## PROS

- Excellent at revealing cluster structure
- Handles non-linear relationships

## CONS

- Slow on large datasets
- Non-deterministic (random_state needed for reproducibility)
- Distances between clusters are NOT meaningful
- Perplexity must be tuned
