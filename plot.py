import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def plot_umap(
    embeddings,
    labels,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    title="",
):
    """
    Helper function for visually exploring UMAP parameters.

    Usage:
        for n_neighbors in [0, 2, 5, 10, 15, 20]:
            plot_umap(embeddings, labels, n_neighbors=n_neighbors)
    """
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
    )
    u = fit.fit_transform(embeddings)
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], range(len(u)), c=labels)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], u[:, 1], c=labels)
    if n_components == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=labels, s=100)
    plt.title(title, fontsize=18)
    plt.show()


def plot_embeddings_umap(embeddings, labels):
    """
    Plot embeddings using Uniform Manifold Approximation and Projection.

    From https://umap-learn.readthedocs.io:
    The algorithm is founded on three assumptions about the data
    - The data is uniformly distributed on Riemannian manifold;
    - The Riemannian metric is locally constant (or can be approximated as such);
    - The manifold is locally connected.
    """
    scaled_embeddings = StandardScaler().fit_transform(embeddings)
    uma = umap.UMAP(
        n_neighbors=5,  # 10
        n_components=3,  # 2
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    projection = uma.fit_transform(scaled_embeddings)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        projection[:, 2],
        c=labels,
        cmap="RdYlGn",
        alpha=0.7,
        s=123,  # dot size
    )

    # Add index labels to points
    for idx, (x, y, z) in enumerate(projection):
        ax.text(x, y, z, str(idx), fontsize=8)

    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()
