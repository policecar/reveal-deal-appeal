from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
import umap.plot
from datasets import Dataset
from sklearn.preprocessing import StandardScaler

from config import Config


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def compute_embeddings(
    texts,
    model,
    tokenizer,
    config: Config,
    device="cuda",
    batch_size=4,
    chunk_overlap=100,
):
    """
    Embed texts with len > max sequence length by splitting each text into
    overlapping chunks, embedding each chunk, and then aggregating the embeddings
    using a weighted average / attention pooling.
    """

    max_seq_length = (
        model.config.max_position_embeddings
    )  # TODO: this won't work for all models
    chunk_size = max_seq_length - 2  # Account for special tokens

    all_chunks = []
    chunk_lengths = []
    text_bounds = []

    for text in texts:
        # Split text into smaller chunks before tokenization
        text_chunks = [
            text[i : i + chunk_size * 4]
            for i in range(0, len(text), chunk_size * 4 - chunk_overlap * 4)
        ]
        text_start = len(all_chunks)

        for chunk in text_chunks:
            tokens = tokenizer.encode(chunk, add_special_tokens=False)[:chunk_size]
            tokens = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]
            all_chunks.append(tokens)
            chunk_lengths.append(len(tokens))

        text_bounds.append((text_start, len(all_chunks)))

    chunk_embeddings = []
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i : i + batch_size]
        max_len = max(len(chunk) for chunk in batch_chunks)

        padded_chunks = [
            chunk + [tokenizer.pad_token_id] * (max_len - len(chunk))
            for chunk in batch_chunks
        ]

        attention_masks = [
            [1] * len(chunk) + [0] * (max_len - len(chunk)) for chunk in batch_chunks
        ]

        inputs = {
            "input_ids": torch.tensor(padded_chunks).to(device),
            "attention_mask": torch.tensor(attention_masks).to(device),
        }

        with torch.no_grad():
            outputs = model(**inputs)
            masked_embeddings = outputs.last_hidden_state * inputs[
                "attention_mask"
            ].unsqueeze(-1)
            sums = masked_embeddings.sum(dim=1)
            counts = inputs["attention_mask"].sum(dim=1, keepdim=True)
            batch_embeddings = sums / counts
            chunk_embeddings.append(batch_embeddings)

    all_embeddings = torch.cat(chunk_embeddings)

    final_embeddings = []
    for start, end in text_bounds:
        text_chunk_embeddings = all_embeddings[start:end]
        weights = torch.tensor(chunk_lengths[start:end]).to(device)
        weights = weights / weights.sum()
        text_embedding = (text_chunk_embeddings * weights.unsqueeze(-1)).sum(dim=0)
        final_embeddings.append(text_embedding)

    return torch.stack(final_embeddings).cpu().numpy()


def save_embeddings(embeddings, labels, dirpath, prefix=""):
    """Save embeddings and labels to separate files."""
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    np.save(f"{dirpath}/{prefix}_embeddings.npy", embeddings)
    np.save(f"{dirpath}/{prefix}_labels.npy", labels)


def load_embeddings(dirpath, prefix=""):
    """Load embeddings and labels from files."""
    embeddings = np.load(
        f"{dirpath}/{prefix}_embeddings.npy", mmap_mode="r"
    )  # memory-mapping for large files
    labels = np.load(f"{dirpath}/{prefix}_labels.npy")
    return embeddings, labels


def draw_umap(
    embeddings,
    labels,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    title="",
):
    """
    Helper function for exploring UMAP parameters.
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

    plt.figure(figsize=(10, 8))
    plt.scatter(
        projection[:, 0],
        projection[:, 1],
        projection[:, 2],
        c=labels,
        cmap="RdYlGn",
        alpha=0.7,
    )
    plt.colorbar()
    plt.show()
