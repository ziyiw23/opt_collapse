import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Style setup
plt.style.use('ggplot')
sns.set_palette("husl")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150, 'savefig.dpi': 300})

import lmdb
import pickle
import gzip
import numpy as np

def load_lmdb_embeddings(lmdb_path):
    """Loads embeddings from a compressed LMDB file and returns NumPy arrays."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    embeddings = []
    labels = []

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            try:
                decompressed_value = gzip.decompress(value)  # Decompress GZIP data
                entry = pickle.loads(decompressed_value)  # Now unpickle the object
                embeddings.append(entry)
                labels.append(key.decode())  # Convert key from bytes to string
            except Exception as e:
                print(f"❌ Error loading key {key}: {e}")
                continue

    return np.array(embeddings), labels

def plot_cosine_similarity(embeddings, metadata):
    """Enhanced similarity visualization"""
    cos_sim = cosine_similarity(embeddings)
    mask = np.triu_indices_from(cos_sim, k=1)  # Exclude diagonal
    similarities = cos_sim[mask]
    
    # Bin by confidence (assuming confidence in 0-100)
    metadata['confidence_bin'] = pd.cut(metadata['confidence'],
                                      bins=[0, 20, 40, 60, 80, 100],
                                      labels=['0-20', '20-40', '40-60', '60-80', '80-100'])
    
    plt.figure(figsize=(8, 5))
    sns.histplot(x=similarities, hue=metadata['confidence_bin'], 
                 element="step", stat="density", common_norm=False,
                 palette="viridis", alpha=0.7)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.title("Pairwise Embedding Similarity Distribution\nColored by Confidence Bins")
    plt.tight_layout()

def plot_tsne(embeddings, metadata, perplexity=30):
    """Enhanced t-SNE visualization"""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=metadata['confidence'],  # Color by confidence
        cmap="viridis",
        s=20,
        alpha=0.7,
        edgecolor='w',
        linewidth=0.3
    )
    
    plt.colorbar(scatter, label='Confidence Score')
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE Projection of Protein Embeddings\nColored by Confidence")
    plt.tight_layout()

def plot_embedding_similarity_vs_confidence(embeddings, metadata):
    """New: Correlation plot between similarity and confidence"""
    cos_sim = cosine_similarity(embeddings)
    upper_tri = np.triu_indices_from(cos_sim, k=1)
    
    pairs = pd.DataFrame({
        'sim': cos_sim[upper_tri],
        'conf_i': np.repeat(metadata['confidence'], cos_sim.shape[0])[upper_tri[0]],
        'conf_j': np.repeat(metadata['confidence'], cos_sim.shape[0])[upper_tri[1]]
    })
    pairs['conf_mean'] = (pairs['conf_i'] + pairs['conf_j']) / 2
    
    plt.figure(figsize=(8, 5))
    sns.regplot(x='conf_mean', y='sim', data=pairs,
                scatter_kws={'alpha':0.3, 's':10},
                line_kws={'color':'red'})
    plt.xlabel("Mean Confidence of Pair")
    plt.ylabel("Cosine Similarity")
    plt.title("Embedding Similarity vs. Confidence Correlation")
    plt.tight_layout()

# Usage
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path', type=str, required=True, help="Path to LMDB file")
    args = parser.parse_args()

    embeddings, metadata = load_lmdb_embeddings(args.lmdb_path)

    plot_cosine_similarity(embeddings, metadata)
    plot_tsne(embeddings, metadata)
    plot_embedding_similarity_vs_confidence(embeddings, metadata)
    plt.show()

if __name__ == "__main__":
    main()

