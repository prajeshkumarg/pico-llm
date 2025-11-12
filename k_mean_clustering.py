# --- Imports ---
import json
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import plotly.express as px

# --- Load your data and models ---
# Assuming you have the clusters saved in JSON and embeddings available
from gensim.models import Word2Vec

# Load Word2Vec model (replace path if saved elsewhere)
w2v_model = Word2Vec.load("w2v_tinystories.model")

# Load clusters JSON (generated from your previous code)
CLUSTERS = 200
with open(f"tinystories_clusters_with_word2vec_{CLUSTERS}.json", "r", encoding="utf-8") as f:
    clusters = json.load(f)

# --- Prepare data for visualization ---
words = []
vectors = []
labels = []

for cluster_id, word_list in clusters.items():
    for word in word_list:
        if word in w2v_model.wv:
            words.append(word)
            vectors.append(w2v_model.wv[word])
            labels.append(int(cluster_id))

vectors = np.array(vectors)
vectors = normalize(vectors)

print(f"✅ Total words for visualization: {len(words)}")

# --- Dimensionality Reduction (TSNE for 2D) ---
print("🎨 Running t-SNE (this can take a few minutes)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
emb_2d = tsne.fit_transform(vectors)

# --- Build DataFrame for Plotly ---
df = pd.DataFrame({
    "word": words,
    "x": emb_2d[:, 0],
    "y": emb_2d[:, 1],
    "cluster": labels
})

# --- Interactive Plot ---
fig = px.scatter(
    df, x="x", y="y",
    color="cluster",
    hover_data=["word"],
    title="TinyStories Word2Vec Clusters (Interactive)",
    template="plotly_white",
    color_continuous_scale="Rainbow"
)

fig.update_traces(marker=dict(size=6, opacity=0.8, line=dict(width=0)))
fig.update_layout(
    width=900, height=700,
    showlegend=False,
    title_font=dict(size=20)
)

fig.show()
