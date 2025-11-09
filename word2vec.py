from datasets import load_dataset
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import nltk, json, re
from nltk.corpus import stopwords
from collections import Counter

# --- NLTK setup ---
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# --- Load TinyStories (streaming) ---
ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

stop = set(stopwords.words("english"))
word_counts = Counter()
sentences = []  # for Word2Vec training

MAX_STORIES = 23000       # adjust as needed
TOP_WORDS = 1000
CLUSTERS = 200
OUTPUT_FILE = f"tinystories_clusters_with_word2vec_{CLUSTERS}.json"

print("📦 Collecting words...")

# --- Collect vocabulary and sentences ---
for i, row in enumerate(ds):
    text = row["text"].lower()
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalpha() and w not in stop]

    word_counts.update(words)
    sentences.append(words)  # store tokenized sentence for Word2Vec

    if i >= MAX_STORIES:
        break

vocab = [w for w, _ in word_counts.most_common(TOP_WORDS)]
print(f"✅ Vocabulary collected: {len(vocab)} words")

# --- Train Word2Vec model ---
print("🧠 Training Word2Vec model...")
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,   # dimensionality of embeddings
    window=5,
    min_count=2,
    workers=4,
    sg=1               # skip-gram for better semantic learning
)

# --- Extract embeddings for top words ---
emb = []
missing_words = []

for word in vocab:
    if word in w2v_model.wv:
        emb.append(w2v_model.wv[word])
    else:
        missing_words.append(word)

print(f"✅ Embeddings created for {len(emb)} words (missing: {len(missing_words)})")

# --- K-Means clustering ---
print("📊 Clustering words...")
kmeans = KMeans(n_clusters=CLUSTERS, random_state=42).fit(emb)
labels = kmeans.labels_

clusters = {i: [] for i in range(CLUSTERS)}
for word, label in zip(vocab, labels):
    if word not in missing_words:
        clusters[int(label)].append(word)

#-----save model ------------
w2v_model.save("w2v_tinystories.model")
print("💾 Saved Word2Vec model to 'w2v_tinystories.model'")

# --- Save to JSON ---
print(f"💾 Saving clusters to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(clusters, f, ensure_ascii=False, indent=2)

print("✅ Done!")