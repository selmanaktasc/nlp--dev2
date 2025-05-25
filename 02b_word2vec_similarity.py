import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import os

os.chdir("Word2vec")

lemma_tokens = ['aynı', 'gün', 'kargo', 'veril', 'gün', 'sonra', 'sabah', 'elim', 'ulaş', 'paketle', 'güzel', 'şarj', 'stant', 'sorunsuz', 'çalış', '', 'şarj', 'ol', 'kesim', 'gayet', 'başarılı', 'tavsiye', 'eder']
stem_tokens = ['aynı', 'gün', 'kargo', 'veril', 'gün', 'sonra', 'sabah', 'elim', 'ulaş', 'paketl', 'güzel', 'şarj', 'stant', 'sorunsuz', 'çalış', '', 'şarj', 'ol', 'kesim', 'gayet', 'başarılı', 'tavsiy', 'eder']

df_lemma = pd.read_csv("../cleaned_with_lemmatization.csv")
df_stem = pd.read_csv("../cleaned_with_stemming.csv")


model_files = sorted([f for f in os.listdir() if f.endswith(".model")])

def get_average_vector(model, tokens):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return None

for model_file in model_files:
    print(f"\n İşleniyor: {model_file}")
    model = Word2Vec.load(model_file)

    if "lemmatized" in model_file:
        df = df_lemma
        input_vec = get_average_vector(model, lemma_tokens)
        content_column = "metin_lemmatized"
    else:
        df = df_stem
        input_vec = get_average_vector(model, stem_tokens)
        content_column = "metin_stemmed"

    if input_vec is None:
        print("Giriş metni vektörü oluşturulamadı, model kelimeleri içermiyor.")
        continue

    doc_vectors = []
    valid_indices = []

    for i, row in df.iterrows():
        tokens = eval(row[content_column]) if isinstance(row[content_column], str) and row[content_column].startswith("[") else row[content_column].split()
        vec = get_average_vector(model, tokens)
        if vec is not None:
            doc_vectors.append(vec)
            valid_indices.append(i)

    sims = cosine_similarity([input_vec], doc_vectors).flatten()
    top_indices = np.argsort(sims)[::-1][:5]

    print("En Benzer 5 Cümle:\n")
    for rank, idx in enumerate(top_indices):
        real_index = valid_indices[idx]
        score = sims[idx]
        print(f"{rank+1}. Index: {real_index}, Skor: {score:.4f}")
        print("   Metin:", df.iloc[real_index][content_column], "\n")
