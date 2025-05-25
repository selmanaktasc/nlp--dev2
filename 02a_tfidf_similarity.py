import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df_lemma = pd.read_csv("cleaned_with_lemmatization.csv")
df_stem = pd.read_csv("cleaned_with_stemming.csv")
tfidf_lemma = pd.read_csv("tfidf_lemmatized.csv")
tfidf_stem = pd.read_csv("tfidf_stemmed.csv")

input_text_lemma = "['aynı', 'gün', 'kargo', 'veril', 'gün', 'sonra', 'sabah', 'elim', 'ulaş', 'paketle', 'güzel', 'şarj', 'stant', 'sorunsuz', 'çalış', '', 'şarj', 'ol', 'kesim', 'gayet', 'başarılı', 'tavsiye', 'eder']"
input_text_stem = "['aynı', 'gün', 'kargo', 'veril', 'gün', 'sonra', 'sabah', 'elim', 'ulaş', 'paketl', 'güzel', 'şarj', 'stant', 'sorunsuz', 'çalış', '', 'şarj', 'ol', 'kesim', 'gayet', 'başarılı', 'tavsiy', 'eder']"

index_lemma = df_lemma[df_lemma['metin_lemmatized'] == input_text_lemma].index[0]
index_stem = df_stem[df_stem['metin_stemmed'] == input_text_stem].index[0]

tfidf_lemma_vecs = tfidf_lemma.to_numpy()
tfidf_stem_vecs = tfidf_stem.to_numpy()

cos_lemma = cosine_similarity([tfidf_lemma_vecs[index_lemma]], tfidf_lemma_vecs).flatten()
cos_stem = cosine_similarity([tfidf_stem_vecs[index_stem]], tfidf_stem_vecs).flatten()

top5_lemma = np.argsort(cos_lemma)[::-1][1:6]
top5_stem = np.argsort(cos_stem)[::-1][1:6]

print(" TF-IDF Lemmatized Sonuçları:\n")
for i in top5_lemma:
    print(f"Index: {i}, Skor: {cos_lemma[i]:.4f}")
    print("Metin:", df_lemma.iloc[i]['metin_lemmatized'], "\n")

print(" TF-IDF Stemmed Sonuçları:\n")
for i in top5_stem:
    print(f"Index: {i}, Skor: {cos_stem[i]:.4f}")
    print("Metin:", df_stem.iloc[i]['metin_stemmed'], "\n")
