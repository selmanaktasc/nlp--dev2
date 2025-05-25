import pandas as pd

A = {25, 1813, 1417, 2133, 2125}
B = {25, 1421, 1417, 2210, 4530}

intersection = A & B
union = A | B
jaccard_score = len(intersection) / len(union)

print("Jaccard Örnek Hesaplama:")
print("A =", A)
print("B =", B)
print("Kesişim:", intersection)
print("Birleşim:", union)
print("Jaccard Skoru:", round(jaccard_score, 3))  
print()

top5_indices = {
    "TFIDF_Lemma":     [1898, 665, 3701, 795, 2731],
    "TFIDF_Stem":      [1898, 665, 3701, 795, 2731],
    "W2V_L_CBOW_w2_d100": [25, 1813, 1417, 2133, 2125],
    "W2V_L_CBOW_w2_d300": [25, 1421, 1417, 2210, 4530],
    "W2V_L_CBOW_w4_d100": [25, 1813, 1417, 4740, 2457],
    "W2V_L_CBOW_w4_d300": [25, 2210, 1417, 1813, 1421],
    "W2V_L_SG_w2_d100":   [25, 1421, 1813, 2457, 4783],
    "W2V_L_SG_w2_d300":   [25, 4740, 1421, 1813, 3569],
    "W2V_L_SG_w4_d100":   [25, 1421, 2457, 1813, 1708],
    "W2V_L_SG_w4_d300":   [25, 1421, 1708, 1813, 2457],
    "W2V_S_CBOW_w2_d100": [25, 1813, 1421, 1782, 446],
    "W2V_S_CBOW_w2_d300": [25, 1417, 486, 4013, 2210],
    "W2V_S_CBOW_w4_d100": [25, 1813, 4013, 3569, 181],
    "W2V_S_CBOW_w4_d300": [25, 1813, 1417, 391, 4740],
    "W2V_S_SG_w2_d100":   [25, 1421, 1813, 4783, 2457],
    "W2V_S_SG_w2_d300":   [25, 1421, 1813, 4740, 796],
    "W2V_S_SG_w4_d100":   [25, 1421, 796, 1708, 3192],
    "W2V_S_SG_w4_d300":   [25, 1421, 2457, 3192, 1708],
}

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)

model_names = list(top5_indices.keys())
jaccard_df = pd.DataFrame(index=model_names, columns=model_names)

for m1 in model_names:
    for m2 in model_names:
        jaccard_df.loc[m1, m2] = round(jaccard(top5_indices[m1], top5_indices[m2]), 3)

jaccard_df = jaccard_df.astype(float)

print(jaccard_df)
