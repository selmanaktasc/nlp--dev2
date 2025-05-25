import pandas as pd

df_stem = pd.read_csv("cleaned_with_stemming.csv")

# İçinde 'şarj', 'tavsiye', 'paketle' gibi anahtar kelimeler geçen satırları bul
filtered = df_stem[df_stem['metin_stemmed'].str.contains("şarj", case=False, na=False)]

# İlk 10 tanesini göster
for i, row in filtered.head(10).iterrows():
    print(f"\nIndex: {i}\n{row['metin_stemmed']}")
