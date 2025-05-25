# NLP Ödev-2: Metin Benzerlik Hesaplama ve Model Karşılaştırması

Bu proje, e-ticaret ürün yorumları veri seti üzerinde TF-IDF ve Word2Vec modellerini kullanarak metin benzerlik hesaplama ve model performanslarını karşılaştırma çalışmasını içermektedir.

## 📋 Proje Özeti

Bu ödevde, önceden hazırlanmış temizlenmiş veri setleri ve eğitilmiş modeller kullanılarak:
- TF-IDF tabanlı metin benzerlik hesaplama
- Word2Vec tabanlı metin benzerlik hesaplama  
- Model performanslarının anlamsal değerlendirmesi
- Jaccard benzerliği ile model tutarlılığı analizi


##Gereksinimler

### Python Kütüphaneleri
```bash
pip install pandas numpy scikit-learn gensim
```

### Gerekli Dosyalar
- Python 3.7+
- Pandas
- NumPy
- Scikit-learn
- Gensim

## Kullanım Talimatları

### Hızlı Başlangıç (Önerilen)

Tüm analizleri tek seferde çalıştırmak için:

```bash
# Gerekli kütüphaneleri yükle
pip install -r requirements.txt

# Tüm analizleri çalıştır
python run_similarity_analysis.py
```

### Manuel Çalıştırma

### 1. TF-IDF Benzerlik Hesaplama

```bash
python 02a_tfidf_similarity.py
```

Bu script:
- Lemmatized ve stemmed TF-IDF vektörlerini yükler
- Örnek giriş metni için en benzer 5 metni bulur
- Cosine similarity skorlarını hesaplar

### 2. Word2Vec Benzerlik Hesaplama

```bash
python 02b_word2vec_similarity.py
```

Bu script:
- 16 farklı Word2Vec modelini yükler
- Her model için örnek giriş metni ile en benzer 5 metni bulur
- Ortalama vektör temsili kullanarak cosine similarity hesaplar

### 3. Jaccard Benzerlik Matrisi

```bash
python 03_jaccard_similarity_matrix.py
```

Bu script:
- Tüm modellerin sonuçları arasında Jaccard benzerliği hesaplar
- 18x18 boyutunda benzerlik matrisi oluşturur
- Model tutarlılığını analiz eder

## Veri Seti Bilgileri

**Veri Seti:** E-ticaret Ürün Yorumları
- **Boyut:** 5,001 yorum
- **Sütunlar:** 
  - `Metin`: Orijinal yorum metni
  - `Durum`: Sentiment etiketi (0: Negatif, 1: Pozitif, 2: Nötr)

**Örnek Giriş Metni:**
```
"Aynı gün kargoya verildi 1 gün sonra sabah elime ulaştı paketleme güzel, şarj standı sorunsuz çalışıyor usb ile de şarj oluyor kesimi gayet başarılı tavsiye ederim."
```

##  Model Konfigürasyonları

### TF-IDF Modelleri
- **TF-IDF Lemmatized:** Lemmatized metinler üzerinde TF-IDF
- **TF-IDF Stemmed:** Stemmed metinler üzerinde TF-IDF

### Word2Vec Modelleri (16 adet)
| Preprocessing | Algorithm | Window Size | Vector Dimension |
|---------------|-----------|-------------|------------------|
| Lemmatized    | CBOW      | 2, 4        | 100, 300         |
| Lemmatized    | Skip-gram | 2, 4        | 100, 300         |
| Stemmed       | CBOW      | 2, 4        | 100, 300         |
| Stemmed       | Skip-gram | 2, 4        | 100, 300         |

## Değerlendirme Metrikleri

### 1. Anlamsal Değerlendirme (Subjective Evaluation)
- **Puan Sistemi:** 1-5 arası
  - 1: Çok alakasız, anlamca zayıf benzerlik
  - 2: Kısmen ilgili ama bağlamı tutmuyor
  - 3: Ortalama düzeyde benzer
  - 4: Anlamlı, açık benzerlik içeriyor
  - 5: Neredeyse aynı temada, çok güçlü benzerlik

### 2. Sıralama Tutarlılığı (Ranking Agreement)
- **Jaccard Benzerliği:** İki modelin önerdiği ilk 5 sonuç arasındaki örtüşme
- **Formül:** J(A,B) = |A ∩ B| / |A ∪ B|

## Çıktı Formatı

### TF-IDF Sonuçları
```
TF-IDF Lemmatized Sonuçları:
Index: 1898, Skor: 0.8234
Metin: ['benzer', 'yorum', 'metni']

TF-IDF Stemmed Sonuçları:
Index: 1898, Skor: 0.8234
Metin: ['benzer', 'yorum', 'metni']
```

### Word2Vec Sonuçları
```
İşleniyor: word2vec_lemmatized_cbow_win2_dim100.model
En Benzer 5 Cümle:

1. Index: 25, Skor: 0.7845
   Metin: ['benzer', 'yorum', 'metni']
```

### Jaccard Benzerlik Matrisi
```
                    TFIDF_Lemma  TFIDF_Stem  W2V_L_CBOW_w2_d100  ...
TFIDF_Lemma              1.000       1.000               0.200  ...
TFIDF_Stem               1.000       1.000               0.200  ...
W2V_L_CBOW_w2_d100       0.200       0.200               1.000  ...
```

## Beklenen Sonuçlar

1. **Model Performans Karşılaştırması:** Hangi modellerin daha anlamlı sonuçlar ürettiği
2. **Preprocessing Etkisi:** Lemmatization vs Stemming karşılaştırması
3. **Word2Vec Parametre Analizi:** Window size ve vector dimension etkisi
4. **Model Tutarlılığı:** Hangi modellerin benzer sonuçlar ürettiği
