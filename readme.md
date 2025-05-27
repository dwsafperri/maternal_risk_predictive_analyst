# Proyek Prediksi Risiko Kesehatan Maternal

## Domain Proyek

Kesehatan maternal (ibu hamil) merupakan aspek krusial dalam sistem pelayanan kesehatan karena berhubungan langsung dengan keselamatan ibu dan bayi. Menurut World Health Organization (WHO), lebih dari 700 wanita meninggal setiap hari sepanjang tahun 2023 akibat penyebab yang sebenarnya dapat dicegah terkait kehamilan dan persalinan. Artinya, setiap 2 menit terjadi satu kematian maternal di dunia.

Walaupun telah terjadi penurunan sekitar 40% angka kematian maternal secara global sejak tahun 2000, angka ini masih tinggi di negara berpendapatan rendah dan menengah ke bawah. Lebih dari 90% kematian maternal tahun 2023 terjadi di kelompok negara ini. Padahal, kematian tersebut dapat dicegah dengan perawatan tepat dari tenaga kesehatan profesional, baik sebelum, saat, maupun setelah persalinan.

Dengan bantuan teknologi machine learning, kita dapat membangun sistem prediksi risiko kesehatan maternal untuk membantu tenaga medis dalam mengidentifikasi ibu hamil berisiko tinggi. Deteksi dini ini penting untuk mempercepat penanganan medis dan menyelamatkan nyawa.

---

## Business Understanding

### Problem Statements

- Bagaimana cara mengklasifikasikan risiko kesehatan maternal (low, mid, high) berdasarkan data vital pasien?
- Apakah model machine learning dapat memberikan klasifikasi risiko yang akurat dan dapat diandalkan dalam pengambilan keputusan medis?
- Fitur apa saja yang paling berkontribusi terhadap klasifikasi risiko maternal, dan bagaimana pengaruhnya terhadap tingkat akurasi model?

### Goals

- Mengembangkan sistem klasifikasi risiko kesehatan maternal berbasis data medis dasar.
- Menerapkan dan membandingkan beberapa algoritma machine learning untuk mendapatkan model terbaik.
- Mengevaluasi performa model menggunakan metrik evaluasi yang sesuai (confusion matrix, akurasi, dsb).

### Solution Statements

- Melakukan analisis statistik dan membangun model machine learning untuk mengukur kontribusi parameter medis (seperti tekanan darah, kadar hemoglobin, usia kehamilan, dll) terhadap klasifikasi risiko.
- Menerapkan beberapa algoritma machine learning dan membandingkan performa akurasi model dalam memprediksi risiko kesehatan maternal.
- Memberikan insight medis melalui analisis kombinasi variabel terhadap tingkat risiko maternal untuk membantu deteksi dini potensi komplikasi kehamilan.

---

## Metodologi

Metodologi yang digunakan adalah klasifikasi multi-kelas dengan beberapa algoritma machine learning untuk mengelompokkan risiko menjadi tiga kategori:

- **Low Risk**
- **Mid Risk**
- **High Risk**

Langkah utama dalam proyek ini:

1. Eksplorasi dan pemahaman data
2. Preprocessing dan pembagian data
3. Penerapan algoritma klasifikasi seperti:
   - SVM
   - KNN
   - Random Forest
   - XGBoost
4. Evaluasi performa model menggunakan confusion matrix dan metrik akurasi

---

## Metrik Evaluasi

- **Confusion Matrix**: Untuk mengevaluasi jumlah prediksi benar dan salah dari masing-masing kelas.
- **Accuracy**: Untuk mengetahui persentase prediksi yang benar secara keseluruhan.
- (Opsional) Precision, Recall, dan F1-Score untuk evaluasi tambahan.

---

## Data Understanding

- **Sumber Dataset**: [Maternal Health Risk Data - UCI Repository](https://archive.ics.uci.edu/dataset/863/maternal+health+risk)
- **Jumlah Data**: 1.014 baris (data pasien ibu hamil)
- **Jumlah Fitur**: 7 kolom (fitur) termasuk label target
- **Missing Values**: Tidak ada

### Deskripsi Variabel

| Fitur       | Tipe     | Deskripsi                                                          |
| ----------- | -------- | ------------------------------------------------------------------ |
| Age         | Numerik  | Usia ibu hamil (tahun)                                             |
| SystolicBP  | Numerik  | Tekanan darah sistolik (mm Hg)                                     |
| DiastolicBP | Numerik  | Tekanan darah diastolik (mm Hg)                                    |
| BS          | Numerik  | Kadar gula darah (mg/dl)                                           |
| BodyTemp    | Numerik  | Suhu tubuh dalam °C (ada yang menuliskan °F, perlu validasi ulang) |
| HeartRate   | Numerik  | Detak jantung per menit (bpm)                                      |
| RiskLevel   | Kategori | Target label: Low, Mid, High                                       |

---

## Insight Awal

- Dataset bersih tanpa missing values.
- Semua fitur adalah numerik kecuali `RiskLevel` sebagai label klasifikasi.
- Kombinasi variabel seperti tekanan darah, kadar gula, dan usia dapat memberikan indikasi kuat terhadap klasifikasi risiko kehamilan.

---

## Catatan

## Proyek ini bertujuan mendukung sistem kesehatan dalam upaya menurunkan angka kematian maternal dengan mengintegrasikan teknologi kecerdasan buatan ke dalam proses identifikasi risiko secara lebih dini dan efisien.

# Prediksi Risiko Kehamilan dengan Machine Learning

Proyek ini bertujuan untuk memprediksi tingkat risiko kehamilan (`low`, `mid`, `high`) berdasarkan data medis ibu hamil dengan menggunakan berbagai algoritma machine learning, yaitu: **SVM**, **KNN**, **Random Forest**, dan **XGBoost**.

## 📁 Struktur Proyek

```
.
├── predictive_analysis_maternal_risk.ipynb
├── README.md
├── images/
│   ├── confusion_matrix_svm.png
│   ├── confusion_matrix_knn.png
│   ├── confusion_matrix_knn_tuned.png
│   ├── confusion_matrix_rf_tuned.png
│   ├── accuracy_comparison.png
│   └── feature_importance.png

```

## 📊 Dataset

Dataset berisi fitur-fitur medis seperti tekanan darah, usia, dan status gula darah, dengan target label berupa kategori risiko (`0` = rendah, `1` = sedang, `2` = tinggi).

## 🔍 Model yang Digunakan

1. **SVM (Support Vector Machine)**

   - Akurasi: 53.85%
   - [Confusion Matrix](images/confusion_matrix_svm.png)

2. **KNN (K-Nearest Neighbors)**

   - Akurasi awal: 62%
   - Akurasi setelah tuning: 67.03%
   - [Confusion Matrix Sebelum Tuning](images/confusion_matrix_knn.png)
   - [Confusion Matrix Setelah Tuning](images/confusion_matrix_knn_tuned.png)

3. **Random Forest**

   - Akurasi setelah tuning: 71.43%
   - [Confusion Matrix Sebelum Tuning](images/confusion_matrix_rf.png)
   - [Confusion Matrix Setelah Tuning](images/confusion_matrix_rf_tuned.png)

4. **XGBoost**

   - Akurasi: 65.93%
   - [Confusion Matrix Sebelum Tuning](images/confusion_matrix_xgb.png)
   - [Confusion Matrix Setelah Tuning](images/confusion_matrix_xgb_tuned.png)

> Perbandingan akurasi setiap model ditampilkan pada grafik berikut:
> ![Akurasi Per Model](images/akurasi_per_model.png)

## ⚙️ Contoh Kode Pelatihan Model

Berikut adalah contoh pelatihan model Random Forest:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Prediksi
y_pred = rf.predict(X_test)
```

## 🧪 Evaluasi Model

Evaluasi dilakukan menggunakan:

- Confusion Matrix
- Precision, Recall, F1-Score
- Akurasi

Contoh pembuatan confusion matrix:

```python
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## 📝 Kesimpulan

- **Random Forest** menunjukkan performa terbaik setelah tuning dengan akurasi 71.43%.
- **SVM** memberikan performa paling rendah.
- **KNN** mengalami peningkatan performa signifikan setelah dilakukan tuning parameter.

## 📌 Catatan

- Semua visualisasi disimpan di folder `images/`.
- Notebook utama: [`predictive_analysis_maternal_risk.ipynb`](predictive_analysis_maternal_risk.ipynb)

---

## Perbandingan Akurasi Model

Berikut adalah grafik perbandingan akurasi keempat model :

![Perbandingan Akurasi](images/accuracy_comparison.png)

## 🔍 Feature Importance

Berikut adalah grafik feature importance dari model Random Forest:

![Feature Importance](images/feature_importance.png)
