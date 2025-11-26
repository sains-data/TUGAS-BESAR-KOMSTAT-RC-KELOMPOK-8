# Simulasi Statistik Penumpang Angkutan Udara ✈️📊

![Bahasa R](https://img.shields.io/badge/Language-R-blue.svg)
![Lisensi](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Selesai-success.svg)

## 📌 Ringkasan Proyek
Repositori ini berisi **Tugas Besar** mata kuliah **Komputasi Statistik**. Penelitian ini berfokus pada simulasi data runtun waktu (*time series*) penumpang angkutan udara bulanan di **Sumatera Selatan (2015–2025)** menggunakan teknik simulasi Monte Carlo.

Kami mengimplementasikan dan membandingkan tiga algoritma Pembangkit Bilangan Acak (*Random Number Generation* - RNG) untuk memodelkan perilaku stokastik data penumpang, yang teridentifikasi mengikuti **Distribusi Normal** berdasarkan uji Shapiro-Wilk.

### 🎯 Tujuan
1.  **Menganalisis** distribusi dan parameter data penumpang angkutan udara dari BPS Sumatera Selatan.
2.  **Mengimplementasikan** tiga metode RNG dalam bahasa pemrograman R:
    * Metode Transformasi Invers (*Inverse-Transform*)
    * Metode Penerimaan-Penolakan (*Acceptance-Rejection*) dengan Proposal Cauchy
    * Metode Transformasi Box-Muller (*Box-Muller Transformation*)
3.  **Mengevaluasi** kinerja setiap metode berdasarkan akurasi statistik (Bias) dan Uji Kesesuaian Distribusi (*Goodness-of-Fit* Kolmogorov-Smirnov).

---

## 📂 Dataset
Data yang digunakan adalah data sekunder yang diperoleh dari **Badan Pusat Statistik (BPS) Provinsi Sumatera Selatan**.
* **Periode:** Januari 2015 – 2025 (Data Bulanan).
* **Variabel:**
    * $X_1, X_2$: Tahun & Bulan
    * $X_3$: Total Keberangkatan
    * $X_4$: Total Kedatangan
    * $X_5$: Total Penumpang (Variabel Utama).
* **Karakteristik:** Fluktuasi tinggi, pola musiman, dan adanya *structural break* akibat pandemi COVID-19.

---

## 🛠️ Metodologi & Algoritma

### 1. Metode Transformasi Invers (*Inverse-Transform Method*)
Membangkitkan variabel acak dengan menghitung invers dari Fungsi Distribusi Kumulatif (CDF) distribusi target.
* *Fungsi kunci:* `qnorm(u, mean, sd)`.

### 2. Metode Penerimaan-Penolakan (*Acceptance-Rejection Method*)
Menggunakan **distribusi Cauchy** sebagai fungsi proposal karena memiliki ekor yang lebih berat (*heavier tails*) dibandingkan distribusi Normal, sehingga dapat "menyelimuti" distribusi target dengan baik.
* *Kondisi penerimaan:* $U \leq \frac{f(x)}{c \cdot g(x)}$.

### 3. Transformasi Box-Muller (Metode Terbaik 🏆)
Metode transformasi geometri yang membangkitkan sepasang variabel normal standar yang saling bebas (*independent*) dari dua variabel acak seragam.
* *Rumus:*
    $$Z_1 = \sqrt{-2 \ln U_1} \cos(2\pi U_2)$$
    $$Z_2 = \sqrt{-2 \ln U_1} \sin(2\pi U_2)$$

---

## 📊 Hasil & Kesimpulan

Berdasarkan simulasi dengan $N=10.000$ sampel:

| Metode | p-value KS-Test | Bias Mean | Bias SD | Keputusan |
| :--- | :---: | :---: | :---: | :--- |
| **Inverse-Transform** | 0.4973 | Sedang | Tinggi (~900) | Valid |
| **Acceptance-Rejection**| **0.7121** | Tinggi | Sedang | Bentuk Distribusi Terbaik |
| **Box-Muller** | 0.4290 | **Terendah** | **Terendah (38.98)** | **Paling Akurat & Robust** |

**Kesimpulan:**
**Transformasi Box-Muller** direkomendasikan sebagai metode terbaik untuk dataset ini karena presisinya yang unggul dalam mereplikasi varians (Standar Deviasi) dari data riil.

---

## 💻 Teknologi yang Digunakan
* **Bahasa:** R (versi 4.x)
* **IDE:** RStudio
* **Pustaka (Libraries):** `ggplot2`, `forecast`, `fitdistrplus`, `stats`
