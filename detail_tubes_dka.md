# 📋 Detail Sistem Pendukung Keputusan (SPK) Kredit Mikro
## Dokumentasi Lengkap & Tutorial Penggunaan

> **Topik:** Penilaian Kelayakan Kredit Mikro menggunakan Logika Fuzzy Mamdani–Sugeno + ML + DL  
> **Dataset:** [Credit Score Classification — Kaggle](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)  
> **Baris Data:** ~100.000 | **Variabel Input:** 5 | **Output:** Credit Score (Poor/Standard/Good)

---

## 🚀 1. Cara Menjalankan Program

### 1.1 Aplikasi Interaktif (Streamlit) — `app.py` ← GUNAKAN INI

```bash
# Dari direktori bab 4:
python -m streamlit run app.py
```

Browser akan otomatis terbuka di **http://localhost:8501**

> **PENTING:** Selalu jalankan dari direktori yang mengandung `train.csv`!  
> Jika tidak, evaluasi batch tidak akan muncul (tapi fuzzy realtime tetap berjalan).

### 1.2 Script Visualisasi untuk Laporan — `visualisasi_spk.py`

```powershell
# Di PowerShell (Windows) — wajib set env var agar emoji tidak error:
$env:PYTHONIOENCODING='utf-8'; python -W ignore visualisasi_spk.py
```

```bash
# Di CMD / Terminal Linux/macOS:
python -W ignore visualisasi_spk.py
```

Menghasilkan 4 file PNG siap laporan:
- `kurva_keanggotaan.png` — Grafik MF Annual Income & Outstanding Debt
- `komparasi_prediksi.png` — Bar chart 4 metode (4 profil nasabah)
- `evaluasi_performa.png` — Confusion Matrix + Akurasi/MAE/RMSE
- `interpretasi_metode.png` — Tabel kelebihan & kekurangan

---

## 📁 2. Struktur File

```
bab 4/
├── app.py                  ← 🔴 UTAMA: Aplikasi interaktif Streamlit
├── visualisasi_spk.py      ← Skrip batch untuk screenshot laporan
├── train.csv               ← Dataset (wajib ada di folder ini)
├── kurva_keanggotaan.png   ← Output gambar (auto-generated)
├── komparasi_prediksi.png  ← Output gambar (auto-generated)
├── evaluasi_performa.png   ← Output gambar (auto-generated)
└── interpretasi_metode.png ← Output gambar (auto-generated)
```

---

## 🖥️ 3. Tutorial Penggunaan app.py (Step-by-Step)

### Langkah 1 — Jalankan Streamlit

```bash
python -m streamlit run app.py
```

### Langkah 2 — Input Data Nasabah (Sidebar Kiri)

Setelah browser terbuka, lihat **panel kiri (sidebar)**. Terdapat 5 input:

| Input | Satuan | Rentang | Default |
|-------|--------|---------|---------|
| **Pendapatan Tahunan** | USD ($) | 0 – 1.500.000 | 75.000 |
| **Utang Beredar** | USD ($) | 0 – 50.000 | 2.500 |
| **Suku Bunga** | % | 0 – 35 | 14,0 |
| **Keterlambatan Jatuh Tempo** | Hari | 0 – 100 | 5 |
| **Frekuensi Keterlambatan** | Kali | 0 – 50 | 2 |

**Cara Input:** Geser slider atau ketik langsung di kotak angka. Output diperbarui **otomatis real-time** setiap ada perubahan.

### Langkah 3 — Baca Output Utama

#### 3a. Kotak Keputusan Utama
Kotak berwarna di bagian atas halaman menampilkan:
- **Skor Mamdani** (0–100) — hasil defuzzifikasi Centroid
- **Skor Sugeno** (0–100) — hasil defuzzifikasi Weighted Average
- **Keputusan** — DISETUJUI 🟢 / PERTIMBANGAN 🟡 / DITOLAK 🔴

| Skor | Keputusan | Kategori |
|------|-----------|----------|
| ≥ 70 | DISETUJUI | BAIK |
| 40–69 | PERTIMBANGAN LEBIH LANJUT | STANDAR |
| < 40 | DITOLAK | BURUK |

#### 3b. Waktu Komputasi
Di bawah skor, tabel membandingkan waktu eksekusi:
- Mamdani: ~10–50 ms (integrasi numerik iteratif)
- Sugeno: ~0.01 ms (aritmatika sederhana)

#### 3c. Tabel Fuzzifikasi
Menampilkan **derajat keanggotaan μ(x)** untuk setiap variabel pada tiap himpunan linguistik. Nilai 0.0–1.0 menunjukkan seberapa "anggota" nilai tersebut pada himpunan yang bersangkutan.

#### 3d. Tabel 15 Basis Aturan (Rule Base)
Menampilkan hasil evaluasi tiap aturan:
- **Kekuatan aturan** = min(μ anteseden1, μ anteseden2, ...) → operator AND
- **Konsekuensi** = BAIK / STANDAR / BURUK
- Baris aktif (kekuatan > 0.01) ditandai berbeda

#### 3e. Informasi Pendukung ML & DL
Di bagian bawah: prediksi **Regresi Linear** dan **ANN/MLP** ditampilkan sebagai *informasi tambahan saja* — bukan penentu keputusan!

### Langkah 4 — Visualisasi Fungsi Keanggotaan Real-time

Scroll ke bawah, lihat grafik **"📉 Visualisasi Fungsi Keanggotaan"**.  
Grafik menampilkan 5 panel (satu per variabel input). Pada tiap panel:
- **Garis berwarna** = kurva MF (Segitiga/Trapesium)
- **Garis putus-putus hitam** = posisi nilai input Anda saat ini
- **Kotak anotasi** = nilai μ untuk tiap himpunan

> Geser slider → grafik langsung berubah!

### Langkah 5 — Evaluasi Performa Batch

Klik expander **"📈 Evaluasi Performa Batch"** untuk melihat:
- Akurasi Mamdani vs Sugeno (vs 300 sampel ground truth)
- Tabel MAE, MSE, RMSE
- Confusion Matrix 3×3 (Poor/Standard/Good)

### Langkah 6 — Interpretasi Metode

Klik expander **"📖 Interpretasi: Kelebihan & Kekurangan"** untuk melihat perbandingan detail Mamdani vs Sugeno.

---

## 🔢 4. Detail Proses Fuzzy Logic (From Scratch)

### 4.1 Variabel Linguistik

Sistem menggunakan **5 variabel input** dan **1 variabel output**:

| No | Variabel | Himpunan Linguistik | Jenis MF |
|----|----------|---------------------|----------|
| 1 | Annual Income | Rendah, Sedang, Tinggi | Trapesium, Segitiga, Trapesium |
| 2 | Outstanding Debt | Sedikit, Sedang, Banyak | Trapesium, Segitiga, Trapesium |
| 3 | Interest Rate | Rendah, Sedang, Tinggi | Trapesium, Segitiga, Trapesium |
| 4 | Delay from Due Date | Singkat, Sedang, Lama | Trapesium, Segitiga, Trapesium |
| 5 | Num of Delayed Payment | Jarang, Sering, SgtSering | Trapesium, Segitiga, Trapesium |
| **OUT** | **Credit Score** | **Baik, Standar, Buruk** | **Trapesium, Segitiga, Trapesium** |

---

### 4.2 Fungsi Keanggotaan (Membership Functions)

Dua jenis fungsi keanggotaan diimplementasikan **100% from scratch** menggunakan Python + NumPy:

#### Fungsi Trapesium μ(x, a, b, c, d)

```
μ(x) = 0           jika x ≤ a atau x ≥ d
      (x-a)/(b-a)   jika a < x < b   ← naik linear
      1             jika b ≤ x ≤ c   ← puncak datar
      (d-x)/(d-c)   jika c < x < d   ← turun linear
```

Parameter kode:
```python
def trapesium(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    if x < b: return (x - a) / (b - a)
    if x <= c: return 1.0
    return (d - x) / (d - c)
```

#### Fungsi Segitiga μ(x, a, b, c)

```
μ(x) = 0           jika x ≤ a atau x ≥ c
      (x-a)/(b-a)   jika a < x ≤ b   ← naik
      (c-x)/(c-b)   jika b < x < c   ← turun
      1             jika x == b      ← puncak
```

Parameter kode:
```python
def segitiga(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if x <= b: return (x - a) / (b - a)
    return (c - x) / (c - b)
```

#### Parameter Tiap Himpunan

**Annual Income (USD)**

| Himpunan | Fungsi | a | b | c | d |
|----------|--------|---|---|---|---|
| Rendah | Trapesium | 0 | 0 | 30.000 | 50.000 |
| Sedang | Segitiga | 30.000 | 65.000 | 100.000 | — |
| Tinggi | Trapesium | 80.000 | 100.000 | 1.500.000 | 2.000.000 |

**Outstanding Debt (USD)**

| Himpunan | Fungsi | a | b | c | d |
|----------|--------|---|---|---|---|
| Sedikit | Trapesium | 0 | 0 | 1.000 | 2.000 |
| Sedang | Segitiga | 1.000 | 2.500 | 4.000 | — |
| Banyak | Trapesium | 3.000 | 5.000 | 100.000 | 100.000 |

**Interest Rate (%)**

| Himpunan | Fungsi | a | b | c | d |
|----------|--------|---|---|---|---|
| Rendah | Trapesium | 0 | 0 | 8 | 12 |
| Sedang | Segitiga | 8 | 15 | 22 | — |
| Tinggi | Trapesium | 18 | 25 | 35 | 100 |

**Delay from Due Date (Hari)**

| Himpunan | Fungsi | a | b | c | d |
|----------|--------|---|---|---|---|
| Singkat | Trapesium | 0 | 0 | 10 | 20 |
| Sedang | Segitiga | 10 | 30 | 45 | — |
| Lama | Trapesium | 35 | 60 | 200 | 200 |

**Num of Delayed Payment (Kali)**

| Himpunan | Fungsi | a | b | c | d |
|----------|--------|---|---|---|---|
| Jarang | Trapesium | 0 | 0 | 5 | 8 |
| Sering | Segitiga | 5 | 12 | 18 | — |
| SgtSering | Trapesium | 15 | 25 | 100 | 100 |

---

### 4.3 Proses Fuzzifikasi

Fuzzifikasi = mengubah nilai input crisp menjadi derajat keanggotaan μ ∈ [0, 1].

**Contoh:** Pendapatan = $75.000
- μ_Rendah = trapesium(75.000, 0, 0, 30.000, 50.000) = **0.0** (di luar range)
- μ_Sedang = segitiga(75.000, 30.000, 65.000, 100.000) = (100.000 - 75.000)/(100.000 - 65.000) = **0.714**
- μ_Tinggi = trapesium(75.000, 80.000, 100.000, ...) = **0.0** (belum naik)

Proses ini dilakukan untuk **semua 5 variabel input**, menghasilkan **15 nilai μ** total (3 himpunan × 5 variabel).

---

### 4.4 Basis Aturan (15 Aturan IF-THEN)

Basis aturan menghubungkan anteseden (kondisi input) dengan konsekuensi (output). Digunakan operator **AND = MIN** (ambil nilai minimum dari semua anteseden).

| # | Jika Pendapatan | Utang | Suku Bunga | Terlambat (Hari) | Terlambat (Kali) | MAKA |
|---|---|---|---|---|---|---|
| R01 | Tinggi | Sedikit | Rendah | Singkat | Jarang | **BAIK** |
| R02 | Tinggi | Sedang | Sedang | Singkat | Jarang | **BAIK** |
| R03 | Sedang | Sedikit | Rendah | Singkat | Jarang | **BAIK** |
| R04 | Tinggi | Sedikit | Tinggi | Sedang | Jarang | **BAIK** |
| R05 | Sedang | Sedang | Sedang | Sedang | Sering | **STANDAR** |
| R06 | Rendah | Sedikit | Rendah | Singkat | Jarang | **STANDAR** |
| R07 | Tinggi | Banyak | Sedang | Singkat | Jarang | **STANDAR** |
| R08 | Sedang | Banyak | Tinggi | Singkat | Jarang | **STANDAR** |
| R09 | Rendah | Sedang | Rendah | Singkat | Jarang | **STANDAR** |
| R10 | Tinggi | Sedang | Tinggi | Sedang | Sering | **STANDAR** |
| R11 | Rendah | Banyak | Tinggi | Lama | SgtSering | **BURUK** |
| R12 | Tinggi | Banyak | Tinggi | Lama | SgtSering | **BURUK** |
| R13 | Sedang | Banyak | Sedang | Sedang | Sering | **BURUK** |
| R14 | Rendah | Sedang | Sedang | Lama | Sering | **BURUK** |
| R15 | Sedang | Sedikit | Tinggi | Lama | SgtSering | **BURUK** |

---

### 4.5 Proses Inferensi (MIN/AND)

Untuk setiap aturan, kekuatan aktivasi dihitung:

```
α_Rn = MIN(μ_anteseden1, μ_anteseden2, μ_anteseden3, μ_anteseden4, μ_anteseden5)
```

Kemudian agregasi per konsekuensi menggunakan **MAX**:

```
μ_BAIK    = MAX(α_R01, α_R02, α_R03, α_R04)
μ_STANDAR = MAX(α_R05, α_R06, α_R07, α_R08, α_R09, α_R10)
μ_BURUK   = MAX(α_R11, α_R12, α_R13, α_R14, α_R15)
```

Implementasi kode:
```python
def inferensi_agregasi(fz):
    aturan = [
        (min(fz['inc']['Tinggi'], fz['debt']['Sedikit'], fz['ir']['Rendah'],
             fz['del']['Singkat'], fz['num']['Jarang']), 'baik'),
        # ... 14 aturan lainnya
    ]
    u_baik    = max((k for k, c in aturan if c == 'baik'),    default=0.0)
    u_standar = max((k for k, c in aturan if c == 'standar'), default=0.0)
    u_buruk   = max((k for k, c in aturan if c == 'buruk'),   default=0.0)
    return u_baik, u_standar, u_buruk
```

---

### 4.6 Defuzzifikasi Mamdani — Metode Centroid

Defuzzifikasi Mamdani menghasilkan nilai crisp dengan menghitung **titik berat (centroid)** dari area gabungan MF output.

**Langkah-langkah:**
1. Buat domain output x ∈ [0, 100] dengan resolusi 500 titik
2. Hitung MF output untuk setiap himpunan:
   - μ_BAIK(x)    = Trapesium(x, 70, 85, 100, 100)
   - μ_STANDAR(x) = Segitiga(x, 30, 50, 75)
   - μ_BURUK(x)   = Trapesium(x, 0, 0, 25, 45)
3. Potong (clip) setiap MF output dengan nilai agregasi:
   - μ'_BAIK(x)    = min(u_baik,    μ_BAIK(x))
   - μ'_STANDAR(x) = min(u_standar, μ_STANDAR(x))
   - μ'_BURUK(x)   = min(u_buruk,   μ_BURUK(x))
4. Gabungkan dengan MAX: μ(x) = max(μ'_BAIK, μ'_STANDAR, μ'_BURUK)
5. Hitung centroid:

```
Skor_Mamdani = Σ[x · μ(x)] / Σ[μ(x)]
```

Implementasi kode:
```python
def defuzz_mamdani(u_baik, u_standar, u_buruk, resolusi=500):
    x = np.linspace(0, 100, resolusi)
    mu_b  = np.minimum(u_baik,    fungsi_trapesium(x, 70, 85, 100, 100))
    mu_s  = np.minimum(u_standar, fungsi_segitiga(x, 30, 50, 75))
    mu_br = np.minimum(u_buruk,   fungsi_trapesium(x, 0, 0, 25, 45))
    mu    = np.maximum(np.maximum(mu_b, mu_s), mu_br)
    den   = mu.sum()
    return float(np.dot(x, mu) / den) if den > 0 else 50.0
```

---

### 4.7 Defuzzifikasi Sugeno — Metode Weighted Average

Defuzzifikasi Sugeno menggunakan **rata-rata berbobot** dari nilai singleton output.

**Nilai Singleton yang Digunakan:**
| Himpunan | Singleton (Z) | Alasan |
|----------|---------------|--------|
| BAIK | Z = 90 | Representatif kelas "Good" |
| STANDAR | Z = 50 | Representatif kelas "Standard" |
| BURUK | Z = 15 | Representatif kelas "Poor" |

**Formula:**

```
Skor_Sugeno = (μ_BAIK × 90 + μ_STANDAR × 50 + μ_BURUK × 15)
              ─────────────────────────────────────────────────
                        (μ_BAIK + μ_STANDAR + μ_BURUK)
```

Implementasi kode:
```python
def defuzz_sugeno(u_baik, u_standar, u_buruk):
    Z_BAIK, Z_STANDAR, Z_BURUK = 90, 50, 15
    total_kuat = u_baik + u_standar + u_buruk
    if total_kuat == 0: return 50.0
    return (u_baik * Z_BAIK + u_standar * Z_STANDAR + u_buruk * Z_BURUK) / total_kuat
```

---

## 🤖 5. Integrasi Machine Learning: Regresi Linear

> ⚠️ **Hanya sebagai informasi pendukung — BUKAN penentu keputusan!**

### Model
- **Algoritma:** `LinearRegression` dari scikit-learn
- **Target (y):** Skor numerik dari Credit_Score (Poor=15, Standard=50, Good=90)
- **Fitur (X):** 5 variabel input yang sudah dibersihkan

### Cara Kerja

```
Ŷ = β₀ + β₁·Annual_Income + β₂·Outstanding_Debt + β₃·Interest_Rate
        + β₄·Delay_days + β₅·Num_Delayed
```

Koefisien β ditemukan dengan minimasi **Sum of Squared Errors (SSE)**:

```
β* = argmin Σ (yᵢ - Xᵢβ)²
```

### Implementasi dalam kode

```python
@st.cache_resource
def muat_dan_latih():
    # ... preprocessing
    from sklearn.linear_model import LinearRegression
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    # ...
```

### Output di UI
- Menampilkan **prediksi skor numerik** dan **kategori** (Poor/Standard/Good)
- Menampilkan **MAE** model di data train

---

## 🧠 6. Integrasi Deep Learning: ANN/MLP (MLPClassifier)

> ⚠️ **Hanya sebagai informasi pendukung — BUKAN penentu keputusan!**

### Arsitektur
- **Algoritma:** `MLPClassifier` dari scikit-learn
- **Jenis:** Multi-Layer Perceptron (Artificial Neural Network)
- **Hidden Layers:** 2 layer tersembunyi (100 neuron, 50 neuron)
- **Aktivasi:** ReLU (Rectified Linear Unit)
- **Optimizer:** Adam
- **Output:** Probabilitas 3 kelas (Poor / Standard / Good)

### Struktur Jaringan

```
Input Layer   Hidden Layer 1   Hidden Layer 2   Output Layer
(5 neuron) → (100 neuron) → (50 neuron) → (3 neuron: Poor/Standard/Good)
   ReLU           ReLU           Softmax
```

### Implementasi dalam kode

```python
from sklearn.neural_network import MLPClassifier
model_ann = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42,
    early_stopping=True,
)
model_ann.fit(X_train, y_train_str)
```

### Output di UI
- **Label prediksi** (Poor/Standard/Good)
- **Probabilitas** tiap kelas (Poor%, Standard%, Good%)
- Metrik **Akurasi** model di data train
- **Catatan eksplisit** bahwa ini bukan penentu keputusan

---

## 📊 7. Evaluasi Performa

### Metodologi

- **Ground Truth:** Label `Credit_Score` dari dataset asli (Poor/Standard/Good)
- **Konversi ke numerik:** Poor=15, Standard=50, Good=90
- **Sampel:** 300 data acak (seed=42, bisa direproduksi)
- **Prediksi:** Skor Mamdani dan Sugeno dikonversi ke label: ≥70=Good, 40-69=Standard, <40=Poor

### Metrik yang Digunakan

| Metrik | Formula | Interpretasi |
|--------|---------|--------------|
| **Akurasi** | (Prediksi benar / Total) × 100% | Makin besar makin baik |
| **MAE** | Mean\|prediksi - aktual\| | Rata-rata selisih absolut (poin) |
| **MSE** | Mean(prediksi - aktual)² | Penalti lebih besar untuk selisih besar |
| **RMSE** | √MSE | Satuan sama dengan MAE, tapi penalti besar |

### Confusion Matrix

Matriks 3×3 menunjukkan distribusi prediksi per kelas:

```
              Prediksi
              Poor  Standard  Good
Aktual Poor  [  ]  [   ]    [  ]
    Standard  [  ]  [   ]    [  ]
       Good   [  ]  [   ]    [  ]
```

- **Diagonal (kiri atas → kanan bawah):** Prediksi benar
- **Di luar diagonal:** Kesalahan klasifikasi

---

## 📖 8. Interpretasi Kelebihan & Kekurangan

### Fuzzy MAMDANI — Metode Centroid

| Aspek | Nilai | Penjelasan |
|-------|-------|-----------|
| Defuzzifikasi | Integrasi Numerik | Centroid dari area MF output |
| Kecepatan | ⚠️ Lebih lambat | ~10–50 ms per inferensi |
| Presisi Output | ✅ Kontinu | Nilai 0–100 presisi tinggi |
| Interpretasi | ✅ Intuitif | Area di bawah kurva = keputusan |
| Sesuai Rule | ✅ Sangat sesuai | MF output alami dan ekspresif |
| Cocok untuk | Keputusan kritis | Saat presisi > kecepatan |

### Fuzzy SUGENO — Metode Weighted Average

| Aspek | Nilai | Penjelasan |
|-------|-------|-----------|
| Defuzzifikasi | Weighted Average | Rata-rata berbobot singleton |
| Kecepatan | ✅ Sangat cepat | < 0.1 ms per inferensi |
| Presisi Output | ⚠️ Diskret | Terbatas pada nilai singleton |
| Interpretasi | ✅ Sederhana | Formula aritmatika langsung |
| Sesuai Rule | ✅ Sesuai | Efisien tanpa MF output |
| Cocok untuk | Real-time / embedded | Saat kecepatan > presisi |

### Kesimpulan Perbandingan

Untuk SPK Kredit Mikro ini, **kedua metode menghasilkan keputusan yang konsisten** — selisih skor biasanya di bawah 5 poin. Dalam sistem hybrid ini:
- Keduanya berjalan **paralel** sebagai *cross-validation*
- Jika skor keduanya sejalan → keputusan **lebih dapat dipercaya**
- Jika skor berbeda jauh → menjadi sinyal untuk **analisis lebih mendalam**

---

## 🏆 9. Pemenuhan Spesifikasi Tugas

| Kriteria | Implementasi | Status |
|----------|-------------|--------|
| Dataset nyata ≥ 5.000 baris | ~100.000 baris dari Kaggle | ✅ |
| ≥ 5 variabel input + 1 output | 5 input + Credit Score | ✅ |
| Sumber dataset dicantumkan | kaggle.com/datasets/parisrohan/credit-score-classification | ✅ |
| Variabel linguistik | 5 var × 3 himpunan linguistik | ✅ |
| Fungsi keanggotaan | Segitiga + Trapesium | ✅ |
| Rule base ≥ 15 | Tepat 15 aturan | ✅ |
| Fuzzy Mamdani (Centroid) | Implementasi from scratch | ✅ |
| Fuzzy Sugeno (Weighted Avg) | Implementasi from scratch | ✅ |
| Tanpa library fuzzy | 100% Python + NumPy | ✅ |
| Perbandingan Mamdani vs Sugeno | Skor, waktu, akurasi, MAE, MSE, RMSE | ✅ |
| Evaluasi (Akurasi/MAE/MSE) | Batch evaluation 300 sampel | ✅ |
| Interpretasi kelebihan/kekurangan | Panel dalam app + dokumentasi | ✅ |
| Web App (Bonus +5) | Streamlit interactive app | ✅ |
| Regresi Linear (Bonus +10) | scikit-learn LinearRegression | ✅ |
| ANN/Deep Learning (Bonus +20) | scikit-learn MLPClassifier | ✅ |

---

## 🔧 10. Data Preprocessing

### Masalah pada Dataset

Dataset raw `train.csv` memiliki beberapa masalah:

1. **Karakter non-numerik** — kolom angka mengandung simbol seperti `$`, `_`, dll.
2. **Nilai NaN** — data hilang atau tidak terisi
3. **Nilai negatif** — keterlambatan hari/jumlah tidak boleh negatif

### Langkah Pembersihan

```python
# Langkah 1: Hilangkan karakter non-numerik
df[col] = df[col].astype(str)\
    .str.replace(r'[^\d.\-]', '', regex=True)\
    .replace('', np.nan)

# Langkah 2: Konversi ke numerik
df[col] = pd.to_numeric(df[col], errors='coerce')

# Langkah 3: Clip nilai negatif ke 0
df['Delay_from_due_date']    = df['Delay_from_due_date'].clip(lower=0)
df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].clip(lower=0)

# Langkah 4: Hapus baris dengan NaN
df.dropna(inplace=True)
```

### Normalisasi untuk ML/DL

Untuk model ML dan DL, fitur dinormalisasi menggunakan **StandardScaler** (z-score normalization):

```
x_normal = (x - mean) / std
```

Agar model tidak bias terhadap fitur dengan skala besar (Annual_Income >> Interest_Rate).

---

## 💡 11. Pertanyaan Umum (FAQ)

**Q: Kenapa skor Mamdani dan Sugeno berbeda?**  
A: Mamdani menggunakan centroid dari area MF penuh (kontinu), Sugeno menggunakan rata-rata berbobot dari nilai singleton (diskret). Untuk kasus "borderline" perbedaan bisa 5–15 poin, tapi keputusan (label) biasanya tetap sama.

**Q: Kenapa akurasi ~55%? Apakah itu baik?**  
A: Dataset Credit Score memiliki distribusi label yang tidak merata (dominan Standard). Sistem fuzzy mengutamakan **interpretabilitas** dan **transparansi** (tiap aturan dapat dibaca manusia), bukan hanya akurasi. Dibanding model black-box ML biasa, trade-off ini dianggap wajar untuk SPK.

**Q: Apakah model ML/DL menggantikan fuzzy?**  
A: **TIDAK.** Sesuai rubrik, ML dan DL **hanya sebagai informasi pendukung**. Keputusan final 100% dari logika fuzzy. Label tersebut ada di UI dengan teks eksplisit.

**Q: Bagaimana cara menjalankan visualisasi untuk laporan?**  
A: Jalankan `python -W ignore visualisasi_spk.py` dari direktori `bab 4/`. File PNG akan otomatis tersimpan di folder yang sama.

---

*Dokumentasi ini dibuat untuk keperluan tugas akademik Sistem Pendukung Keputusan (SPK).  
Semua kode fuzzy logic diimplementasikan from scratch tanpa menggunakan library fuzzy eksternal.*
