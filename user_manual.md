# 📘 User Manual — SPK Kredit Mikro Hybrid
### Sistem Pendukung Keputusan Berbasis Logika Fuzzy Mamdani–Sugeno + ML + DL

---

## 🔑 Pertanyaan yang Sering Ditanyakan (Before You Start)

### ❓ Pertanyaan 1: Kapan Regresi Linear dikatakan "Baik"?

Label yang muncul di **Regresi Linear (ML)** adalah **prediksi model**, bukan indikator kualitas modelnya.

| Label yang Tampil | Artinya |
|---|---|
| **Baik** | Model LR memprediksi skor ≥ 70 untuk nasabah ini |
| **Standar** | Model LR memprediksi skor 40–69 |
| **Buruk** | Model LR memprediksi skor < 40 |

**Untuk mendapat label "Baik" dari LR**, nasabah perlu profil yang jelas menguntungkan, misalnya:
- Pendapatan > $100.000
- Utang < $1.000
- Suku Bunga < 8%
- Keterlambatan sangat sedikit

> ⚠️ **Penting:** Label "Standar" di LR tidak berarti regresinya buruk — itu hanya prediksinya.  
> Model LR dan ANN **hanya sebagai informasi pendukung**, bukan penentu keputusan kredit!

---

## 🚀 Cara Menjalankan

### Langkah 1 — Buka terminal di folder `bab 4`

```powershell
python -m streamlit run app.py
```

Browser otomatis terbuka di **http://localhost:8501**

> Jika tidak terbuka, buka manual di browser dan ketik: `http://localhost:8501`

---

## 🖥️ Panduan Antarmuka Lengkap

### AREA 1: Sidebar Kiri — Input Data Nasabah

Semua input ada di **panel kiri**. Geser slider atau ketik langsung nilainya.

```
╔══════════════════════════════╗
║  👤 Data Nasabah             ║
║  ─────────────────────────── ║
║  Pendapatan Tahunan  [$]     ║
║  [════════●══════════]       ║
║  Utang Beredar       [$]     ║
║  [══════●════════════]       ║
║  Suku Bunga          [%]     ║
║  [═══════●═══════════]       ║
║  Keterlambatan       [Hari]  ║
║  [═●════════════════]        ║
║  Frekuensi Terlambat [Kali]  ║
║  [═●════════════════]        ║
╚══════════════════════════════╝
```

| Input | Range | Catatan |
|---|---|---|
| **Pendapatan Tahunan** | $0 – $1.500.000 | Gunakan angka tahunan, bukan bulanan |
| **Utang Beredar** | $0 – $50.000 | Total utang aktif saat ini |
| **Suku Bunga** | 0% – 35% | Suku bunga pinjaman yang diajukan |
| **Keterlambatan Jatuh Tempo** | 0 – 100 hari | Rata-rata hari keterlambatan |
| **Frekuensi Keterlambatan** | 0 – 50 kali | Berapa kali pernah terlambat |

> **Tip:** Output diperbarui **otomatis real-time** setiap kali slider digeser!

---

### AREA 2: Kotak Keputusan Utama

Setelah mengisi input, lihat blok berwarna di bagian atas halaman:

```
╔═══════════════════════════════════════════╗
║  🟡 PERTIMBANGAN LEBIH LANJUT            ║
║  Skor Fuzzy: 50.71 (Mamdani)             ║
║  Kategori  : STANDAR                      ║
╚═══════════════════════════════════════════╝
```

**Tabel Interpretasi Skor:**

| Warna | Skor Fuzzy | Label Keputusan | Kategori |
|---|---|---|---|
| 🟢 Hijau | 70 – 100 | **DISETUJUI** | BAIK |
| 🟡 Kuning | 40 – 69 | **PERTIMBANGAN LEBIH LANJUT** | STANDAR |
| 🔴 Merah | 0 – 39 | **DITOLAK** | BURUK |

---

### AREA 3: Perbandingan Skor Fuzzy Mamdani vs Sugeno

```
┌──────────────────────┬──────────────────────┐
│ Skor Mamdani         │ Skor Sugeno          │
│ (Centroid/Integrasi) │ (Rata-rata Terbobot) │
│                      │                      │
│     50.7143          │     50.0000          │
│                      │                      │
│ ⏱ 12.3456 ms        │ ⏱  0.0025 ms        │
└──────────────────────┴──────────────────────┘
```

**Cara membaca:**
- **Mamdani** lebih presisi tapi lebih lambat — menggunakan integrasi numerik
- **Sugeno** sangat cepat — menggunakan rumus rata-rata berbobot
- Jika kedua skor **sejalan** (selisih < 10) → keputusan lebih dapat dipercaya
- Jika kedua skor **berbeda jauh** → perlu analisis manual lebih lanjut

---

### AREA 4: 🔥 Detail Kekuatan Aturan Fuzzy

```
┌─────────────────┬─────────────────┬─────────────────┐
│ Kekuatan BAIK   │ Kekuatan STANDAR│ Kekuatan BURUK  │
│   u_baik        │   u_standar     │   u_buruk       │
│   0.0000        │   0.7143        │   0.0000        │
└─────────────────┴─────────────────┴─────────────────┘
```

Ini menampilkan **hasil agregasi MAX** dari semua aturan fuzzy:
- Nilai 0.0 = tidak ada aturan yang mendukung kategori ini
- Nilai 1.0 = minimal satu aturan aktif dengan kekuatan penuh

**Klik expander "Lihat Rincian 22 Basis Aturan Fuzzy"** untuk melihat:

| Kolom | Penjelasan |
|---|---|
| **No.** | Nomor aturan (R01–R22) |
| **Deskripsi Aturan** | Kondisi anteseden dan konsekuensi |
| **Firing Strength** | Nilai MIN dari semua anteseden (0.0 – 1.0) |
| **Konsekuensi** | Hasil: BAIK / STANDAR / BURUK |
| **Aktif?** | ✅ Ya jika firing strength > 0.001 |

**Tip untuk screenshot laporan:** Cari baris dengan "✅ Ya" — itulah aturan yang berkontribusi pada keputusan saat ini.

---

### AREA 5: 🔢 Nilai Fuzzifikasi per Variabel

Klik expander **"Lihat Nilai Fuzzifikasi per Variabel Input"** untuk melihat derajat keanggotaan (μ) setiap variabel:

| Variabel | Himpunan | μ(x) |
|---|---|---|
| Pendapatan | Rendah | 0.0000 |
| Pendapatan | **Sedang** | **0.7143** |
| Pendapatan | Tinggi | 0.0000 |
| Utang | Sedikit | 0.0000 |
| Utang | **Sedang** | **1.0000** |
| ... | ... | ... |

**Cara membaca:**
- Nilai mendekati **1.0** = "sangat anggota" himpunan tersebut
- Nilai mendekati **0.0** = "tidak anggota" himpunan tersebut
- Nilai di antaranya = "sebagian anggota" (fuzzy!)

---

### AREA 6: ℹ️ Informasi Pendukung ML & DL

```
┌──────────────┬─────────────┬──────────────┬─────────────┐
│ Regresi LR   │ ANN — Buruk │ ANN — Standar│ ANN — Baik  │
│  Standar     │    6.1%     │    37.8%     │    56.1%    │
└──────────────┴─────────────┴──────────────┴─────────────┘
```

- **Regresi Linear**: Prediksi label berdasarkan persamaan linear
- **ANN Prob. Buruk**: Probabilitas ANN memprediksi "Poor"
- **ANN Prob. Standar**: Probabilitas ANN memprediksi "Standard"
- **ANN Prob. Baik**: Probabilitas ANN memprediksi "Good"

> ⚠️ **Ingat: ini BUKAN penentu keputusan!** Hanya pembanding informatif.

---

### AREA 7: 📉 Visualisasi Fungsi Keanggotaan (Real-time)

Scroll ke bawah untuk melihat 5 panel grafik:

```
┌──────────────┬──────────────┬──────────────┐
│ Annual Income│ Outst. Debt  │ Interest Rate│
│  [grafik MF] │  [grafik MF] │  [grafik MF] │
├──────────────┴──────────────┤              │
│ Delay Days   │ Num Delayed  │              │
│  [grafik MF] │  [grafik MF] │  Legenda     │
└──────────────┴──────────────┘              │
```

**Cara membaca setiap panel:**
- **Garis berwarna** = kurva fungsi keanggotaan (Segitiga/Trapesium)
- **Garis putus-putus hitam** = posisi nilai input Anda saat ini
- **Kotak teks** = nilai μ(x) untuk tiap himpunan
- **Warna kurva**: 🔴 Rendah/Sedikit/Singkat/Jarang | 🟡 Sedang/Sering | 🟢 Tinggi/Banyak/Lama/SgtSering

> **Ketika menggeser slider** → grafik langsung update! Ini fitur real-time utama.

---

### AREA 8: 📈 Evaluasi Performa Batch *(klik expander)*

Menampilkan perbandingan akurasi fuzzy vs ground truth dataset (300 sampel):

```
┌────────────────┬──────────┬──────────┐
│ Metrik         │ Mamdani  │ Sugeno   │
├────────────────┼──────────┼──────────┤
│ Akurasi        │  XX.X%   │  XX.X%   │
│ MAE            │  XX.XX   │  XX.XX   │
│ MSE            │  XXXX    │  XXXX    │
│ RMSE           │  XX.XX   │  XX.XX   │
└────────────────┴──────────┴──────────┘
```

Plus **Confusion Matrix 3×3** (Poor/Standard/Good).

---

### AREA 9: 📖 Interpretasi Metode *(klik expander)*

Perbandingan detail kelebihan dan kekurangan Mamdani vs Sugeno untuk keperluan analisis dan laporan.

---

## 🎯 Skenario Pengujian — Coba Ini!

### Skenario A: Nasabah Ideal (Harus DISETUJUI 🟢)
```
Pendapatan     : $120.000
Utang          : $500
Suku Bunga     : 5%
Keterlambatan  : 2 hari
Frekuensi      : 0 kali
```
**Hasil yang diharapkan:** Skor > 70, keputusan DISETUJUI

---

### Skenario B: Nasabah Menengah (Harus PERTIMBANGAN 🟡)
```
Pendapatan     : $75.000
Utang          : $2.500
Suku Bunga     : 14%
Keterlambatan  : 5 hari
Frekuensi      : 2 kali
```
**Hasil yang diharapkan:** Skor 40–69, keputusan PERTIMBANGAN

---

### Skenario C: Nasabah Risiko Tinggi (Harus DITOLAK 🔴)
```
Pendapatan     : $18.000
Utang          : $7.500
Suku Bunga     : 28%
Keterlambatan  : 50 hari
Frekuensi      : 20 kali
```
**Hasil yang diharapkan:** Skor < 40, keputusan DITOLAK

---

## 📸 Tips untuk Screenshot Laporan

### 1. Screenshot Keputusan Utama
- Set ke **Skenario A** → Screenshot kotak hijau DISETUJUI
- Set ke **Skenario C** → Screenshot kotak merah DITOLAK

### 2. Screenshot Tabel Aturan Aktif
- Set Skenario A → Klik expander 22 Aturan → Screenshot (cari baris ✅)
- Tampilkan bahwa ada rule yang aktif dengan firing strength > 0

### 3. Screenshot Fungsi Keanggotaan
- Scroll ke grafik → Posisi garis putus-putus terlihat pada kurva
- Pastikan nilai μ terlihat di kotak kecil dalam grafik

### 4. Screenshot Evaluasi Batch
- Klik expander "Evaluasi Performa Batch"
- Screenshot confusion matrix + tabel metrik

### 5. Screenshot Perbandingan ML vs Fuzzy
- Gunakan Skenario B agar ada perbedaan menarik antar metode

---

## 🔧 Troubleshooting

| Masalah | Penyebab | Solusi |
|---|---|---|
| Skor tetap 50.0000 | Kombinasi input di "dead zone" | Pindahkan slider ke nilai ekstrem dulu, lalu kembali |
| Evaluasi Batch tidak muncul | `train.csv` tidak ditemukan | Pastikan `train.csv` ada di folder `bab 4/` |
| Grafik MF kosong | matplotlib error | Refresh browser (F5) |
| Model ML/DL lama loading | Pertama kali latih model | Tunggu ~30 detik, progress bar akan muncul |
| Error saat buka | Port 8501 sudah dipakai | Jalankan: `python -m streamlit run app.py --server.port 8502` |

---

## 🏆 Peta Rubrik vs Fitur

| Kriteria Rubrik | Fitur di app.py | Lokasi |
|---|---|---|
| Fuzzifikasi (Segitiga+Trapesium) | Tabel Fuzzifikasi | Area 5 |
| 22 Basis Aturan (MIN/AND) | Tabel 22 Aturan | Area 4 → Expander |
| Defuzzifikasi Mamdani (Centroid) | Skor Mamdani | Area 3 |
| Defuzzifikasi Sugeno (Weighted Avg) | Skor Sugeno | Area 3 |
| Perbandingan Mamdani vs Sugeno | Waktu + Akurasi | Area 3 + Area 8 |
| Evaluasi (Akurasi/MAE/MSE/RMSE) | Evaluasi Batch | Area 8 |
| Interpretasi kelebihan/kekurangan | Expander Interpretasi | Area 9 |
| Regresi Linear (Bonus +10) | Info Pendukung ML | Area 6 |
| ANN/Deep Learning (Bonus +20) | Info Pendukung DL | Area 6 |
| Web App (Bonus +5) | Seluruh Streamlit App | — |

---

*User Manual ini dibuat untuk keperluan tugas akademik SPK — Sistem Pendukung Keputusan Kredit Mikro.*
