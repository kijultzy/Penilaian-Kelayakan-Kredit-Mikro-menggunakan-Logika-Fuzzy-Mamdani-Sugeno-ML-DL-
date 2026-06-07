"""
SPK Kelayakan Kredit Mikro
Sistem Pendukung Keputusan berbasis Fuzzy Logic (Mamdani & Sugeno)
Integrasi: Regresi Linear (ML) + ANN/MLP (DL)
Kode ini SEPENUHNYA from scratch menggunakan Python & NumPy untuk mesin fuzzy.
"""

# ─── Impor Pustaka ────────────────────────────────────────────────────────────
import time
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Supresi seluruh warning dari Scikit-Learn dan Pandas agar UI bersih
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# ─── Konfigurasi Halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPK Kredit Mikro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ SPK Kelayakan Kredit Mikro (Mamdani vs Sugeno)")
st.subheader("Integrasi Hibrida: Regresi Linear (ML) + ANN (DL) + Logika Fuzzy")
st.caption(
    "⚠️ Pengambilan keputusan akhir **sepenuhnya** ditentukan oleh Logika Fuzzy. "
    "Prediksi ML & DL hanya bersifat informasi pendukung."
)

# ─── Preprocessing & Pelatihan Model (Di-cache agar hanya sekali dijalankan) ──
@st.cache_resource(show_spinner=False)
def muat_dan_latih():
    """
    Memuat train.csv, melakukan data cleaning menyeluruh,
    lalu melatih model Regresi Linear dan ANN (MLP).
    Mengembalikan (model_lr, model_dl, scaler).
    """
    # ── 1. Muat Data ──────────────────────────────────────────────────────────
    df = pd.read_csv("train.csv", low_memory=False)

    kolom_fitur = [
        "Annual_Income",
        "Outstanding_Debt",
        "Interest_Rate",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
    ]
    kolom_target = "Credit_Score"
    kolom_semua = kolom_fitur + [kolom_target]

    df_c = df[kolom_semua].copy()

    # ── 2. Data Cleaning Menyeluruh ───────────────────────────────────────────
    # a) Bersihkan seluruh kolom numerik dari karakter/string asing
    for col in kolom_fitur:
        df_c[col] = (
            df_c[col]
            .astype(str)
            .str.replace(r"[^\d.\-]", "", regex=True)   # pertahankan '-' agar bisa deteksi minus
            .replace("", np.nan)
        )
        df_c[col] = pd.to_numeric(df_c[col], errors="coerce")

    # b) Ubah nilai minus menjadi 0 pada kolom hari keterlambatan
    df_c["Delay_from_due_date"] = df_c["Delay_from_due_date"].clip(lower=0)
    df_c["Num_of_Delayed_Payment"] = df_c["Num_of_Delayed_Payment"].clip(lower=0)

    # c) Hapus baris yang masih mengandung NaN setelah konversi
    df_c.dropna(inplace=True)

    # d) Petakan label Credit_Score ke numerik; buang baris tak dikenali
    peta_skor = {"Poor": 0, "Standard": 1, "Good": 2}
    df_c["Target_Numerik"] = df_c[kolom_target].map(peta_skor)
    df_c.dropna(subset=["Target_Numerik"], inplace=True)
    df_c["Target_Numerik"] = df_c["Target_Numerik"].astype(int)

    # ── 3. Persiapan Fitur & Label ────────────────────────────────────────────
    X = df_c[kolom_fitur].reset_index(drop=True)
    y_clf = df_c["Target_Numerik"].reset_index(drop=True)
    y_reg = y_clf.astype(float)

    X_latih, X_uji, y_latih_r, y_uji_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    _, _, y_latih_c, _ = train_test_split(
        X, y_clf, test_size=0.2, random_state=42
    )

    # ── 4. Model ML: Regresi Linear ───────────────────────────────────────────
    model_lr = LinearRegression()
    model_lr.fit(X_latih, y_latih_r)

    # ── 5. Model DL: ANN (MLP Classifier) ────────────────────────────────────
    scaler = StandardScaler()
    X_latih_s = scaler.fit_transform(X_latih)
    model_dl = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_dl.fit(X_latih_s, y_latih_c)

    return model_lr, model_dl, scaler, kolom_fitur


with st.spinner("⏳ Memuat & melatih model ML dan DL dari data historis... Mohon tunggu."):
    model_lr, model_dl, scaler, KOLOM_FITUR = muat_dan_latih()

st.success("✅ Model berhasil dimuat dan siap digunakan.")

# ─── Fungsi Keanggotaan Fuzzy (from scratch, NumPy) ───────────────────────────

def segitiga(x: float, a: float, b: float, c: float) -> float:
    """Fungsi keanggotaan kurva Segitiga."""
    if x <= a or x >= c:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0


def trapesium(x: float, a: float, b: float, c: float, d: float) -> float:
    """Fungsi keanggotaan kurva Trapesium."""
    if x <= a or x >= d:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a) if b != a else 1.0
    if b <= x <= c:
        return 1.0
    return (d - x) / (d - c) if d != c else 1.0


# ─── Sidebar: Input Data Nasabah ──────────────────────────────────────────────
st.sidebar.header("📥 Input Data Nasabah")
st.sidebar.markdown("---")

pendapatan = st.sidebar.number_input(
    "Pendapatan Tahunan (Annual Income, $)", min_value=0, value=45_000, step=1_000
)
utang = st.sidebar.number_input(
    "Utang Beredar (Outstanding Debt, $)", min_value=0, value=1_500, step=100
)
suku_bunga = st.sidebar.slider(
    "Suku Bunga (Interest Rate, %)", min_value=1, max_value=35, value=12
)
keterlambatan_hari = st.sidebar.slider(
    "Keterlambatan dari Jatuh Tempo (Hari)", min_value=0, max_value=60, value=15
)
jumlah_terlambat = st.sidebar.slider(
    "Jumlah Pembayaran Terlambat (Kali)", min_value=0, max_value=25, value=4
)

st.sidebar.markdown("---")

# ─── Fuzzifikasi ──────────────────────────────────────────────────────────────
# Variabel INPUT → himpunan fuzzy

fz = {
    # Pendapatan Tahunan
    "inc": {
        "Rendah":  trapesium(pendapatan, 0,      0,      30_000,  50_000),
        "Sedang":  segitiga( pendapatan, 30_000, 65_000, 100_000),
        "Tinggi":  trapesium(pendapatan, 80_000, 100_000, 1_500_000, 2_000_000),
    },
    # Utang Beredar
    "debt": {
        "Sedikit": trapesium(utang, 0,      0,      1_000,  2_000),
        "Sedang":  segitiga( utang, 1_000,  2_500,  4_000),
        "Banyak":  trapesium(utang, 3_000,  5_000,  100_000, 100_000),
    },
    # Suku Bunga
    "ir": {
        "Rendah":  trapesium(suku_bunga, 0,  0,  8,   12),
        "Sedang":  segitiga( suku_bunga, 8,  15, 22),
        "Tinggi":  trapesium(suku_bunga, 18, 25, 35,  100),
    },
    # Keterlambatan (Hari)
    "del": {
        "Singkat": trapesium(keterlambatan_hari, 0,  0,  10, 20),
        "Sedang":  segitiga( keterlambatan_hari, 10, 30, 45),
        "Lama":    trapesium(keterlambatan_hari, 35, 60, 200, 200),
    },
    # Jumlah Terlambat
    "num": {
        "Jarang":   trapesium(jumlah_terlambat, 0,  0,  5,   8),
        "Sering":   segitiga( jumlah_terlambat, 5,  12, 18),
        "SgtSering":trapesium(jumlah_terlambat, 15, 25, 100, 100),
    },
}

# ─── Inferensi: 22 Basis Aturan (MIN/AND) ────────────────────────────────────
# Konsekuensi: BAIK, STANDAR, BURUK
# Format: (kekuatan_firing, konsekuensi)
aturan = [
    # ── Aturan → BAIK (Kredit Layak Disetujui) ──────────────────────────────
    (min(fz["inc"]["Tinggi"],  fz["debt"]["Sedikit"], fz["ir"]["Rendah"],  fz["del"]["Singkat"], fz["num"]["Jarang"]),    "baik"),    # R01
    (min(fz["inc"]["Tinggi"],  fz["debt"]["Sedang"],  fz["ir"]["Sedang"],  fz["del"]["Singkat"], fz["num"]["Jarang"]),    "baik"),    # R02
    (min(fz["inc"]["Sedang"],  fz["debt"]["Sedikit"], fz["ir"]["Rendah"],  fz["del"]["Singkat"], fz["num"]["Jarang"]),    "baik"),    # R03
    (min(fz["inc"]["Tinggi"],  fz["debt"]["Sedikit"], fz["ir"]["Tinggi"],  fz["del"]["Sedang"],  fz["num"]["Jarang"]),    "baik"),    # R04
    (min(fz["inc"]["Sedang"],  fz["debt"]["Sedikit"], fz["ir"]["Sedang"],  fz["del"]["Singkat"], fz["num"]["Jarang"]),    "baik"),    # R16 ★ Baru
    (min(fz["inc"]["Tinggi"],  fz["debt"]["Sedang"],  fz["ir"]["Rendah"],  fz["del"]["Singkat"], fz["num"]["Jarang"]),    "baik"),    # R17 ★ Baru
    # ── Aturan → STANDAR (Kredit Perlu Pertimbangan) ─────────────────────────
    (min(fz["inc"]["Sedang"],  fz["debt"]["Sedang"],  fz["ir"]["Sedang"],  fz["del"]["Sedang"],  fz["num"]["Sering"]),    "standar"), # R05
    (min(fz["inc"]["Rendah"],  fz["debt"]["Sedikit"], fz["ir"]["Rendah"],  fz["del"]["Singkat"], fz["num"]["Jarang"]),    "standar"), # R06
    (min(fz["inc"]["Tinggi"],  fz["debt"]["Banyak"],  fz["ir"]["Sedang"],  fz["del"]["Singkat"], fz["num"]["Jarang"]),    "standar"), # R07
    (min(fz["inc"]["Sedang"],  fz["debt"]["Banyak"],  fz["ir"]["Tinggi"],  fz["del"]["Singkat"], fz["num"]["Jarang"]),    "standar"), # R08
    (min(fz["inc"]["Rendah"],  fz["debt"]["Sedang"],  fz["ir"]["Rendah"],  fz["del"]["Singkat"], fz["num"]["Jarang"]),    "standar"), # R09
    (min(fz["inc"]["Tinggi"],  fz["debt"]["Sedang"],  fz["ir"]["Tinggi"],  fz["del"]["Sedang"],  fz["num"]["Sering"]),    "standar"), # R10
    (min(fz["inc"]["Sedang"],  fz["debt"]["Sedang"],  fz["ir"]["Sedang"],  fz["del"]["Singkat"], fz["num"]["Jarang"]),    "standar"), # R18 ★ Baru: kasus umum "titik tengah"
    (min(fz["inc"]["Rendah"],  fz["debt"]["Sedikit"], fz["ir"]["Sedang"],  fz["del"]["Singkat"], fz["num"]["Jarang"]),    "standar"), # R19 ★ Baru
    (min(fz["inc"]["Sedang"],  fz["debt"]["Sedang"],  fz["ir"]["Tinggi"],  fz["del"]["Singkat"], fz["num"]["Jarang"]),    "standar"), # R20 ★ Baru
    # ── Aturan → BURUK (Kredit Ditolak) ──────────────────────────────────────
    (min(fz["inc"]["Rendah"],  fz["debt"]["Banyak"],  fz["ir"]["Tinggi"],  fz["del"]["Lama"],    fz["num"]["SgtSering"]), "buruk"),   # R11
    (min(fz["inc"]["Tinggi"],  fz["debt"]["Banyak"],  fz["ir"]["Tinggi"],  fz["del"]["Lama"],    fz["num"]["SgtSering"]), "buruk"),   # R12
    (min(fz["inc"]["Sedang"],  fz["debt"]["Banyak"],  fz["ir"]["Sedang"],  fz["del"]["Sedang"],  fz["num"]["Sering"]),    "buruk"),   # R13
    (min(fz["inc"]["Rendah"],  fz["debt"]["Sedang"],  fz["ir"]["Sedang"],  fz["del"]["Lama"],    fz["num"]["Sering"]),    "buruk"),   # R14
    (min(fz["inc"]["Sedang"],  fz["debt"]["Sedikit"], fz["ir"]["Tinggi"],  fz["del"]["Lama"],    fz["num"]["SgtSering"]), "buruk"),   # R15
    (min(fz["inc"]["Rendah"],  fz["debt"]["Sedang"],  fz["ir"]["Tinggi"],  fz["del"]["Sedang"],  fz["num"]["Sering"]),    "buruk"),   # R21 ★ Baru
    (min(fz["inc"]["Sedang"],  fz["debt"]["Banyak"],  fz["ir"]["Tinggi"],  fz["del"]["Lama"],    fz["num"]["Sering"]),    "buruk"),   # R22 ★ Baru
]


# Agregasi (MAX per konsekuensi)
u_baik    = max((kuat for kuat, kons in aturan if kons == "baik"),    default=0.0)
u_standar = max((kuat for kuat, kons in aturan if kons == "standar"), default=0.0)
u_buruk   = max((kuat for kuat, kons in aturan if kons == "buruk"),   default=0.0)

# ─── Defuzzifikasi Mamdani (Centroid / Integrasi Numerik) ─────────────────────
def defuzz_mamdani(u_baik: float, u_standar: float, u_buruk: float, resolusi: int = 200) -> float:
    """
    Defuzzifikasi Mamdani menggunakan metode Centroid (integrasi numerik).
    Fungsi keanggotaan output:
      - BAIK    : Trapesium [70, 85, 100, 100]
      - STANDAR : Segitiga  [30, 50, 75]
      - BURUK   : Trapesium [0, 0, 25, 45]
    Jika total firing strength = 0 (tidak ada aturan aktif), kembalikan nilai tengah = 50.
    """
    if u_baik == 0.0 and u_standar == 0.0 and u_buruk == 0.0:
        return 50.0   # Nilai default tengah (mid-range) — penanganan ZeroDivisionError

    x_vals = np.linspace(0, 100, resolusi)
    pembilang = 0.0
    penyebut   = 0.0

    for x in x_vals:
        mu_b  = min(u_baik,    trapesium(x, 70, 85, 100, 100))
        mu_s  = min(u_standar, segitiga(x, 30, 50, 75))
        mu_br = min(u_buruk,   trapesium(x, 0, 0, 25, 45))
        mu_maks = max(mu_b, mu_s, mu_br)
        pembilang += x * mu_maks
        penyebut   += mu_maks

    return pembilang / penyebut if penyebut != 0.0 else 50.0


# ─── Defuzzifikasi Sugeno (Weighted Average) ──────────────────────────────────
def defuzz_sugeno(u_baik: float, u_standar: float, u_buruk: float) -> float:
    """
    Defuzzifikasi Sugeno menggunakan Weighted Average.
    Konstanta output singleton:
      - BAIK    : z_baik    = 90
      - STANDAR : z_standar = 50
      - BURUK   : z_buruk   = 15
    Jika total firing strength = 0, kembalikan nilai default tengah = 50.
    """
    Z_BAIK    = 90.0
    Z_STANDAR = 50.0
    Z_BURUK   = 15.0

    total_kuat = u_baik + u_standar + u_buruk
    if total_kuat == 0.0:
        return 50.0   # Nilai default tengah — penanganan ZeroDivisionError

    return (u_baik * Z_BAIK + u_standar * Z_STANDAR + u_buruk * Z_BURUK) / total_kuat


# ─── Fungsi Vectorized (untuk Chart MF & Evaluasi Batch) ─────────────────────

def _vtr(x, a, b, c, d):
    """Trapesium vectorized — menerima skalar maupun array NumPy."""
    x = np.asarray(x, dtype=float)
    r = np.zeros_like(x)
    if b != a: r[(x > a) & (x < b)]  = (x[(x > a) & (x < b)] - a)  / (b - a)
    r[(x >= b) & (x <= c)] = 1.0
    if d != c: r[(x > c) & (x < d)]  = (d - x[(x > c) & (x < d)]) / (d - c)
    return r


def _vsg(x, a, b, c):
    """Segitiga vectorized — menerima skalar maupun array NumPy."""
    x = np.asarray(x, dtype=float)
    r = np.zeros_like(x)
    if b != a: r[(x > a) & (x <= b)] = (x[(x > a) & (x <= b)] - a) / (b - a)
    if c != b: r[(x > b) & (x < c)]  = (c - x[(x > b) & (x < c)]) / (c - b)
    r[x == b] = 1.0
    return r


def buat_chart_mf(
    pendapatan, utang, suku_bunga, kel_hari, jml_terlambat, fz_dict
):
    """
    Membuat figure matplotlib 2×3 yang menampilkan kurva keanggotaan
    semua 5 variabel input BESERTA posisi nilai input saat ini.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.5))
    fig.patch.set_facecolor('#FFFFFF')

    CLR = ['#EF5350', '#FFA726', '#26A69A']   # Merah, Oranye, Teal

    def _panel(ax, val, title, xlabel, x_range, kurva, fz_keys, xfmt):
        ax.set_facecolor('#F8F9FA')
        x = np.linspace(x_range[0], x_range[1], 600)
        for i, (lbl, fn, prm, clr) in enumerate(kurva):
            y = fn(x, *prm)
            ax.plot(x, y, color=clr, lw=2.0, label=lbl)
            ax.fill_between(x, y, alpha=0.10, color=clr)
        # Garis vertikal nilai saat ini
        ax.axvline(val, color='#222222', lw=1.8, ls='--', zorder=5, label='Nilai Input')
        # Anotasi derajat keanggotaan
        mu_lines = []
        for vk, sk in fz_keys:
            mu = fz_dict[vk].get(sk, 0.0)
            mu_lines.append(f'μ_{sk} = {mu:.3f}')
        ax.text(0.02, 0.97, '\n'.join(mu_lines), transform=ax.transAxes,
                va='top', ha='left', fontsize=7.5, color='#1C1C1C',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.85, edgecolor='#CCCCCC'))
        ax.set_title(title, fontsize=9.5, fontweight='bold', color='#1C1C1C')
        ax.set_xlabel(xlabel, fontsize=8, color='#444444')
        ax.set_ylabel('μ(x)', fontsize=8, color='#444444')
        ax.set_ylim(-0.05, 1.18)
        ax.set_xlim(x_range)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(xfmt))
        ax.legend(fontsize=6.5, loc='upper right', framealpha=0.7, ncol=1)
        ax.grid(True, alpha=0.25, axis='y')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(colors='#555555', labelsize=7)

    _panel(axes[0, 0], pendapatan, 'Annual Income (Pendapatan)', 'USD ($)',
           (0, 200_000),
           [('Rendah (Trap.)', _vtr, [0, 0, 30_000, 50_000], CLR[0]),
            ('Sedang (Segt.)', _vsg, [30_000, 65_000, 100_000], CLR[1]),
            ('Tinggi (Trap.)', _vtr, [80_000, 100_000, 200_000, 200_000], CLR[2])],
           [('inc','Rendah'), ('inc','Sedang'), ('inc','Tinggi')],
           lambda v, _: f'${v/1000:.0f}K')

    _panel(axes[0, 1], utang, 'Outstanding Debt (Utang)', 'USD ($)',
           (0, 8_000),
           [('Sedikit (Trap.)', _vtr, [0, 0, 1_000, 2_000], CLR[2]),
            ('Sedang (Segt.)', _vsg, [1_000, 2_500, 4_000], CLR[1]),
            ('Banyak (Trap.)', _vtr, [3_000, 5_000, 8_000, 8_000], CLR[0])],
           [('debt','Sedikit'), ('debt','Sedang'), ('debt','Banyak')],
           lambda v, _: f'${v/1000:.1f}K')

    _panel(axes[0, 2], suku_bunga, 'Interest Rate (Suku Bunga)', '%',
           (0, 35),
           [('Rendah (Trap.)', _vtr, [0, 0, 8, 12], CLR[2]),
            ('Sedang (Segt.)', _vsg, [8, 15, 22], CLR[1]),
            ('Tinggi (Trap.)', _vtr, [18, 25, 35, 35], CLR[0])],
           [('ir','Rendah'), ('ir','Sedang'), ('ir','Tinggi')],
           lambda v, _: f'{v:.0f}%')

    _panel(axes[1, 0], kel_hari, 'Delay from Due Date (Keterlambatan)', 'Hari',
           (0, 60),
           [('Singkat (Trap.)', _vtr, [0, 0, 10, 20], CLR[2]),
            ('Sedang (Segt.)', _vsg, [10, 30, 45], CLR[1]),
            ('Lama (Trap.)', _vtr, [35, 60, 60, 60], CLR[0])],
           [('del','Singkat'), ('del','Sedang'), ('del','Lama')],
           lambda v, _: f'{v:.0f}h')

    _panel(axes[1, 1], jml_terlambat, 'Num of Delayed Payment (Frekuensi)', 'Kali',
           (0, 25),
           [('Jarang (Trap.)', _vtr, [0, 0, 5, 8], CLR[2]),
            ('Sering (Segt.)', _vsg, [5, 12, 18], CLR[1]),
            ('SgtSering (Trap.)', _vtr, [15, 25, 25, 25], CLR[0])],
           [('num','Jarang'), ('num','Sering'), ('num','SgtSering')],
           lambda v, _: f'{v:.0f}x')

    # Panel kanan bawah: legenda warna
    axes[1, 2].set_visible(False)
    fig.text(0.72, 0.32,
             '── ── Garis putus-putus = Nilai Input Saat Ini\n'
             'μ(x) = Derajat Keanggotaan (0 – 1)\n\n'
             'Kode Warna:\n'
             '  🔴 Rendah / Sedikit / Singkat / Jarang\n'
             '  🟡 Sedang / Sering\n'
             '  🟢 Tinggi / Banyak / Lama / SgtSering',
             fontsize=9, color='#333333', va='center',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#F0F4F8',
                       edgecolor='#BBBBBB', alpha=0.9))

    fig.suptitle('Visualisasi Fungsi Keanggotaan — Posisi Input Saat Ini',
                 fontsize=12, fontweight='bold', color='#1C1C1C', y=1.01)
    plt.tight_layout(pad=1.5)
    return fig


@st.cache_data(show_spinner=False)
def _evaluasi_batch_st(n_sampel: int = 300) -> dict:
    """
    Evaluasi batch vectorized pada sampel acak dari train.csv.
    Di-cache Streamlit — hanya dijalankan sekali saat startup.
    Mengembalikan dict metrik atau None jika file tidak tersedia.
    """
    import os
    if not os.path.exists("train.csv"):
        return None

    kolom = ['Annual_Income', 'Outstanding_Debt', 'Interest_Rate',
             'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Credit_Score']
    df_ev = pd.read_csv("train.csv", low_memory=False)[kolom].copy()
    for col in kolom[:-1]:
        df_ev[col] = pd.to_numeric(
            df_ev[col].astype(str).str.replace(r'[^\d.\-]', '', regex=True).replace('', np.nan),
            errors='coerce'
        )
    df_ev['Delay_from_due_date']    = df_ev['Delay_from_due_date'].clip(lower=0)
    df_ev['Num_of_Delayed_Payment'] = df_ev['Num_of_Delayed_Payment'].clip(lower=0)
    df_ev.dropna(inplace=True)
    df_ev = df_ev[df_ev['Credit_Score'].isin(['Poor', 'Standard', 'Good'])]
    n_sampel = min(n_sampel, len(df_ev))
    df_ev = df_ev.sample(n=n_sampel, random_state=42).reset_index(drop=True)

    inc  = df_ev['Annual_Income'].values;       debt = df_ev['Outstanding_Debt'].values
    ir   = df_ev['Interest_Rate'].values;       dd   = df_ev['Delay_from_due_date'].values
    nd   = df_ev['Num_of_Delayed_Payment'].values
    lgt  = df_ev['Credit_Score'].values

    fz_v = {
        'inc':  {'Rendah': _vtr(inc,0,0,30_000,50_000), 'Sedang': _vsg(inc,30_000,65_000,100_000),
                 'Tinggi': _vtr(inc,80_000,100_000,1_500_000,2_000_000)},
        'debt': {'Sedikit': _vtr(debt,0,0,1_000,2_000), 'Sedang': _vsg(debt,1_000,2_500,4_000),
                 'Banyak': _vtr(debt,3_000,5_000,100_000,100_000)},
        'ir':   {'Rendah': _vtr(ir,0,0,8,12), 'Sedang': _vsg(ir,8,15,22),
                 'Tinggi': _vtr(ir,18,25,35,100)},
        'del':  {'Singkat': _vtr(dd,0,0,10,20), 'Sedang': _vsg(dd,10,30,45),
                 'Lama': _vtr(dd,35,60,200,200)},
        'num':  {'Jarang': _vtr(nd,0,0,5,8), 'Sering': _vsg(nd,5,12,18),
                 'SgtSering': _vtr(nd,15,25,100,100)},
    }
    vm = np.minimum
    def vmx(*a): return np.maximum.reduce(a)

    ub  = vmx(vm(fz_v['inc']['Tinggi'], vm(fz_v['debt']['Sedikit'], vm(fz_v['ir']['Rendah'],  vm(fz_v['del']['Singkat'], fz_v['num']['Jarang'])))),   # R01
              vm(fz_v['inc']['Tinggi'], vm(fz_v['debt']['Sedang'],  vm(fz_v['ir']['Sedang'],  vm(fz_v['del']['Singkat'], fz_v['num']['Jarang'])))),   # R02
              vm(fz_v['inc']['Sedang'], vm(fz_v['debt']['Sedikit'], vm(fz_v['ir']['Rendah'],  vm(fz_v['del']['Singkat'], fz_v['num']['Jarang'])))),   # R03
              vm(fz_v['inc']['Tinggi'], vm(fz_v['debt']['Sedikit'], vm(fz_v['ir']['Tinggi'],  vm(fz_v['del']['Sedang'],  fz_v['num']['Jarang'])))),   # R04
              vm(fz_v['inc']['Sedang'], vm(fz_v['debt']['Sedikit'], vm(fz_v['ir']['Sedang'],  vm(fz_v['del']['Singkat'], fz_v['num']['Jarang'])))),   # R16
              vm(fz_v['inc']['Tinggi'], vm(fz_v['debt']['Sedang'],  vm(fz_v['ir']['Rendah'],  vm(fz_v['del']['Singkat'], fz_v['num']['Jarang']))))    # R17
    )
    us  = vmx(vm(fz_v['inc']['Sedang'], vm(fz_v['debt']['Sedang'],  vm(fz_v['ir']['Sedang'],  vm(fz_v['del']['Sedang'],  fz_v['num']['Sering'])))),   # R05
              vm(fz_v['inc']['Rendah'], vm(fz_v['debt']['Sedikit'], vm(fz_v['ir']['Rendah'],  vm(fz_v['del']['Singkat'], fz_v['num']['Jarang'])))),   # R06
              vm(fz_v['inc']['Tinggi'], vm(fz_v['debt']['Banyak'],  vm(fz_v['ir']['Sedang'],  vm(fz_v['del']['Singkat'], fz_v['num']['Jarang'])))),   # R07
              vm(fz_v['inc']['Sedang'], vm(fz_v['debt']['Banyak'],  vm(fz_v['ir']['Tinggi'],  vm(fz_v['del']['Singkat'], fz_v['num']['Jarang'])))),   # R08
              vm(fz_v['inc']['Rendah'], vm(fz_v['debt']['Sedang'],  vm(fz_v['ir']['Rendah'],  vm(fz_v['del']['Singkat'], fz_v['num']['Jarang'])))),   # R09
              vm(fz_v['inc']['Tinggi'], vm(fz_v['debt']['Sedang'],  vm(fz_v['ir']['Tinggi'],  vm(fz_v['del']['Sedang'],  fz_v['num']['Sering'])))),   # R10
              vm(fz_v['inc']['Sedang'], vm(fz_v['debt']['Sedang'],  vm(fz_v['ir']['Sedang'],  vm(fz_v['del']['Singkat'], fz_v['num']['Jarang'])))),   # R18
              vm(fz_v['inc']['Rendah'], vm(fz_v['debt']['Sedikit'], vm(fz_v['ir']['Sedang'],  vm(fz_v['del']['Singkat'], fz_v['num']['Jarang'])))),   # R19
              vm(fz_v['inc']['Sedang'], vm(fz_v['debt']['Sedang'],  vm(fz_v['ir']['Tinggi'],  vm(fz_v['del']['Singkat'], fz_v['num']['Jarang']))))    # R20
    )
    ubr = vmx(vm(fz_v['inc']['Rendah'], vm(fz_v['debt']['Banyak'],  vm(fz_v['ir']['Tinggi'],  vm(fz_v['del']['Lama'],    fz_v['num']['SgtSering'])))),  # R11
              vm(fz_v['inc']['Tinggi'], vm(fz_v['debt']['Banyak'],  vm(fz_v['ir']['Tinggi'],  vm(fz_v['del']['Lama'],    fz_v['num']['SgtSering'])))),  # R12
              vm(fz_v['inc']['Sedang'], vm(fz_v['debt']['Banyak'],  vm(fz_v['ir']['Sedang'],  vm(fz_v['del']['Sedang'],  fz_v['num']['Sering'])))),     # R13
              vm(fz_v['inc']['Rendah'], vm(fz_v['debt']['Sedang'],  vm(fz_v['ir']['Sedang'],  vm(fz_v['del']['Lama'],    fz_v['num']['Sering'])))),     # R14
              vm(fz_v['inc']['Sedang'], vm(fz_v['debt']['Sedikit'], vm(fz_v['ir']['Tinggi'],  vm(fz_v['del']['Lama'],    fz_v['num']['SgtSering'])))),  # R15
              vm(fz_v['inc']['Rendah'], vm(fz_v['debt']['Sedang'],  vm(fz_v['ir']['Tinggi'],  vm(fz_v['del']['Sedang'],  fz_v['num']['Sering'])))),     # R21
              vm(fz_v['inc']['Sedang'], vm(fz_v['debt']['Banyak'],  vm(fz_v['ir']['Tinggi'],  vm(fz_v['del']['Lama'],    fz_v['num']['Sering']))))      # R22
    )


    tot = ub + us + ubr
    ps  = np.where(tot == 0, 50.0, (ub*90 + us*50 + ubr*15) / np.where(tot == 0, 1, tot))

    x_o = np.linspace(0, 100, 100)
    mfb = _vtr(x_o,70,85,100,100); mfs = _vsg(x_o,30,50,75); mfbr = _vtr(x_o,0,0,25,45)
    pm  = np.zeros(n_sampel)
    for i in range(n_sampel):
        if ub[i] == us[i] == ubr[i] == 0.0: pm[i] = 50.0; continue
        mu  = np.maximum(np.maximum(np.minimum(ub[i],mfb), np.minimum(us[i],mfs)), np.minimum(ubr[i],mfbr))
        den = mu.sum()
        pm[i] = float(np.dot(x_o, mu) / den) if den > 0 else 50.0

    def slbl(s): return 'Good' if s >= 70 else ('Standard' if s >= 40 else 'Poor')
    def lnum(l): return {'Poor':15.,'Standard':50.,'Good':90.}[l]

    pm_l = np.array([slbl(s) for s in pm])
    ps_l = np.array([slbl(s) for s in ps])
    pm_n = np.array([lnum(l) for l in pm_l])
    ps_n = np.array([lnum(l) for l in ps_l])
    gt_n = np.array([lnum(l) for l in lgt])

    KELAS = ['Poor', 'Standard', 'Good']
    cm_m = np.zeros((3, 3), dtype=int); cm_s = np.zeros((3, 3), dtype=int)
    km = {'Poor':0,'Standard':1,'Good':2}
    for g, pm_i, ps_i in zip(lgt, pm_l, ps_l):
        cm_m[km[g]][km[pm_i]] += 1
        cm_s[km[g]][km[ps_i]] += 1

    return {
        'n': n_sampel, 'kelas': KELAS,
        'akurasi_m': float(np.mean(pm_l == lgt)), 'akurasi_s': float(np.mean(ps_l == lgt)),
        'mae_m': float(np.mean(np.abs(pm_n-gt_n))), 'mae_s': float(np.mean(np.abs(ps_n-gt_n))),
        'mse_m': float(np.mean((pm_n-gt_n)**2)),    'mse_s': float(np.mean((ps_n-gt_n)**2)),
        'rmse_m': float(np.sqrt(np.mean((pm_n-gt_n)**2))), 'rmse_s': float(np.sqrt(np.mean((ps_n-gt_n)**2))),
        'cm_m': cm_m, 'cm_s': cm_s,
    }


# ─── Pengukuran Waktu Komputasi ───────────────────────────────────────────────
t0_mamdani = time.perf_counter()
skor_mamdani = defuzz_mamdani(u_baik, u_standar, u_buruk)
t1_mamdani = time.perf_counter()
waktu_mamdani_ms = (t1_mamdani - t0_mamdani) * 1_000

t0_sugeno = time.perf_counter()
skor_sugeno = defuzz_sugeno(u_baik, u_standar, u_buruk)
t1_sugeno = time.perf_counter()
waktu_sugeno_ms = (t1_sugeno - t0_sugeno) * 1_000

# ─── Prediksi Model ML & DL (Informasi Pendukung) ────────────────────────────
fitur_input = pd.DataFrame(
    [[pendapatan, utang, suku_bunga, keterlambatan_hari, jumlah_terlambat]],
    columns=KOLOM_FITUR,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pred_ml = float(model_lr.predict(fitur_input)[0])
    fitur_scaled = scaler.transform(fitur_input)
    prob_dl = model_dl.predict_proba(fitur_scaled)[0]

# Indeks label: 0=Poor, 1=Standard, 2=Good
prob_poor     = prob_dl[0] * 100
prob_standard = prob_dl[1] * 100
prob_good     = prob_dl[2] * 100

# Petakan prediksi regresi ke label teks
if pred_ml >= 1.5:
    label_ml = "Baik"
elif pred_ml >= 0.5:
    label_ml = "Standar"
else:
    label_ml = "Buruk"

# ─── Antarmuka Utama (UI) ─────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Hasil Analisis Sistem Pendukung Keputusan")

# --- Keputusan Utama Fuzzy ---
st.subheader("🎯 Keputusan Utama Berdasarkan Logika Fuzzy")

if skor_sugeno >= 70:
    st.success("🟢 **DISETUJUI — Kredit Layak (BAIK)**")
elif skor_sugeno >= 40:
    st.warning("🟡 **PERTIMBANGAN — Perlu Evaluasi Lebih Lanjut (STANDAR)**")
else:
    st.error("🔴 **DITOLAK — Kredit Tidak Layak (BURUK)**")

st.markdown("---")

# --- Perbandingan Skor Fuzzy ---
st.subheader("📐 Perbandingan Metode Defuzzifikasi Fuzzy")
kol_m, kol_s = st.columns(2)

with kol_m:
    st.metric(
        label="Skor Mamdani (Centroid / Integrasi)",
        value=f"{skor_mamdani:.4f}",
        help="Defuzzifikasi menggunakan metode Centroid dengan integrasi numerik (resolusi 200 titik).",
    )
    st.caption(f"⏱️ Waktu komputasi: **{waktu_mamdani_ms:.4f} ms**")

with kol_s:
    st.metric(
        label="Skor Sugeno (Rata-Rata Terbobot)",
        value=f"{skor_sugeno:.4f}",
        help="Defuzzifikasi menggunakan metode Weighted Average (Sugeno Orde-0).",
    )
    st.caption(f"⏱️ Waktu komputasi: **{waktu_sugeno_ms:.4f} ms**")

# Perbandingan waktu
selisih_ms = abs(waktu_mamdani_ms - waktu_sugeno_ms)
metode_lebih_cepat = "Sugeno" if waktu_sugeno_ms < waktu_mamdani_ms else "Mamdani"
st.info(
    f"🏁 **Komparasi Kinerja:** Metode **{metode_lebih_cepat}** lebih cepat "
    f"dengan selisih **{selisih_ms:.4f} ms**. "
    f"(Mamdani: {waktu_mamdani_ms:.4f} ms | Sugeno: {waktu_sugeno_ms:.4f} ms)"
)

st.markdown("---")

# --- Detail Kekuatan Firing Aturan ---
st.subheader("🔥 Detail Kekuatan Aturan Fuzzy (Firing Strength)")
kol_b, kol_st, kol_br = st.columns(3)
kol_b.metric("Kekuatan BAIK (u_baik)",        f"{u_baik:.4f}")
kol_st.metric("Kekuatan STANDAR (u_standar)", f"{u_standar:.4f}")
kol_br.metric("Kekuatan BURUK (u_buruk)",     f"{u_buruk:.4f}")

with st.expander("📋 Lihat Rincian 22 Basis Aturan Fuzzy"):
    tabel_aturan = []
    nama_aturan = [
        "R01: inc=Tinggi,  debt=Sedikit, ir=Rendah,  del=Singkat, num=Jarang    → BAIK",
        "R02: inc=Tinggi,  debt=Sedang,  ir=Sedang,  del=Singkat, num=Jarang    → BAIK",
        "R03: inc=Sedang,  debt=Sedikit, ir=Rendah,  del=Singkat, num=Jarang    → BAIK",
        "R04: inc=Tinggi,  debt=Sedikit, ir=Tinggi,  del=Sedang,  num=Jarang    → BAIK",
        "R16: inc=Sedang,  debt=Sedikit, ir=Sedang,  del=Singkat, num=Jarang    → BAIK",
        "R17: inc=Tinggi,  debt=Sedang,  ir=Rendah,  del=Singkat, num=Jarang    → BAIK",
        "R05: inc=Sedang,  debt=Sedang,  ir=Sedang,  del=Sedang,  num=Sering    → STANDAR",
        "R06: inc=Rendah,  debt=Sedikit, ir=Rendah,  del=Singkat, num=Jarang    → STANDAR",
        "R07: inc=Tinggi,  debt=Banyak,  ir=Sedang,  del=Singkat, num=Jarang    → STANDAR",
        "R08: inc=Sedang,  debt=Banyak,  ir=Tinggi,  del=Singkat, num=Jarang    → STANDAR",
        "R09: inc=Rendah,  debt=Sedang,  ir=Rendah,  del=Singkat, num=Jarang    → STANDAR",
        "R10: inc=Tinggi,  debt=Sedang,  ir=Tinggi,  del=Sedang,  num=Sering    → STANDAR",
        "R18: inc=Sedang,  debt=Sedang,  ir=Sedang,  del=Singkat, num=Jarang    → STANDAR",
        "R19: inc=Rendah,  debt=Sedikit, ir=Sedang,  del=Singkat, num=Jarang    → STANDAR",
        "R20: inc=Sedang,  debt=Sedang,  ir=Tinggi,  del=Singkat, num=Jarang    → STANDAR",
        "R11: inc=Rendah,  debt=Banyak,  ir=Tinggi,  del=Lama,    num=SgtSering → BURUK",
        "R12: inc=Tinggi,  debt=Banyak,  ir=Tinggi,  del=Lama,    num=SgtSering → BURUK",
        "R13: inc=Sedang,  debt=Banyak,  ir=Sedang,  del=Sedang,  num=Sering    → BURUK",
        "R14: inc=Rendah,  debt=Sedang,  ir=Sedang,  del=Lama,    num=Sering    → BURUK",
        "R15: inc=Sedang,  debt=Sedikit, ir=Tinggi,  del=Lama,    num=SgtSering → BURUK",
        "R21: inc=Rendah,  debt=Sedang,  ir=Tinggi,  del=Sedang,  num=Sering    → BURUK",
        "R22: inc=Sedang,  debt=Banyak,  ir=Tinggi,  del=Lama,    num=Sering    → BURUK",
    ]
    for i, (kuat, kons) in enumerate(aturan):
        tabel_aturan.append({
            "No.": nama_aturan[i].split(":")[0],
            "Deskripsi Aturan": nama_aturan[i].split(": ", 1)[1],
            "Firing Strength": f"{kuat:.4f}",
            "Konsekuensi": kons.upper(),
            "Aktif?": "✅ Ya" if kuat > 0.001 else "❌ Tidak",
        })
    st.dataframe(pd.DataFrame(tabel_aturan), width='stretch')

st.markdown("---")

# --- Fuzzifikasi Detail ---
with st.expander("🔢 Lihat Nilai Fuzzifikasi per Variabel Input"):
    for var_label, var_data in {
        "Pendapatan Tahunan (inc)": fz["inc"],
        "Utang Beredar (debt)": fz["debt"],
        "Suku Bunga (ir)": fz["ir"],
        "Keterlambatan Hari (del)": fz["del"],
        "Jumlah Terlambat (num)": fz["num"],
    }.items():
        st.write(f"**{var_label}**")
        df_fz = pd.DataFrame(
            [(k, f"{v:.4f}") for k, v in var_data.items()],
            columns=["Himpunan Fuzzy", "Nilai Keanggotaan (μ)"],
        )
        st.dataframe(df_fz, width='stretch')

st.markdown("---")

# --- Informasi Pendukung ML & DL ---
st.subheader("ℹ️ Informasi Pendukung: Model ML & DL (Bukan Penentu Keputusan)")
st.caption(
    "Nilai di bawah ini hanya bersifat informatif sebagai cross-reference. "
    "Keputusan kredit **tidak** ditentukan oleh model ini."
)

kol_ml, kol_dl1, kol_dl2, kol_dl3 = st.columns(4)
kol_ml.metric(
    label="🤖 Regresi Linear (ML)",
    value=label_ml,
    help=f"Nilai prediksi mentah: {pred_ml:.4f} (0=Buruk, 1=Standar, 2=Baik)",
)
kol_dl1.metric(
    label="🧠 ANN — Prob. Buruk",
    value=f"{prob_poor:.1f}%",
    help="Probabilitas ANN memprediksi kredit sebagai 'Poor'.",
)
kol_dl2.metric(
    label="🧠 ANN — Prob. Standar",
    value=f"{prob_standard:.1f}%",
    help="Probabilitas ANN memprediksi kredit sebagai 'Standard'.",
)
kol_dl3.metric(
    label="🧠 ANN — Prob. Baik",
    value=f"{prob_good:.1f}%",
    help="Probabilitas ANN memprediksi kredit sebagai 'Good'.",
)

# ─── Visualisasi Fungsi Keanggotaan Real-time ────────────────────────────────
st.markdown("---")
st.subheader("📉 Visualisasi Fungsi Keanggotaan — Posisi Input Saat Ini")
st.caption(
    "Grafik menampilkan kurva Segitiga & Trapesium tiap variabel input. "
    "Garis putus-putus = posisi nilai input Anda saat ini. Tabel μ = derajat keanggotaan."
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _fig_mf = buat_chart_mf(
        pendapatan, utang, suku_bunga, keterlambatan_hari, jumlah_terlambat, fz
    )
    st.pyplot(_fig_mf, width='stretch')
    plt.close(_fig_mf)

# ─── Evaluasi Performa Batch ──────────────────────────────────────────────────
st.markdown("---")
with st.expander(
    "📈 Evaluasi Performa Batch — Mamdani vs Sugeno vs Ground Truth Dataset",
    expanded=False,
):
    st.caption(
        "Evaluasi di-cache dan dijalankan SEKALI saat startup "
        f"menggunakan 300 sampel acak dari train.csv."
    )
    ev = _evaluasi_batch_st(n_sampel=300)

    if ev is None:
        st.warning(
            "⚠️ File `train.csv` tidak ditemukan. "
            "Pastikan Streamlit dijalankan dari direktori yang sama dengan train.csv."
        )
    else:
        st.markdown(f"**Jumlah Sampel Evaluasi:** `{ev['n']}` data")

        # ── Metrik Ringkasan ──────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Akurasi Mamdani",  f"{ev['akurasi_m']*100:.1f}%")
        c2.metric("Akurasi Sugeno",   f"{ev['akurasi_s']*100:.1f}%",
                  delta=f"{(ev['akurasi_s']-ev['akurasi_m'])*100:+.1f}%")
        c3.metric("MAE Mamdani",  f"{ev['mae_m']:.2f} poin")
        c4.metric("MAE Sugeno",   f"{ev['mae_s']:.2f} poin",
                  delta=f"{ev['mae_s']-ev['mae_m']:+.2f}")

        # ── Tabel Metrik Lengkap ───────────────────────────────────────────────
        df_met = pd.DataFrame({
            "Metrik": ["Akurasi (%)", "MAE (poin)", "MSE (poin²)", "RMSE (poin)"],
            "Mamdani": [
                f"{ev['akurasi_m']*100:.2f}%", f"{ev['mae_m']:.4f}",
                f"{ev['mse_m']:.4f}",           f"{ev['rmse_m']:.4f}",
            ],
            "Sugeno": [
                f"{ev['akurasi_s']*100:.2f}%", f"{ev['mae_s']:.4f}",
                f"{ev['mse_s']:.4f}",           f"{ev['rmse_s']:.4f}",
            ],
            "Lebih Baik": [
                "✅ Sugeno"  if ev['akurasi_s'] >= ev['akurasi_m'] else "✅ Mamdani",
                "✅ Sugeno"  if ev['mae_s']     <= ev['mae_m']     else "✅ Mamdani",
                "✅ Sugeno"  if ev['mse_s']     <= ev['mse_m']     else "✅ Mamdani",
                "✅ Sugeno"  if ev['rmse_s']    <= ev['rmse_m']    else "✅ Mamdani",
            ],
        })
        st.dataframe(df_met, width='stretch', hide_index=True)
        st.caption("↑ Akurasi: makin besar makin baik  |  ↓ MAE/MSE/RMSE: makin kecil makin baik")

        # ── Confusion Matrix ──────────────────────────────────────────────────
        st.markdown("**Confusion Matrix Perbandingan:**")
        fig_cm, ax_cm = plt.subplots(1, 2, figsize=(12, 4))
        fig_cm.patch.set_facecolor('white')
        KELAS = ev['kelas']
        for ax_i, cm_d, judul, cmp in [
            (ax_cm[0], ev['cm_m'], f"Mamdani  (Akurasi {ev['akurasi_m']*100:.1f}%)", 'Oranges'),
            (ax_cm[1], ev['cm_s'], f"Sugeno   (Akurasi {ev['akurasi_s']*100:.1f}%)", 'Greens'),
        ]:
            cm_n = cm_d.astype(float) / (cm_d.sum(axis=1, keepdims=True) + 1e-9)
            im = ax_i.imshow(cm_n, cmap=cmp, vmin=0, vmax=1, aspect='auto')
            ax_i.set_xticks(range(3)); ax_i.set_yticks(range(3))
            ax_i.set_xticklabels(KELAS, fontsize=9)
            ax_i.set_yticklabels(KELAS, fontsize=9)
            ax_i.set_xlabel('Prediksi Fuzzy'); ax_i.set_ylabel('Aktual (Ground Truth)')
            ax_i.set_title(judul, fontweight='bold', pad=10)
            for i in range(3):
                for j in range(3):
                    ax_i.text(j, i, f"{cm_d[i,j]}\n({cm_n[i,j]*100:.0f}%)",
                              ha='center', va='center', fontsize=9, fontweight='bold',
                              color='white' if cm_n[i,j] > 0.5 else '#333333')
            plt.colorbar(im, ax=ax_i, fraction=0.046, pad=0.04)
        plt.tight_layout()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.pyplot(fig_cm, width='stretch')
        plt.close(fig_cm)

# ─── Interpretasi Kelebihan & Kekurangan ─────────────────────────────────────
st.markdown("---")
with st.expander(
    "📖 Interpretasi: Kelebihan & Kekurangan Mamdani vs Sugeno",
    expanded=False,
):
    ki1, ki2 = st.columns(2)
    with ki1:
        st.markdown("### 🟢 Fuzzy MAMDANI — Centroid")
        st.markdown("""
**✅ Kelebihan:**
- Output berupa **nilai kontinu** → presisi lebih tinggi
- Defuzzifikasi mudah diinterpretasikan *(luas area kurva)*
- Sangat **sesuai** dengan desain rule linguistik
- Representasi output **alami** menggunakan MF penuh

**❌ Kekurangan:**
- Komputasi **lebih berat** *(integrasi numerik iteratif)*
- Waktu eksekusi **lebih lama** dibanding Sugeno
- Resolusi numerik memengaruhi presisi
- Tidak efisien untuk sistem real-time / embedded
        """)
    with ki2:
        st.markdown("### 🟠 Fuzzy SUGENO — Weighted Average")
        st.markdown("""
**✅ Kelebihan:**
- Komputasi **SANGAT cepat** *(aritmatika sederhana)*
- Efisien untuk **dataset besar** & inferensi real-time
- Tidak perlu fungsi keanggotaan output
- Cocok untuk integrasi **sistem kontrol**

**❌ Kekurangan:**
- Output **diskret** *(nilai singleton tetap)*
- Kurang fleksibel jika output perlu representasi MF
- Konstanta singleton harus ditentukan **manual**
- Kurang "alami" dibanding output Mamdani
        """)
    st.info(
        "💡 **Kesimpulan:** Kedua metode menghasilkan keputusan **konsisten** "
        "(selisih skor biasanya < 5 poin). "
        "Mamdani unggul pada **presisi**, Sugeno unggul pada **kecepatan**. "
        "Dalam sistem hybrid ini keduanya berjalan **paralel** sebagai cross-validation."
    )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🛠️ Dikembangkan sebagai Tugas Sistem Pendukung Keputusan (SPK). "
    "Mesin Fuzzy dibangun 100% from scratch menggunakan Python & NumPy. "
    "Tidak menggunakan pustaka fuzzy eksternal (scikit-fuzzy, dll.)."
)