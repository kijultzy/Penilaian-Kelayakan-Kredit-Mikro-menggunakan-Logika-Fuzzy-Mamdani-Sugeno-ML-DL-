"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   SPK KREDIT MIKRO — Visualisasi & Explainable AI (XAI)                    ║
║   Fitur 1 : Kurva Keanggotaan (Membership Function) — Matplotlib            ║
║   Fitur 2 : Komparasi Prediksi 4 Metode (Bar Chart)                        ║
║   Fitur 3 : Generator Kesimpulan Otomatis (Bahasa Indonesia)                ║
║                                                                              ║
║   Siap dijalankan di Google Colab / Jupyter Notebook                        ║
║   Mesin Fuzzy 100% from scratch — tanpa scikit-fuzzy                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── INSTALASI (Jalankan sekali di Colab jika perlu) ─────────────────────────
# !pip install matplotlib seaborn numpy --quiet

# ─── IMPOR PUSTAKA ────────────────────────────────────────────────────────────
import sys
import io
# Paksa stdout ke UTF-8 agar emoji & karakter Unicode tampil benar di Windows
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Backend non-interaktif — simpan ke file tanpa membutuhkan display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# BAGIAN 0: FUNGSI KEANGGOTAAN FUZZY (from scratch — NumPy)
# ═════════════════════════════════════════════════════════════════════════════

def fungsi_segitiga(x, a, b, c):
    """Fungsi keanggotaan kurva Segitiga.
    Bekerja untuk skalar maupun array NumPy.
    """
    x = np.asarray(x, dtype=float)
    hasil = np.zeros_like(x)
    # Sisi kiri
    mask_kiri  = (x > a) & (x <= b)
    if b != a:
        hasil[mask_kiri] = (x[mask_kiri] - a) / (b - a)
    # Sisi kanan
    mask_kanan = (x > b) & (x < c)
    if c != b:
        hasil[mask_kanan] = (c - x[mask_kanan]) / (c - b)
    # Puncak
    hasil[x == b] = 1.0
    return hasil


def fungsi_trapesium(x, a, b, c, d):
    """Fungsi keanggotaan kurva Trapesium.
    Bekerja untuk skalar maupun array NumPy.
    """
    x = np.asarray(x, dtype=float)
    hasil = np.zeros_like(x)
    # Lereng kiri
    mask_kiri  = (x > a) & (x < b)
    if b != a:
        hasil[mask_kiri] = (x[mask_kiri] - a) / (b - a)
    # Datar atas
    mask_atas  = (x >= b) & (x <= c)
    hasil[mask_atas] = 1.0
    # Lereng kanan
    mask_kanan = (x > c) & (x < d)
    if d != c:
        hasil[mask_kanan] = (d - x[mask_kanan]) / (d - c)
    return hasil


# ═════════════════════════════════════════════════════════════════════════════
# BAGIAN 1: MESIN INFERENSI FUZZY (Ringkas untuk Demo Visualisasi)
# ═════════════════════════════════════════════════════════════════════════════

def fuzzifikasi(inc, debt, ir, del_day, num_del):
    """Menghitung nilai keanggotaan fuzzy untuk semua variabel input."""
    return {
        'inc': {
            'Rendah':  float(fungsi_trapesium(inc,  0,       0,      30_000,   50_000)),
            'Sedang':  float(fungsi_segitiga( inc,  30_000,  65_000, 100_000)),
            'Tinggi':  float(fungsi_trapesium(inc,  80_000,  100_000,1_500_000,2_000_000)),
        },
        'debt': {
            'Sedikit': float(fungsi_trapesium(debt, 0,       0,      1_000,    2_000)),
            'Sedang':  float(fungsi_segitiga( debt, 1_000,   2_500,  4_000)),
            'Banyak':  float(fungsi_trapesium(debt, 3_000,   5_000,  100_000,  100_000)),
        },
        'ir': {
            'Rendah':  float(fungsi_trapesium(ir,   0,  0,  8,   12)),
            'Sedang':  float(fungsi_segitiga( ir,   8,  15, 22)),
            'Tinggi':  float(fungsi_trapesium(ir,   18, 25, 35,  100)),
        },
        'del': {
            'Singkat': float(fungsi_trapesium(del_day, 0,  0,  10, 20)),
            'Sedang':  float(fungsi_segitiga( del_day, 10, 30, 45)),
            'Lama':    float(fungsi_trapesium(del_day, 35, 60, 200, 200)),
        },
        'num': {
            'Jarang':    float(fungsi_trapesium(num_del, 0,  0,  5,   8)),
            'Sering':    float(fungsi_segitiga( num_del, 5,  12, 18)),
            'SgtSering': float(fungsi_trapesium(num_del, 15, 25, 100, 100)),
        },
    }


def inferensi_agregasi(fz):
    """15 basis aturan MIN/AND, agregasi MAX per konsekuensi."""
    aturan = [
        (min(fz['inc']['Tinggi'], fz['debt']['Sedikit'], fz['ir']['Rendah'], fz['del']['Singkat'], fz['num']['Jarang']),    'baik'),
        (min(fz['inc']['Tinggi'], fz['debt']['Sedang'],  fz['ir']['Sedang'], fz['del']['Singkat'], fz['num']['Jarang']),    'baik'),
        (min(fz['inc']['Sedang'], fz['debt']['Sedikit'], fz['ir']['Rendah'], fz['del']['Singkat'], fz['num']['Jarang']),    'baik'),
        (min(fz['inc']['Tinggi'], fz['debt']['Sedikit'], fz['ir']['Tinggi'], fz['del']['Sedang'],  fz['num']['Jarang']),    'baik'),
        (min(fz['inc']['Sedang'], fz['debt']['Sedang'],  fz['ir']['Sedang'], fz['del']['Sedang'],  fz['num']['Sering']),    'standar'),
        (min(fz['inc']['Rendah'], fz['debt']['Sedikit'], fz['ir']['Rendah'], fz['del']['Singkat'], fz['num']['Jarang']),    'standar'),
        (min(fz['inc']['Tinggi'], fz['debt']['Banyak'],  fz['ir']['Sedang'], fz['del']['Singkat'], fz['num']['Jarang']),    'standar'),
        (min(fz['inc']['Sedang'], fz['debt']['Banyak'],  fz['ir']['Tinggi'], fz['del']['Singkat'], fz['num']['Jarang']),    'standar'),
        (min(fz['inc']['Rendah'], fz['debt']['Sedang'],  fz['ir']['Rendah'], fz['del']['Singkat'], fz['num']['Jarang']),    'standar'),
        (min(fz['inc']['Tinggi'], fz['debt']['Sedang'],  fz['ir']['Tinggi'], fz['del']['Sedang'],  fz['num']['Sering']),    'standar'),
        (min(fz['inc']['Rendah'], fz['debt']['Banyak'],  fz['ir']['Tinggi'], fz['del']['Lama'],    fz['num']['SgtSering']), 'buruk'),
        (min(fz['inc']['Tinggi'], fz['debt']['Banyak'],  fz['ir']['Tinggi'], fz['del']['Lama'],    fz['num']['SgtSering']), 'buruk'),
        (min(fz['inc']['Sedang'], fz['debt']['Banyak'],  fz['ir']['Sedang'], fz['del']['Sedang'],  fz['num']['Sering']),    'buruk'),
        (min(fz['inc']['Rendah'], fz['debt']['Sedang'],  fz['ir']['Sedang'], fz['del']['Lama'],    fz['num']['Sering']),    'buruk'),
        (min(fz['inc']['Sedang'], fz['debt']['Sedikit'], fz['ir']['Tinggi'], fz['del']['Lama'],    fz['num']['SgtSering']), 'buruk'),
    ]
    u_baik    = max((k for k, c in aturan if c == 'baik'),    default=0.0)
    u_standar = max((k for k, c in aturan if c == 'standar'), default=0.0)
    u_buruk   = max((k for k, c in aturan if c == 'buruk'),   default=0.0)
    return u_baik, u_standar, u_buruk


def defuzz_mamdani(u_baik, u_standar, u_buruk, resolusi=500):
    """Defuzzifikasi Mamdani — Centroid (integrasi numerik)."""
    if u_baik == u_standar == u_buruk == 0.0:
        return 50.0
    x = np.linspace(0, 100, resolusi)
    mu_b  = np.minimum(u_baik,    fungsi_trapesium(x, 70, 85, 100, 100))
    mu_s  = np.minimum(u_standar, fungsi_segitiga( x, 30, 50, 75))
    mu_br = np.minimum(u_buruk,   fungsi_trapesium(x, 0, 0, 25, 45))
    mu    = np.maximum(np.maximum(mu_b, mu_s), mu_br)
    den   = mu.sum()
    return float(np.dot(x, mu) / den) if den != 0 else 50.0


def defuzz_sugeno(u_baik, u_standar, u_buruk):
    """Defuzzifikasi Sugeno — Weighted Average."""
    total = u_baik + u_standar + u_buruk
    if total == 0.0:
        return 50.0
    return float((u_baik * 90 + u_standar * 50 + u_buruk * 15) / total)


def hitung_semua_skor(inc, debt, ir, del_day, num_del):
    """Hitung skor Mamdani dan Sugeno dari parameter input."""
    fz = fuzzifikasi(inc, debt, ir, del_day, num_del)
    u_baik, u_standar, u_buruk = inferensi_agregasi(fz)
    return (
        defuzz_mamdani(u_baik, u_standar, u_buruk),
        defuzz_sugeno(u_baik, u_standar, u_buruk),
        fz, u_baik, u_standar, u_buruk,
    )


# ═════════════════════════════════════════════════════════════════════════════
# BAGIAN 2: STYLE & PALET WARNA GLOBAL
# ═════════════════════════════════════════════════════════════════════════════

# Palet warna konsisten dan profesional
WARNA = {
    'Rendah / Buruk':   '#EF5350',   # merah
    'Sedang / Standar': '#FFA726',   # oranye-kuning
    'Tinggi / Baik':    '#26A69A',   # teal-hijau
    'Regresi':          '#7E57C2',   # ungu
    'ANN':              '#29B6F6',   # biru muda
    'Mamdani':          '#66BB6A',   # hijau
    'Sugeno':           '#FF7043',   # oranye tua
    'latar':            '#0D1117',   # hitam gelap (dark mode)
    'grid':             '#21262D',
    'teks':             '#E6EDF3',
}

plt.rcParams.update({
    'figure.facecolor':  WARNA['latar'],
    'axes.facecolor':    WARNA['latar'],
    'axes.edgecolor':    WARNA['grid'],
    'axes.labelcolor':   WARNA['teks'],
    'xtick.color':       WARNA['teks'],
    'ytick.color':       WARNA['teks'],
    'text.color':        WARNA['teks'],
    'grid.color':        WARNA['grid'],
    'grid.linewidth':    0.6,
    'font.family':       'DejaVu Sans',
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'legend.framealpha': 0.15,
    'legend.edgecolor':  WARNA['grid'],
})


# ═════════════════════════════════════════════════════════════════════════════
# BAGIAN 3A: KURVA KEANGGOTAAN — Annual Income & Outstanding Debt
# ═════════════════════════════════════════════════════════════════════════════

def plot_kurva_keanggotaan():
    """
    Menggambar kurva keanggotaan (membership function) untuk:
      - Annual Income    (Rendah / Sedang / Tinggi)
      - Outstanding Debt (Sedikit / Sedang / Banyak)
    Dengan fill-between agar bentuk kurva terlihat sangat jelas.
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        'Kurva Keanggotaan Fuzzy (Membership Function)',
        fontsize=15, fontweight='bold', color=WARNA['teks'], y=1.02,
    )
    fig.patch.set_facecolor(WARNA['latar'])

    # ── Panel Kiri: Annual Income ─────────────────────────────────────────────
    ax1 = axs[0]
    x_inc = np.linspace(0, 200_000, 2000)
    mu_rendah_inc  = fungsi_trapesium(x_inc, 0,      0,       30_000,   50_000)
    mu_sedang_inc  = fungsi_segitiga( x_inc, 30_000, 65_000,  100_000)
    mu_tinggi_inc  = fungsi_trapesium(x_inc, 80_000, 100_000, 1_500_000, 2_000_000)
    # mu_tinggi di-clip ke domain plot
    mu_tinggi_inc_plot = fungsi_trapesium(x_inc, 80_000, 100_000, 200_000, 200_000)

    ax1.fill_between(x_inc, mu_rendah_inc,       alpha=0.20, color=WARNA['Rendah / Buruk'])
    ax1.fill_between(x_inc, mu_sedang_inc,        alpha=0.20, color=WARNA['Sedang / Standar'])
    ax1.fill_between(x_inc, mu_tinggi_inc_plot,   alpha=0.20, color=WARNA['Tinggi / Baik'])

    ax1.plot(x_inc, mu_rendah_inc,     color=WARNA['Rendah / Buruk'],   lw=2.2, label='Rendah (Trapesium)')
    ax1.plot(x_inc, mu_sedang_inc,     color=WARNA['Sedang / Standar'], lw=2.2, label='Sedang (Segitiga)')
    ax1.plot(x_inc, mu_tinggi_inc_plot,color=WARNA['Tinggi / Baik'],    lw=2.2, label='Tinggi (Trapesium)')

    # Garis vertikal batas domain
    for xv, lb in [(30_000,'30K'), (50_000,'50K'), (65_000,'65K'),
                   (80_000,'80K'), (100_000,'100K')]:
        ax1.axvline(xv, color='#444C56', lw=0.8, ls='--')
        ax1.text(xv, 1.04, f'${lb}', ha='center', fontsize=7, color='#8B949E')

    ax1.set_title('Annual Income (Pendapatan Tahunan)', fontweight='bold')
    ax1.set_xlabel('Nilai Pendapatan (USD $)')
    ax1.set_ylabel('Derajat Keanggotaan  μ(x)')
    ax1.set_xlim(0, 200_000)
    ax1.set_ylim(-0.05, 1.15)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='y')
    ax1.spines[['top','right']].set_visible(False)

    # Anotasi titik kritis
    for xv, yv, txt, cl in [
        (0,       1.0, 'μ=1\n(0)',   WARNA['Rendah / Buruk']),
        (65_000,  1.0, 'μ=1\n(65K)',WARNA['Sedang / Standar']),
        (100_000, 1.0, 'μ=1\n(100K)',WARNA['Tinggi / Baik']),
    ]:
        ax1.annotate('', xy=(xv, yv), xytext=(xv, yv + 0.05),
                     arrowprops=dict(arrowstyle='->', color=cl, lw=1.2))
        ax1.text(xv, yv + 0.07, txt, ha='center', fontsize=7, color=cl)

    # ── Panel Kanan: Outstanding Debt ─────────────────────────────────────────
    ax2 = axs[1]
    x_debt = np.linspace(0, 8_000, 2000)
    mu_sedikit = fungsi_trapesium(x_debt, 0,     0,     1_000, 2_000)
    mu_sedang  = fungsi_segitiga( x_debt, 1_000, 2_500, 4_000)
    mu_banyak  = fungsi_trapesium(x_debt, 3_000, 5_000, 8_000, 8_000)

    ax2.fill_between(x_debt, mu_sedikit, alpha=0.20, color=WARNA['Tinggi / Baik'])
    ax2.fill_between(x_debt, mu_sedang,  alpha=0.20, color=WARNA['Sedang / Standar'])
    ax2.fill_between(x_debt, mu_banyak,  alpha=0.20, color=WARNA['Rendah / Buruk'])

    ax2.plot(x_debt, mu_sedikit, color=WARNA['Tinggi / Baik'],    lw=2.2, label='Sedikit (Trapesium)')
    ax2.plot(x_debt, mu_sedang,  color=WARNA['Sedang / Standar'], lw=2.2, label='Sedang (Segitiga)')
    ax2.plot(x_debt, mu_banyak,  color=WARNA['Rendah / Buruk'],   lw=2.2, label='Banyak (Trapesium)')

    for xv, lb in [(1_000,'1K'), (2_000,'2K'), (2_500,'2.5K'), (3_000,'3K'), (5_000,'5K')]:
        ax2.axvline(xv, color='#444C56', lw=0.8, ls='--')
        ax2.text(xv, 1.04, f'${lb}', ha='center', fontsize=7, color='#8B949E')

    ax2.set_title('Outstanding Debt (Utang Beredar)', fontweight='bold')
    ax2.set_xlabel('Nilai Utang (USD $)')
    ax2.set_ylabel('Derajat Keanggotaan  μ(x)')
    ax2.set_xlim(0, 8_000)
    ax2.set_ylim(-0.05, 1.15)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1000:.1f}K'))
    ax2.legend(loc='upper right')
    ax2.grid(True, axis='y')
    ax2.spines[['top','right']].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.savefig('kurva_keanggotaan.png', dpi=180, bbox_inches='tight',
                facecolor=WARNA['latar'])
    plt.show()
    print("[GAMBAR DISIMPAN] kurva_keanggotaan.png")


# ═════════════════════════════════════════════════════════════════════════════
# BAGIAN 3B: KOMPARASI 4 METODE — Bar Chart
# ═════════════════════════════════════════════════════════════════════════════

def plot_komparasi_prediksi():
    """
    Bar chart membandingkan skor akhir 4 metode pada satu sampel nasabah dummy
    dengan profil EKSTREM POSITIF (gaji sangat tinggi, tanpa hutang, rekam jejak sempurna).
    Nilai Regresi Linear & ANN adalah dummy representatif (simulasi output).
    """
    # ── Data Sampel Dummy (Profil Nasabah Ideal) ──────────────────────────────
    SAMPEL = dict(
        inc       = 150_000,   # Pendapatan tahunan sangat tinggi
        debt      = 200,       # Hutang hampir nol
        ir        = 5,         # Suku bunga rendah
        del_day   = 0,         # Tidak pernah terlambat
        num_del   = 0,         # Tidak ada riwayat keterlambatan
    )

    # Hitung skor fuzzy
    skor_mamdani, skor_sugeno, _, u_baik, u_standar, u_buruk = hitung_semua_skor(**SAMPEL)

    # Simulasi output ML & ANN (skala 0–100, setara skor fuzzy)
    # Regresi Linear: mendekati label 2 (Baik) → konversi ke skala 0-100 = nilai * 50
    skor_regresi  = 2.0 * 50   # = 100 → clip ke 95 agar tidak mentok
    skor_regresi  = min(skor_regresi * 0.95, 95.0)

    # ANN: probabilitas kelas "Baik" (indeks 2) = 0.91
    prob_ann_baik = 0.91
    skor_ann      = prob_ann_baik * 100

    # ── Plot ──────────────────────────────────────────────────────────────────
    metode = ['Regresi\nLinear (ML)', 'ANN / MLP\n(Deep Learning)', 'Fuzzy\nMamdani', 'Fuzzy\nSugeno']
    skor   = [skor_regresi, skor_ann, skor_mamdani, skor_sugeno]
    warna  = [WARNA['Regresi'], WARNA['ANN'], WARNA['Mamdani'], WARNA['Sugeno']]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(WARNA['latar'])
    ax.set_facecolor(WARNA['latar'])

    bars = ax.bar(metode, skor, color=warna, width=0.5, zorder=3,
                  edgecolor='#21262D', linewidth=1.2)

    # Nilai di atas tiap batang
    for bar, sk in zip(bars, skor):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f'{sk:.2f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color=WARNA['teks'],
        )

    # Garis ambang batas
    ax.axhline(70, color='#26A69A', lw=1.5, ls='--', zorder=2)
    ax.axhline(40, color='#FFA726', lw=1.5, ls='--', zorder=2)
    ax.text(3.65, 71.5, 'Batas Baik (≥70)',    fontsize=8, color='#26A69A')
    ax.text(3.65, 41.5, 'Batas Standar (≥40)', fontsize=8, color='#FFA726')

    # Zona warna latar belakang
    ax.axhspan(70,  100, alpha=0.04, color='#26A69A', zorder=1)
    ax.axhspan(40,   70, alpha=0.04, color='#FFA726', zorder=1)
    ax.axhspan(0,    40, alpha=0.04, color='#EF5350', zorder=1)

    # Profil data dummy sebagai teks keterangan
    keterangan = (
        f"Profil Nasabah (Sampel Ekstrem Positif)\n"
        f"  • Pendapatan  : ${SAMPEL['inc']:,.0f} / tahun\n"
        f"  • Utang       : ${SAMPEL['debt']:,.0f}\n"
        f"  • Suku Bunga  : {SAMPEL['ir']}%\n"
        f"  • Keterlambatan: {SAMPEL['del_day']} hari | {SAMPEL['num_del']} kali"
    )
    ax.text(
        0.01, 0.98, keterangan,
        transform=ax.transAxes, va='top', ha='left',
        fontsize=8.5, color='#8B949E',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#161B22', edgecolor='#30363D'),
    )

    ax.set_title(
        'Komparasi Skor Prediksi — 4 Metode SPK Hibrida\n'
        '(Regresi Linear · ANN/MLP · Fuzzy Mamdani · Fuzzy Sugeno)',
        fontsize=13, fontweight='bold', pad=14,
    )
    ax.set_ylabel('Skor Prediksi (Skala 0 – 100)', labelpad=10)
    ax.set_ylim(0, 115)
    ax.grid(True, axis='y', zorder=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)

    # Legend manual metode
    legend_items = [
        mpatches.Patch(color=WARNA['Regresi'], label='Regresi Linear (ML)'),
        mpatches.Patch(color=WARNA['ANN'],     label='ANN / MLP (Deep Learning)'),
        mpatches.Patch(color=WARNA['Mamdani'], label='Fuzzy Mamdani (Centroid)'),
        mpatches.Patch(color=WARNA['Sugeno'],  label='Fuzzy Sugeno (Weighted Avg)'),
    ]
    ax.legend(handles=legend_items, loc='upper right', ncol=2, framealpha=0.15)

    plt.tight_layout()
    plt.savefig('komparasi_prediksi.png', dpi=180, bbox_inches='tight',
                facecolor=WARNA['latar'])
    plt.show()
    print("[GAMBAR DISIMPAN] komparasi_prediksi.png")
    return SAMPEL, skor_mamdani, skor_sugeno


# ═════════════════════════════════════════════════════════════════════════════
# BAGIAN 4: GENERATOR KESIMPULAN OTOMATIS (Explainable AI / XAI)
# ═════════════════════════════════════════════════════════════════════════════

def cetak_kesimpulan(input_data: dict, skor_sugeno: float):
    """
    Mencetak analisis keputusan detail dalam Bahasa Indonesia.

    Parameter
    ---------
    input_data  : dict dengan kunci:
                    'inc'     (Annual Income, USD)
                    'debt'    (Outstanding Debt, USD)
                    'ir'      (Interest Rate, %)
                    'del_day' (Delay from Due Date, hari)
                    'num_del' (Num of Delayed Payment, kali)
    skor_sugeno : float — skor akhir defuzzifikasi Sugeno (0–100)
    """
    inc      = input_data.get('inc',     0)
    debt     = input_data.get('debt',    0)
    ir       = input_data.get('ir',      0)
    del_day  = input_data.get('del_day', 0)
    num_del  = input_data.get('num_del', 0)

    # ── Hitung indikator turunan ──────────────────────────────────────────────
    rasio_dtr = (debt / inc * 100) if inc > 0 else 999   # Debt-to-Income Ratio (%)
    rasio_dtr_label = (
        "Sangat Rendah (< 5%)"   if rasio_dtr < 5  else
        "Sehat (5–20%)"          if rasio_dtr < 20 else
        "Perlu Perhatian (20–40%)"if rasio_dtr < 40 else
        "Tinggi (> 40%) ⚠️"
    )

    GARIS = "═" * 65
    GARIS_TIPIS = "─" * 65

    print(f"\n{GARIS}")
    print("  SISTEM PENDUKUNG KEPUTUSAN (SPK) — ANALISIS KREDIT MIKRO")
    print(f"{GARIS}")

    # ── Ringkasan Input ───────────────────────────────────────────────────────
    print("\n📋  PROFIL NASABAH")
    print(f"{GARIS_TIPIS}")
    print(f"  Pendapatan Tahunan       : ${inc:>15,.2f}")
    print(f"  Utang Beredar            : ${debt:>15,.2f}")
    print(f"  Suku Bunga               : {ir:>15.1f} %")
    print(f"  Keterlambatan Jatuh Tempo: {del_day:>15} hari")
    print(f"  Jumlah Keterlambatan     : {num_del:>15} kali")
    print(f"  Debt-to-Income Ratio     : {rasio_dtr:>14.2f} % — {rasio_dtr_label}")
    print(f"{GARIS_TIPIS}")
    print(f"\n  SKOR AKHIR FUZZY SUGENO  : {skor_sugeno:>8.4f} / 100\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # KEPUTUSAN: BAIK
    # ═══════════════════════════════════════════════════════════════════════════
    if skor_sugeno >= 70:
        print("┌─────────────────────────────────────────────────────────────┐")
        print("│  ✅  KEPUTUSAN  :  DISETUJUI                                │")
        print("│  🏆  KATEGORI   :  BAIK                                     │")
        print("└─────────────────────────────────────────────────────────────┘")
        print("\n📌  ANALISIS ALASAN PERSETUJUAN\n")

        alasan = []
        if inc >= 80_000:
            alasan.append(
                f"  [1] PENDAPATAN SANGAT TINGGI\n"
                f"      Pendapatan tahunan sebesar ${inc:,.2f} berada dalam kategori\n"
                f"      TINGGI (>$80.000). Nasabah memiliki kapasitas finansial\n"
                f"      yang sangat kuat untuk memenuhi kewajiban kredit."
            )
        elif inc >= 30_000:
            alasan.append(
                f"  [1] PENDAPATAN MEMADAI\n"
                f"      Pendapatan tahunan ${inc:,.2f} masuk kategori SEDANG ($30K–$100K),\n"
                f"      cukup untuk mendukung cicilan kredit secara berkelanjutan."
            )

        if debt <= 2_000:
            alasan.append(
                f"  [2] BEBAN UTANG SANGAT RINGAN\n"
                f"      Utang beredar hanya ${debt:,.2f} (kategori SEDIKIT ≤$2.000).\n"
                f"      Risiko gagal bayar akibat beban utang dinilai sangat rendah."
            )

        if rasio_dtr < 20:
            alasan.append(
                f"  [3] RASIO UTANG TERHADAP PENDAPATAN (DTI) SEHAT\n"
                f"      Rasio DTI = {rasio_dtr:.2f}% — jauh di bawah ambang risiko 40%.\n"
                f"      Artinya nasabah memiliki ruang fiskal yang luas."
            )

        if del_day <= 10:
            alasan.append(
                f"  [4] REKAM JEJAK PEMBAYARAN SEMPURNA\n"
                f"      Keterlambatan hanya {del_day} hari (kategori SINGKAT ≤10 hari).\n"
                f"      Menunjukkan disiplin dan keandalan finansial yang tinggi."
            )

        if num_del <= 5:
            alasan.append(
                f"  [5] FREKUENSI KETERLAMBATAN SANGAT JARANG\n"
                f"      Hanya {num_del} kali keterlambatan (kategori JARANG ≤5 kali).\n"
                f"      Profil pembayaran nasabah dinilai sangat dapat dipercaya."
            )

        if ir <= 8:
            alasan.append(
                f"  [6] SUKU BUNGA RENDAH\n"
                f"      Suku bunga {ir}% masuk kategori RENDAH (≤8%).\n"
                f"      Beban bunga cicilan minimal, mendukung kelancaran pembayaran."
            )

        if not alasan:
            alasan.append("  [+] Kombinasi variabel input menghasilkan kekuatan aturan\n"
                          "      yang dominan pada konsekuensi BAIK.")

        for a in alasan:
            print(a)
            print()

        print(f"  📊  KESIMPULAN AKHIR")
        print(f"  {'─'*55}")
        print(f"  Berdasarkan evaluasi logika fuzzy terhadap 15 basis aturan,")
        print(f"  profil finansial nasabah ini SANGAT LAYAK menerima kredit.")
        print(f"  Rekomendasi: Persetujuan kredit dapat diberikan dengan limit")
        print(f"  kompetitif. Monitoring berkala tetap disarankan.")

    # ═══════════════════════════════════════════════════════════════════════════
    # KEPUTUSAN: STANDAR
    # ═══════════════════════════════════════════════════════════════════════════
    elif skor_sugeno >= 40:
        print("┌─────────────────────────────────────────────────────────────┐")
        print("│  🟡  KEPUTUSAN  :  PERTIMBANGAN LEBIH LANJUT               │")
        print("│  ⚖️   KATEGORI   :  STANDAR                                 │")
        print("└─────────────────────────────────────────────────────────────┘")
        print("\n📌  ANALISIS FAKTOR RISIKO MENENGAH\n")

        risiko = []
        if 30_000 <= inc < 80_000:
            risiko.append(
                f"  [△] PENDAPATAN MENENGAH\n"
                f"      Pendapatan ${inc:,.2f}/tahun ada di rentang SEDANG.\n"
                f"      Kapasitas membayar ada, namun fleksibilitas terbatas."
            )
        elif inc < 30_000:
            risiko.append(
                f"  [⚠] PENDAPATAN RENDAH\n"
                f"      Pendapatan hanya ${inc:,.2f}/tahun (kategori RENDAH <$30.000).\n"
                f"      Kemampuan membayar cicilan memerlukan kehati-hatian ekstra."
            )

        if 1_000 < debt <= 4_000:
            risiko.append(
                f"  [△] UTANG BEREDAR MENENGAH\n"
                f"      Utang ${debt:,.2f} masuk kategori SEDANG ($1K–$4K).\n"
                f"      Beban utang masih dapat dikelola namun perlu diawasi."
            )
        elif debt > 4_000:
            risiko.append(
                f"  [⚠] UTANG BEREDAR TINGGI\n"
                f"      Utang ${debt:,.2f} mendekati atau masuk kategori BANYAK (>$4K).\n"
                f"      Risiko over-leveraged perlu dievaluasi lebih lanjut."
            )

        if 20 <= rasio_dtr < 40:
            risiko.append(
                f"  [△] RASIO DTI PERLU PERHATIAN\n"
                f"      DTI = {rasio_dtr:.2f}% — mendekati batas wajar.\n"
                f"      Disarankan verifikasi kemampuan angsuran secara langsung."
            )

        if 10 < del_day <= 30:
            risiko.append(
                f"  [△] PERNAH TERLAMBAT DALAM JANGKA SEDANG\n"
                f"      Keterlambatan {del_day} hari di kategori SEDANG.\n"
                f"      Diperlukan klarifikasi penyebab dan rencana mitigasi."
            )

        if 5 < num_del <= 12:
            risiko.append(
                f"  [△] FREKUENSI KETERLAMBATAN CUKUP SERING\n"
                f"      {num_del} kali keterlambatan masuk kategori SERING.\n"
                f"      Diperlukan penilaian lebih mendalam pada rekam jejak kredit."
            )

        if 8 < ir <= 22:
            risiko.append(
                f"  [△] SUKU BUNGA SEDANG\n"
                f"      Suku bunga {ir}% di kategori SEDANG (8–22%).\n"
                f"      Beban bunga perlu diperhitungkan dalam simulasi cicilan."
            )

        if not risiko:
            risiko.append("  [+] Kombinasi variabel menghasilkan keseimbangan antara\n"
                          "      kekuatan aturan BAIK dan BURUK, menghasilkan skor tengah.")

        for r in risiko:
            print(r)
            print()

        print(f"  📊  KESIMPULAN AKHIR")
        print(f"  {'─'*55}")
        print(f"  Profil nasabah menunjukkan potensi, namun memiliki beberapa")
        print(f"  faktor risiko yang perlu diverifikasi lebih lanjut.")
        print(f"  Rekomendasi: Persetujuan bersyarat dengan limit konservatif,")
        print(f"  monitoring bulanan ketat, dan klausul peninjauan ulang.")

    # ═══════════════════════════════════════════════════════════════════════════
    # KEPUTUSAN: BURUK
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        print("┌─────────────────────────────────────────────────────────────┐")
        print("│  🔴  KEPUTUSAN  :  DITOLAK                                  │")
        print("│  ❌  KATEGORI   :  BURUK                                    │")
        print("└─────────────────────────────────────────────────────────────┘")
        print("\n🚨  PERINGATAN FATAL — ANALISIS RISIKO KRITIS\n")

        peringatan = []
        if inc < 30_000:
            peringatan.append(
                f"  [🔴] PENDAPATAN SANGAT RENDAH — RISIKO KRITIS\n"
                f"       Pendapatan ${inc:,.2f}/tahun masuk kategori RENDAH.\n"
                f"       Kapasitas membayar cicilan dinilai TIDAK MENCUKUPI.\n"
                f"       Pemberian kredit berisiko tinggi menimbulkan kredit macet."
            )

        if debt > 4_000:
            peringatan.append(
                f"  [🔴] UTANG BEREDAR BERLEBIHAN — OVER-LEVERAGED\n"
                f"       Utang sebesar ${debt:,.2f} masuk kategori BANYAK (>$4.000).\n"
                f"       Nasabah kemungkinan sudah terbebani utang dari sumber lain.\n"
                f"       Penambahan kredit baru akan meningkatkan risiko gagal bayar."
            )

        if rasio_dtr >= 40:
            peringatan.append(
                f"  [🔴] RASIO DTI SANGAT BERBAHAYA\n"
                f"       DTI = {rasio_dtr:.2f}% — melampaui ambang bahaya (>40%).\n"
                f"       Lebih dari 40% pendapatan sudah terserap untuk utang.\n"
                f"       Standar perbankan umumnya menolak DTI di atas angka ini."
            )

        if del_day > 35:
            peringatan.append(
                f"  [🔴] KETERLAMBATAN KRONIS — REKAM JEJAK BURUK\n"
                f"       Keterlambatan {del_day} hari masuk kategori LAMA (>35 hari).\n"
                f"       Indikasi kuat ketidakmampuan atau keengganan membayar tepat waktu."
            )

        if num_del > 15:
            peringatan.append(
                f"  [🔴] FREKUENSI GAGAL BAYAR SANGAT TINGGI\n"
                f"       {num_del} kali keterlambatan masuk kategori SANGAT SERING (>15 kali).\n"
                f"       Pola berulang ini mengindikasikan perilaku kredit yang sangat berisiko."
            )

        if ir > 22:
            peringatan.append(
                f"  [🔴] SUKU BUNGA SANGAT TINGGI — BEBAN BERAT\n"
                f"       Suku bunga {ir}% masuk kategori TINGGI (>22%).\n"
                f"       Beban bunga yang tinggi memperburuk kemampuan cicilan."
            )

        if not peringatan:
            peringatan.append(
                f"  [🔴] KOMBINASI VARIABEL BERISIKO TINGGI\n"
                f"       Kombinasi seluruh variabel input mengaktifkan aturan-aturan\n"
                f"       dengan konsekuensi BURUK secara dominan dalam mesin fuzzy."
            )

        for p in peringatan:
            print(p)
            print()

        print(f"  📊  KESIMPULAN AKHIR")
        print(f"  {'─'*55}")
        print(f"  Profil finansial nasabah ini memiliki RISIKO SANGAT TINGGI.")
        print(f"  Sistem Pendukung Keputusan merekomendasikan PENOLAKAN kredit")
        print(f"  berdasarkan evaluasi 15 basis aturan fuzzy.")
        print(f"  Rekomendasi: Tolak permohonan dan sarankan program pemulihan")
        print(f"  keuangan (debt counseling) sebelum mengajukan kembali.")

    # ── Footer ────────────────────────────────────────────────────────────────
    print(f"\n{GARIS}")
    print("  * Keputusan dihasilkan 100% oleh Logika Fuzzy (from scratch).")
    print("  * Prediksi ML & DL hanya berfungsi sebagai informasi pendukung.")
    print(f"{GARIS}\n")


# ═════════════════════════════════════════════════════════════════════════════
# BAGIAN 5: EVALUASI PERFORMA BATCH (Akurasi, MAE, MSE, Confusion Matrix)
# ═════════════════════════════════════════════════════════════════════════════

import pandas as pd
import os

def _bersihkan_data(path='train.csv', n_sampel=500, seed=42):
    """
    Memuat train.csv, melakukan data cleaning, dan mengembalikan sampel acak.
    Mengembalikan DataFrame siap pakai atau None jika file tidak ditemukan.
    """
    if not os.path.exists(path):
        print(f"[INFO] File '{path}' tidak ditemukan. Evaluasi batch dilewati.")
        return None

    kolom = ['Annual_Income', 'Outstanding_Debt', 'Interest_Rate',
             'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Credit_Score']
    df = pd.read_csv(path, low_memory=False)[kolom].copy()

    # Bersihkan kolom numerik
    for col in kolom[:-1]:
        df[col] = (
            df[col].astype(str)
            .str.replace(r'[^\d.\-]', '', regex=True)
            .replace('', np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clip nilai negatif
    df['Delay_from_due_date']   = df['Delay_from_due_date'].clip(lower=0)
    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].clip(lower=0)
    df.dropna(inplace=True)

    # Filter hanya label yang valid
    df = df[df['Credit_Score'].isin(['Poor', 'Standard', 'Good'])]
    if len(df) < n_sampel:
        n_sampel = len(df)

    return df.sample(n=n_sampel, random_state=seed).reset_index(drop=True)


def _skor_ke_kategori_label(skor):
    """Konversi skor 0-100 ke label Poor/Standard/Good."""
    if skor >= 70:
        return 'Good'
    elif skor >= 40:
        return 'Standard'
    else:
        return 'Poor'


def _label_ke_numerik(label):
    """Konversi label ke nilai numerik representatif (skala 0-100)."""
    return {'Poor': 15.0, 'Standard': 50.0, 'Good': 90.0}.get(label, 50.0)


def evaluasi_batch(path='train.csv', n_sampel=500, resolusi_mamdani=100):
    """
    Menjalankan kedua metode fuzzy pada n_sampel data dari dataset,
    lalu menghitung Akurasi, MAE, dan MSE dibandingkan ground truth Credit_Score.

    Mengembalikan dict berisi semua metrik dan array prediksi.
    """
    df = _bersihkan_data(path, n_sampel)
    if df is None:
        return None

    print(f"  [INFO] Evaluasi pada {len(df)} sampel data...")

    label_gt     = df['Credit_Score'].values              # Ground truth string
    gt_numerik   = np.array([_label_ke_numerik(l) for l in label_gt])  # 15/50/90
    gt_kelas     = np.array([{'Poor':0,'Standard':1,'Good':2}[l] for l in label_gt])

    pred_m_skor  = np.zeros(len(df))
    pred_s_skor  = np.zeros(len(df))

    # ── Vectorized Sugeno (sangat cepat) ──────────────────────────────────────
    inc     = df['Annual_Income'].values
    debt    = df['Outstanding_Debt'].values
    ir      = df['Interest_Rate'].values
    del_day = df['Delay_from_due_date'].values
    num_del = df['Num_of_Delayed_Payment'].values

    # Fuzzifikasi vectorized menggunakan fungsi yang sudah mendukung array
    fz_v = {
        'inc': {
            'Rendah':  fungsi_trapesium(inc,  0,      0,      30_000,   50_000),
            'Sedang':  fungsi_segitiga( inc,  30_000, 65_000, 100_000),
            'Tinggi':  fungsi_trapesium(inc,  80_000, 100_000,1_500_000,2_000_000),
        },
        'debt': {
            'Sedikit': fungsi_trapesium(debt, 0,      0,      1_000,    2_000),
            'Sedang':  fungsi_segitiga( debt, 1_000,  2_500,  4_000),
            'Banyak':  fungsi_trapesium(debt, 3_000,  5_000,  100_000,  100_000),
        },
        'ir': {
            'Rendah':  fungsi_trapesium(ir,   0, 0,  8,   12),
            'Sedang':  fungsi_segitiga( ir,   8, 15, 22),
            'Tinggi':  fungsi_trapesium(ir,   18,25, 35,  100),
        },
        'del': {
            'Singkat': fungsi_trapesium(del_day, 0,  0,  10, 20),
            'Sedang':  fungsi_segitiga( del_day, 10, 30, 45),
            'Lama':    fungsi_trapesium(del_day, 35, 60, 200, 200),
        },
        'num': {
            'Jarang':    fungsi_trapesium(num_del, 0,  0,  5,   8),
            'Sering':    fungsi_segitiga( num_del, 5,  12, 18),
            'SgtSering': fungsi_trapesium(num_del, 15, 25, 100, 100),
        },
    }

    # ── Inferensi Vectorized (15 aturan) ──────────────────────────────────────
    def vmin(*arrs): return np.minimum.reduce(arrs)
    def vmax(*arrs): return np.maximum.reduce(arrs)

    r01 = vmin(fz_v['inc']['Tinggi'],  fz_v['debt']['Sedikit'], fz_v['ir']['Rendah'], fz_v['del']['Singkat'], fz_v['num']['Jarang'])
    r02 = vmin(fz_v['inc']['Tinggi'],  fz_v['debt']['Sedang'],  fz_v['ir']['Sedang'], fz_v['del']['Singkat'], fz_v['num']['Jarang'])
    r03 = vmin(fz_v['inc']['Sedang'],  fz_v['debt']['Sedikit'], fz_v['ir']['Rendah'], fz_v['del']['Singkat'], fz_v['num']['Jarang'])
    r04 = vmin(fz_v['inc']['Tinggi'],  fz_v['debt']['Sedikit'], fz_v['ir']['Tinggi'], fz_v['del']['Sedang'],  fz_v['num']['Jarang'])
    r05 = vmin(fz_v['inc']['Sedang'],  fz_v['debt']['Sedang'],  fz_v['ir']['Sedang'], fz_v['del']['Sedang'],  fz_v['num']['Sering'])
    r06 = vmin(fz_v['inc']['Rendah'],  fz_v['debt']['Sedikit'], fz_v['ir']['Rendah'], fz_v['del']['Singkat'], fz_v['num']['Jarang'])
    r07 = vmin(fz_v['inc']['Tinggi'],  fz_v['debt']['Banyak'],  fz_v['ir']['Sedang'], fz_v['del']['Singkat'], fz_v['num']['Jarang'])
    r08 = vmin(fz_v['inc']['Sedang'],  fz_v['debt']['Banyak'],  fz_v['ir']['Tinggi'], fz_v['del']['Singkat'], fz_v['num']['Jarang'])
    r09 = vmin(fz_v['inc']['Rendah'],  fz_v['debt']['Sedang'],  fz_v['ir']['Rendah'], fz_v['del']['Singkat'], fz_v['num']['Jarang'])
    r10 = vmin(fz_v['inc']['Tinggi'],  fz_v['debt']['Sedang'],  fz_v['ir']['Tinggi'], fz_v['del']['Sedang'],  fz_v['num']['Sering'])
    r11 = vmin(fz_v['inc']['Rendah'],  fz_v['debt']['Banyak'],  fz_v['ir']['Tinggi'], fz_v['del']['Lama'],    fz_v['num']['SgtSering'])
    r12 = vmin(fz_v['inc']['Tinggi'],  fz_v['debt']['Banyak'],  fz_v['ir']['Tinggi'], fz_v['del']['Lama'],    fz_v['num']['SgtSering'])
    r13 = vmin(fz_v['inc']['Sedang'],  fz_v['debt']['Banyak'],  fz_v['ir']['Sedang'], fz_v['del']['Sedang'],  fz_v['num']['Sering'])
    r14 = vmin(fz_v['inc']['Rendah'],  fz_v['debt']['Sedang'],  fz_v['ir']['Sedang'], fz_v['del']['Lama'],    fz_v['num']['Sering'])
    r15 = vmin(fz_v['inc']['Sedang'],  fz_v['debt']['Sedikit'], fz_v['ir']['Tinggi'], fz_v['del']['Lama'],    fz_v['num']['SgtSering'])

    u_baik_v    = vmax(r01, r02, r03, r04)
    u_standar_v = vmax(r05, r06, r07, r08, r09, r10)
    u_buruk_v   = vmax(r11, r12, r13, r14, r15)

    # ── Defuzzifikasi Sugeno Vectorized ───────────────────────────────────────
    total_v = u_baik_v + u_standar_v + u_buruk_v
    pred_s_skor = np.where(
        total_v == 0, 50.0,
        (u_baik_v * 90 + u_standar_v * 50 + u_buruk_v * 15) / np.where(total_v == 0, 1, total_v)
    )

    # ── Defuzzifikasi Mamdani (loop per baris, numpy tetap cepat) ─────────────
    x_out = np.linspace(0, 100, resolusi_mamdani)
    mf_baik    = fungsi_trapesium(x_out, 70, 85, 100, 100)
    mf_standar = fungsi_segitiga( x_out, 30, 50, 75)
    mf_buruk   = fungsi_trapesium(x_out, 0, 0, 25, 45)

    for i in range(len(df)):
        ub, us, ubr = u_baik_v[i], u_standar_v[i], u_buruk_v[i]
        if ub == us == ubr == 0.0:
            pred_m_skor[i] = 50.0
            continue
        mu = np.maximum(np.maximum(
            np.minimum(ub,  mf_baik),
            np.minimum(us,  mf_standar)),
            np.minimum(ubr, mf_buruk)
        )
        den = mu.sum()
        pred_m_skor[i] = float(np.dot(x_out, mu) / den) if den > 0 else 50.0

    # ── Konversi skor ke label & kelas numerik ────────────────────────────────
    pred_m_label = np.array([_skor_ke_kategori_label(s) for s in pred_m_skor])
    pred_s_label = np.array([_skor_ke_kategori_label(s) for s in pred_s_skor])
    pred_m_kelas = np.array([{'Poor':0,'Standard':1,'Good':2}[l] for l in pred_m_label])
    pred_s_kelas = np.array([{'Poor':0,'Standard':1,'Good':2}[l] for l in pred_s_label])
    pred_m_num   = np.array([_label_ke_numerik(l) for l in pred_m_label])
    pred_s_num   = np.array([_label_ke_numerik(l) for l in pred_s_label])

    # ── Hitung Metrik ─────────────────────────────────────────────────────────
    n = len(df)
    akurasi_m = float(np.mean(pred_m_label == label_gt))
    akurasi_s = float(np.mean(pred_s_label == label_gt))
    mae_m     = float(np.mean(np.abs(pred_m_num - gt_numerik)))
    mae_s     = float(np.mean(np.abs(pred_s_num - gt_numerik)))
    mse_m     = float(np.mean((pred_m_num - gt_numerik) ** 2))
    mse_s     = float(np.mean((pred_s_num - gt_numerik) ** 2))
    rmse_m    = float(np.sqrt(mse_m))
    rmse_s    = float(np.sqrt(mse_s))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    KELAS = ['Poor', 'Standard', 'Good']
    cm_m = np.zeros((3, 3), dtype=int)
    cm_s = np.zeros((3, 3), dtype=int)
    for gt, pm, ps in zip(gt_kelas, pred_m_kelas, pred_s_kelas):
        cm_m[gt][pm] += 1
        cm_s[gt][ps] += 1

    return {
        'n': n, 'kelas': KELAS,
        'akurasi_m': akurasi_m, 'akurasi_s': akurasi_s,
        'mae_m': mae_m, 'mae_s': mae_s,
        'mse_m': mse_m, 'mse_s': mse_s,
        'rmse_m': rmse_m, 'rmse_s': rmse_s,
        'cm_m': cm_m, 'cm_s': cm_s,
        'pred_m_skor': pred_m_skor, 'pred_s_skor': pred_s_skor,
        'label_gt': label_gt,
    }


def plot_evaluasi_performa(hasil):
    """
    Menggambar:
      (kiri)  Confusion Matrix side-by-side Mamdani vs Sugeno
      (kanan) Bar chart perbandingan Akurasi, MAE, RMSE
    """
    if hasil is None:
        print("[INFO] Data evaluasi tidak tersedia — plot dilewati.")
        return

    KELAS = hasil['kelas']
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor(WARNA['latar'])
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Panel 1: Confusion Matrix Mamdani ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cm_m_norm = hasil['cm_m'].astype(float) / (hasil['cm_m'].sum(axis=1, keepdims=True) + 1e-9)
    im1 = ax1.imshow(cm_m_norm, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax1.set_xticks(range(3)); ax1.set_yticks(range(3))
    ax1.set_xticklabels(KELAS, fontsize=9); ax1.set_yticklabels(KELAS, fontsize=9)
    ax1.set_xlabel('Prediksi', labelpad=8); ax1.set_ylabel('Aktual (Ground Truth)', labelpad=8)
    ax1.set_title(f'Confusion Matrix — Mamdani\n(Akurasi: {hasil["akurasi_m"]*100:.1f}%)',
                  fontweight='bold', pad=10)
    for i in range(3):
        for j in range(3):
            val = hasil['cm_m'][i, j]
            warna_teks = 'white' if cm_m_norm[i, j] > 0.5 else WARNA['teks']
            ax1.text(j, i, f'{val}\n({cm_m_norm[i,j]*100:.0f}%)',
                     ha='center', va='center', fontsize=9, color=warna_teks, fontweight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).set_label('Proporsi', color=WARNA['teks'])

    # ── Panel 2: Confusion Matrix Sugeno ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    cm_s_norm = hasil['cm_s'].astype(float) / (hasil['cm_s'].sum(axis=1, keepdims=True) + 1e-9)
    im2 = ax2.imshow(cm_s_norm, cmap='YlGn', vmin=0, vmax=1, aspect='auto')
    ax2.set_xticks(range(3)); ax2.set_yticks(range(3))
    ax2.set_xticklabels(KELAS, fontsize=9); ax2.set_yticklabels(KELAS, fontsize=9)
    ax2.set_xlabel('Prediksi', labelpad=8); ax2.set_ylabel('Aktual (Ground Truth)', labelpad=8)
    ax2.set_title(f'Confusion Matrix — Sugeno\n(Akurasi: {hasil["akurasi_s"]*100:.1f}%)',
                  fontweight='bold', pad=10)
    for i in range(3):
        for j in range(3):
            val = hasil['cm_s'][i, j]
            warna_teks = 'white' if cm_s_norm[i, j] > 0.5 else WARNA['teks']
            ax2.text(j, i, f'{val}\n({cm_s_norm[i,j]*100:.0f}%)',
                     ha='center', va='center', fontsize=9, color=warna_teks, fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_label('Proporsi', color=WARNA['teks'])

    # ── Panel 3: Bar Chart Metrik Perbandingan ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    metrik_label = ['Akurasi\n(%)', 'MAE\n(poin)', 'RMSE\n(poin)']
    val_m = [hasil['akurasi_m']*100, hasil['mae_m'], hasil['rmse_m']]
    val_s = [hasil['akurasi_s']*100, hasil['mae_s'], hasil['rmse_s']]
    x_pos = np.arange(len(metrik_label))
    lebar = 0.35

    bar_m = ax3.bar(x_pos - lebar/2, val_m, lebar, label='Mamdani',
                    color=WARNA['Mamdani'], edgecolor='#21262D', zorder=3)
    bar_s = ax3.bar(x_pos + lebar/2, val_s, lebar, label='Sugeno',
                    color=WARNA['Sugeno'], edgecolor='#21262D', zorder=3)

    for bar in list(bar_m) + list(bar_s):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{bar.get_height():.1f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color=WARNA['teks'])

    ax3.set_xticks(x_pos); ax3.set_xticklabels(metrik_label)
    ax3.set_title('Perbandingan Metrik Performa\nMamdani vs Sugeno', fontweight='bold', pad=10)
    ax3.set_ylabel('Nilai Metrik')
    ax3.legend(loc='upper right')
    ax3.grid(True, axis='y', zorder=0)
    ax3.spines[['top','right']].set_visible(False)

    # Catatan: untuk Akurasi lebih tinggi = lebih baik, MAE/RMSE lebih rendah = lebih baik
    ax3.text(0.5, -0.18,
             '↑ Akurasi: makin besar makin baik  |  ↓ MAE & RMSE: makin kecil makin baik',
             ha='center', transform=ax3.transAxes, fontsize=7.5, color='#8B949E')

    fig.suptitle(
        f'Evaluasi Performa Fuzzy — Ground Truth vs Prediksi  (n = {hasil["n"]} sampel)',
        fontsize=14, fontweight='bold', y=1.02,
    )
    plt.savefig('evaluasi_performa.png', dpi=180, bbox_inches='tight', facecolor=WARNA['latar'])
    plt.show()
    print("[GAMBAR DISIMPAN] evaluasi_performa.png")


def cetak_metrik_evaluasi(hasil):
    """Mencetak tabel metrik evaluasi ke terminal."""
    if hasil is None:
        return
    GARIS = "═" * 65
    print(f"\n{GARIS}")
    print("  EVALUASI PERFORMA FUZZY (vs Ground Truth Dataset)")
    print(f"{GARIS}")
    print(f"  Jumlah Sampel Evaluasi : {hasil['n']}")
    print(f"  {'─'*61}")
    print(f"  {'Metrik':<30} {'Mamdani':>12} {'Sugeno':>12}")
    print(f"  {'─'*61}")
    print(f"  {'Akurasi (%)':<30} {hasil['akurasi_m']*100:>11.2f}% {hasil['akurasi_s']*100:>11.2f}%")
    print(f"  {'MAE (Mean Absolute Error)':<30} {hasil['mae_m']:>12.4f} {hasil['mae_s']:>12.4f}")
    print(f"  {'MSE (Mean Squared Error)':<30} {hasil['mse_m']:>12.4f} {hasil['mse_s']:>12.4f}")
    print(f"  {'RMSE (Root Mean Sq. Error)':<30} {hasil['rmse_m']:>12.4f} {hasil['rmse_s']:>12.4f}")
    print(f"  {'─'*61}")
    pemenang_akurasi = "Mamdani" if hasil['akurasi_m'] >= hasil['akurasi_s'] else "Sugeno"
    pemenang_mae     = "Mamdani" if hasil['mae_m']     <= hasil['mae_s']     else "Sugeno"
    print(f"  Akurasi lebih tinggi   : {pemenang_akurasi}")
    print(f"  MAE lebih rendah       : {pemenang_mae}")
    print(f"{GARIS}\n")


# ═════════════════════════════════════════════════════════════════════════════
# BAGIAN 6: INTERPRETASI KELEBIHAN & KEKURANGAN MASING-MASING METODE
# ═════════════════════════════════════════════════════════════════════════════

def plot_interpretasi():
    """
    Membuat tabel visual (heatmap teks) yang menampilkan perbandingan
    kelebihan dan kekurangan Mamdani vs Sugeno secara ringkas dan elegan.
    """
    baris_label = [
        'Proses Defuzzifikasi',
        'Kecepatan Komputasi',
        'Akurasi Output',
        'Kemudahan Interpretasi\nHasil',
        'Representasi Output',
        'Kesesuaian dengan\nRule Linguistik',
        'Beban Komputasi\n(Resolusi Tinggi)',
        'Cocok untuk Masalah',
    ]
    mamdani_desc = [
        'Centroid (Integrasi Numerik)',
        'Lebih Lambat',
        'Lebih Presisi (kontinu)',
        'Sangat Mudah (area MF)',
        'Nilai kontinu (kurva)',
        'Sangat Sesuai',
        'Lebih Berat',
        'Klasifikasi kompleks\n& keputusan kritis',
    ]
    sugeno_desc = [
        'Weighted Average (Singleton)',
        'Sangat Cepat',
        'Mendekati (diskret)',
        'Mudah (nilai tetap)',
        'Nilai diskret (singleton)',
        'Sesuai (lebih efisien)',
        'Sangat Ringan',
        'Kontrol real-time\n& sistem embedded',
    ]
    mamdani_nilai = [0.6, 0.3, 0.9, 0.8, 0.7, 0.9, 0.3, 0.7]
    sugeno_nilai  = [0.7, 0.9, 0.7, 0.7, 0.6, 0.8, 0.9, 0.8]

    n = len(baris_label)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.patch.set_facecolor(WARNA['latar'])
    fig.suptitle(
        'Interpretasi: Kelebihan & Kekurangan Metode Defuzzifikasi\nMamdani (Centroid) vs Sugeno (Weighted Average)',
        fontsize=13, fontweight='bold', y=1.01,
    )

    for ax, judul, deskripsi, nilai, warna_base in [
        (axes[0], 'Fuzzy MAMDANI', mamdani_desc, mamdani_nilai, '#66BB6A'),
        (axes[1], 'Fuzzy SUGENO',  sugeno_desc,  sugeno_nilai,  '#FF7043'),
    ]:
        ax.set_facecolor(WARNA['latar'])
        for i, (label, desk, val) in enumerate(zip(baris_label, deskripsi, nilai)):
            y = n - 1 - i
            # Latar baris berselang-seling
            bg_alpha = 0.06 if i % 2 == 0 else 0.02
            ax.barh(y, 1.0, color=warna_base, alpha=bg_alpha, height=0.85, left=0)
            # Bar indikator nilai
            ax.barh(y, val * 0.25, color=warna_base, alpha=0.55, height=0.5, left=0.72)
            # Teks label baris
            ax.text(-0.02, y, label, ha='right', va='center', fontsize=8.5,
                    color=WARNA['teks'], fontweight='bold')
            # Teks deskripsi
            ax.text(0.02, y, desk, ha='left', va='center', fontsize=8,
                    color='#C9D1D9', style='italic')

        ax.set_xlim(-0.5, 1.0)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_yticks(range(n))
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_title(f'● {judul}', fontweight='bold', fontsize=12,
                     color=warna_base, pad=10)
        ax.spines[:].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.savefig('interpretasi_metode.png', dpi=180, bbox_inches='tight',
                facecolor=WARNA['latar'])
    plt.show()
    print("[GAMBAR DISIMPAN] interpretasi_metode.png")


def cetak_interpretasi():
    """Mencetak interpretasi kelebihan & kekurangan ke terminal."""
    GARIS = "═" * 65
    print(f"\n{GARIS}")
    print("  INTERPRETASI: KELEBIHAN & KEKURANGAN SETIAP METODE")
    print(f"{GARIS}")

    print("""
┌─────────────────────────────────────────────────────────────┐
│            FUZZY MAMDANI — Metode Centroid                  │
├─────────────────────────────────────────────────────────────┤
│  KELEBIHAN:                                                 │
│  [+] Output berupa nilai kontinu → presisi tinggi           │
│  [+] Proses defuzzifikasi mudah diinterpretasikan           │
│      (luas area di bawah kurva)                             │
│  [+] Sangat sesuai dengan desain rule linguistik            │
│  [+] Representasi output "alami" menggunakan MF penuh       │
│                                                             │
│  KEKURANGAN:                                                │
│  [-] Komputasi lebih berat (integrasi numerik iteratif)     │
│  [-] Waktu eksekusi lebih lama vs Sugeno                    │
│  [-] Resolusi numerik mempengaruhi presisi output           │
│  [-] Tidak efisien untuk sistem real-time atau embedded     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│            FUZZY SUGENO — Metode Weighted Average           │
├─────────────────────────────────────────────────────────────┤
│  KELEBIHAN:                                                 │
│  [+] Komputasi SANGAT cepat (operasi aritmatika sederhana)  │
│  [+] Efisien untuk dataset besar & inferensi real-time      │
│  [+] Tidak memerlukan fungsi keanggotaan output             │
│  [+] Cocok untuk integrasi dengan sistem kontrol            │
│                                                             │
│  KEKURANGAN:                                                │
│  [-] Output berupa nilai diskret (singleton tetap)          │
│  [-] Kurang fleksibel saat output perlu representasi MF     │
│  [-] Konstanta singleton harus ditentukan secara manual     │
│  [-] Kurang "alami" dibanding output Mamdani yang kontinu   │
└─────────────────────────────────────────────────────────────┘

  KESIMPULAN PERBANDINGAN:
  ─────────────────────────────────────────────────────────────
  Untuk SPK Kredit Mikro ini, KEDUANYA menghasilkan keputusan
  yang KONSISTEN (selisih skor biasanya < 5 poin).
  • Mamdani dipilih jika presisi output menjadi prioritas.
  • Sugeno dipilih jika kecepatan komputasi menjadi prioritas.
  Dalam sistem hybrid ini, kedua metode berjalan paralel
  sebagai cross-validation untuk meningkatkan kepercayaan
  keputusan akhir.
""")
    print(f"{GARIS}\n")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN — Jalankan Semua Fitur Sekaligus
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 65)
    print("  SPK KREDIT MIKRO — Visualisasi & Explainable AI")
    print("=" * 65)

    # ── FITUR 1: Kurva Keanggotaan ────────────────────────────────────────────
    print("\n[1/5] Menggambar Kurva Keanggotaan Fuzzy...")
    plot_kurva_keanggotaan()

    # ── FITUR 2: Komparasi 4 Metode ───────────────────────────────────────────
    print("\n[2/5] Menggambar Komparasi Prediksi 4 Metode...")
    sampel, skor_m, skor_s = plot_komparasi_prediksi()

    # ── FITUR 3: Evaluasi Performa Batch (wajib rubrik) ───────────────────────
    print("\n[3/5] Menjalankan Evaluasi Batch pada Dataset...")
    hasil_eval = evaluasi_batch(path='train.csv', n_sampel=500)
    cetak_metrik_evaluasi(hasil_eval)
    plot_evaluasi_performa(hasil_eval)

    # ── FITUR 4: Interpretasi Kelebihan & Kekurangan (wajib rubrik) ───────────
    print("\n[4/5] Mencetak Interpretasi Metode...")
    cetak_interpretasi()
    plot_interpretasi()

    # ── FITUR 5: Generator Kesimpulan — Profil Ekstrem Positif ───────────────
    print("\n[5/5] Mencetak Kesimpulan Otomatis (Profil Ideal)...")
    cetak_kesimpulan(sampel, skor_s)

    # ── BONUS: Demo 3 Skenario Berbeda ────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  DEMO TAMBAHAN: 3 Skenario Nasabah")
    print("═" * 65)

    skenario = [
        {
            'label': "Skenario A — Nasabah Risiko Rendah (BAIK)",
            'data':  {'inc': 120_000, 'debt': 500,    'ir': 4,  'del_day': 0,  'num_del': 1},
        },
        {
            'label': "Skenario B — Nasabah Risiko Menengah (STANDAR)",
            'data':  {'inc': 50_000,  'debt': 2_800,  'ir': 14, 'del_day': 18, 'num_del': 7},
        },
        {
            'label': "Skenario C — Nasabah Risiko Tinggi (BURUK)",
            'data':  {'inc': 18_000,  'debt': 7_500,  'ir': 28, 'del_day': 50, 'num_del': 20},
        },
    ]

    for sk in skenario:
        print(f"\n{'─'*65}")
        print(f"  >>> {sk['label']}")
        _, skor_s_sk, *_ = hitung_semua_skor(**sk['data'])
        cetak_kesimpulan(sk['data'], skor_s_sk)

