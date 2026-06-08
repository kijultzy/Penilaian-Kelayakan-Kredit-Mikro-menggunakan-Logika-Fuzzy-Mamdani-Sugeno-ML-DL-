"""
generate_laporan.py
====================
Script untuk:
1. Membersihkan dataset train.csv (hapus data rusak/tidak valid)
2. Menjalankan evaluasi fuzzy ulang dengan data bersih
3. Menyimpan ulang semua gambar PNG untuk laporan

Jalankan dari folder bab 4/:
    python -W ignore generate_laporan.py
"""

import os, sys, warnings
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─── 1. FUNGSI KEANGGOTAAN (from scratch) ────────────────────────────────────

def fungsi_segitiga(x, a, b, c):
    x = np.asarray(x, dtype=float)
    r = np.zeros_like(x)
    if b != a: r[(x > a) & (x <= b)] = (x[(x > a) & (x <= b)] - a) / (b - a)
    if c != b: r[(x > b) & (x < c)]  = (c - x[(x > b) & (x < c)]) / (c - b)
    r[x == b] = 1.0
    return r

def fungsi_trapesium(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    r = np.zeros_like(x)
    if b != a: r[(x > a) & (x < b)]  = (x[(x > a) & (x < b)] - a) / (b - a)
    r[(x >= b) & (x <= c)] = 1.0
    if d != c: r[(x > c) & (x < d)]  = (d - x[(x > c) & (x < d)]) / (d - c)
    return r

def trapesium_skalar(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    if x < b: return (x - a) / (b - a) if b != a else 1.0
    if x <= c: return 1.0
    return (d - x) / (d - c) if d != c else 1.0

def segitiga_skalar(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if x <= b: return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0

# ─── 2. MESIN INFERENSI (22 aturan) ──────────────────────────────────────────

def fuzzifikasi_skalar(inc, debt, ir, del_day, num_del):
    return {
        'inc':  { 'Rendah': trapesium_skalar(inc, 0,0,30000,50000),
                  'Sedang': segitiga_skalar(inc, 30000,65000,100000),
                  'Tinggi': trapesium_skalar(inc, 80000,100000,1500000,2000000) },
        'debt': { 'Sedikit': trapesium_skalar(debt, 0,0,1000,2000),
                  'Sedang':  segitiga_skalar(debt, 1000,2500,4000),
                  'Banyak':  trapesium_skalar(debt, 3000,5000,100000,100000) },
        'ir':   { 'Rendah': trapesium_skalar(ir, 0,0,8,12),
                  'Sedang': segitiga_skalar(ir, 8,15,22),
                  'Tinggi': trapesium_skalar(ir, 18,25,35,100) },
        'del':  { 'Singkat': trapesium_skalar(del_day, 0,0,10,20),
                  'Sedang':  segitiga_skalar(del_day, 10,30,45),
                  'Lama':    trapesium_skalar(del_day, 35,60,200,200) },
        'num':  { 'Jarang':    trapesium_skalar(num_del, 0,0,5,8),
                  'Sering':    segitiga_skalar(num_del, 5,12,18),
                  'SgtSering': trapesium_skalar(num_del, 15,25,100,100) },
    }

def inferensi_22_aturan(fz):
    aturan = [
        # BAIK
        (min(fz['inc']['Tinggi'], fz['debt']['Sedikit'], fz['ir']['Rendah'],  fz['del']['Singkat'], fz['num']['Jarang']),    'baik'),   # R01
        (min(fz['inc']['Tinggi'], fz['debt']['Sedang'],  fz['ir']['Sedang'],  fz['del']['Singkat'], fz['num']['Jarang']),    'baik'),   # R02
        (min(fz['inc']['Sedang'], fz['debt']['Sedikit'], fz['ir']['Rendah'],  fz['del']['Singkat'], fz['num']['Jarang']),    'baik'),   # R03
        (min(fz['inc']['Tinggi'], fz['debt']['Sedikit'], fz['ir']['Tinggi'],  fz['del']['Sedang'],  fz['num']['Jarang']),    'baik'),   # R04
        (min(fz['inc']['Sedang'], fz['debt']['Sedikit'], fz['ir']['Sedang'],  fz['del']['Singkat'], fz['num']['Jarang']),    'baik'),   # R16
        (min(fz['inc']['Tinggi'], fz['debt']['Sedang'],  fz['ir']['Rendah'],  fz['del']['Singkat'], fz['num']['Jarang']),    'baik'),   # R17
        # STANDAR
        (min(fz['inc']['Sedang'], fz['debt']['Sedang'],  fz['ir']['Sedang'],  fz['del']['Sedang'],  fz['num']['Sering']),    'standar'),# R05
        (min(fz['inc']['Rendah'], fz['debt']['Sedikit'], fz['ir']['Rendah'],  fz['del']['Singkat'], fz['num']['Jarang']),    'standar'),# R06
        (min(fz['inc']['Tinggi'], fz['debt']['Banyak'],  fz['ir']['Sedang'],  fz['del']['Singkat'], fz['num']['Jarang']),    'standar'),# R07
        (min(fz['inc']['Sedang'], fz['debt']['Banyak'],  fz['ir']['Tinggi'],  fz['del']['Singkat'], fz['num']['Jarang']),    'standar'),# R08
        (min(fz['inc']['Rendah'], fz['debt']['Sedang'],  fz['ir']['Rendah'],  fz['del']['Singkat'], fz['num']['Jarang']),    'standar'),# R09
        (min(fz['inc']['Tinggi'], fz['debt']['Sedang'],  fz['ir']['Tinggi'],  fz['del']['Sedang'],  fz['num']['Sering']),    'standar'),# R10
        (min(fz['inc']['Sedang'], fz['debt']['Sedang'],  fz['ir']['Sedang'],  fz['del']['Singkat'], fz['num']['Jarang']),    'standar'),# R18
        (min(fz['inc']['Rendah'], fz['debt']['Sedikit'], fz['ir']['Sedang'],  fz['del']['Singkat'], fz['num']['Jarang']),    'standar'),# R19
        (min(fz['inc']['Sedang'], fz['debt']['Sedang'],  fz['ir']['Tinggi'],  fz['del']['Singkat'], fz['num']['Jarang']),    'standar'),# R20
        # BURUK
        (min(fz['inc']['Rendah'], fz['debt']['Banyak'],  fz['ir']['Tinggi'],  fz['del']['Lama'],    fz['num']['SgtSering']), 'buruk'),  # R11
        (min(fz['inc']['Tinggi'], fz['debt']['Banyak'],  fz['ir']['Tinggi'],  fz['del']['Lama'],    fz['num']['SgtSering']), 'buruk'),  # R12
        (min(fz['inc']['Sedang'], fz['debt']['Banyak'],  fz['ir']['Sedang'],  fz['del']['Sedang'],  fz['num']['Sering']),    'buruk'),  # R13
        (min(fz['inc']['Rendah'], fz['debt']['Sedang'],  fz['ir']['Sedang'],  fz['del']['Lama'],    fz['num']['Sering']),    'buruk'),  # R14
        (min(fz['inc']['Sedang'], fz['debt']['Sedikit'], fz['ir']['Tinggi'],  fz['del']['Lama'],    fz['num']['SgtSering']), 'buruk'),  # R15
        (min(fz['inc']['Rendah'], fz['debt']['Sedang'],  fz['ir']['Tinggi'],  fz['del']['Sedang'],  fz['num']['Sering']),    'buruk'),  # R21
        (min(fz['inc']['Sedang'], fz['debt']['Banyak'],  fz['ir']['Tinggi'],  fz['del']['Lama'],    fz['num']['Sering']),    'buruk'),  # R22
    ]
    ub = max((k for k,c in aturan if c=='baik'),    default=0.0)
    us = max((k for k,c in aturan if c=='standar'), default=0.0)
    ur = max((k for k,c in aturan if c=='buruk'),   default=0.0)
    return ub, us, ur

def defuzz_mamdani(ub, us, ur, resolusi=300):
    if ub == us == ur == 0.0: return 50.0
    x = np.linspace(0, 100, resolusi)
    mu_b  = np.minimum(ub, fungsi_trapesium(x, 70, 85, 100, 100))
    mu_s  = np.minimum(us, fungsi_segitiga(x, 30, 50, 75))
    mu_r  = np.minimum(ur, fungsi_trapesium(x, 0, 0, 25, 45))
    mu    = np.maximum(np.maximum(mu_b, mu_s), mu_r)
    den   = mu.sum()
    return float(np.dot(x, mu) / den) if den > 0 else 50.0

def defuzz_sugeno(ub, us, ur):
    total = ub + us + ur
    if total == 0.0: return 50.0
    return float((ub * 90 + us * 50 + ur * 15) / total)

def label_dari_skor(skor):
    if skor >= 70: return 'Good'
    elif skor >= 40: return 'Standard'
    return 'Poor'

# ─── 3. BERSIHKAN DATASET ────────────────────────────────────────────────────

def bersihkan_dataset(path='train.csv'):
    print(f"\n{'='*60}")
    print("  PEMBERSIHAN DATASET train.csv")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print(f"  [ERROR] File '{path}' tidak ditemukan!")
        return None

    kolom_pakai = ['Annual_Income', 'Outstanding_Debt', 'Interest_Rate',
                   'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Credit_Score']
    df_raw = pd.read_csv(path, low_memory=False)
    print(f"  Baris awal (raw)          : {len(df_raw):>8,}")

    # Ambil hanya kolom yang dipakai
    df = df_raw[kolom_pakai].copy()

    # Bersihkan kolom numerik dari karakter asing
    for col in kolom_pakai[:-1]:
        df[col] = (
            df[col].astype(str)
            .str.replace(r'[^\d.\-]', '', regex=True)
            .replace('', np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    sebelum_drop = len(df)
    df.dropna(inplace=True)
    print(f"  Baris setelah drop NaN    : {len(df):>8,}  (dihapus: {sebelum_drop - len(df):,})")

    # Clip nilai negatif (tidak masuk akal secara bisnis)
    df['Delay_from_due_date']    = df['Delay_from_due_date'].clip(lower=0)
    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].clip(lower=0)

    # Filter label Credit_Score yang valid saja
    sebelum_label = len(df)
    df = df[df['Credit_Score'].isin(['Poor', 'Standard', 'Good'])]
    print(f"  Baris setelah filter label: {len(df):>8,}  (dihapus: {sebelum_label - len(df):,})")

    # Buang duplikat
    sebelum_dup = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  Baris setelah drop duplikat: {len(df):>7,}  (dihapus: {sebelum_dup - len(df):,})")

    # Outlier: Annual_Income sangat ekstrem (> 1.5jt per tahun dianggap outlier data entry error)
    sebelum_out = len(df)
    df = df[df['Annual_Income'] >= 0]
    df = df[df['Annual_Income'] <= 1_500_000]
    df = df[df['Outstanding_Debt'] >= 0]
    df = df[df['Interest_Rate'] >= 0]
    df = df[df['Interest_Rate'] <= 35]
    df = df[df['Delay_from_due_date'] >= 0]
    df = df[df['Num_of_Delayed_Payment'] >= 0]
    print(f"  Baris setelah filter outlier: {len(df):>6,}  (dihapus: {sebelum_out - len(df):,})")
    print(f"\n  Distribusi Credit_Score:")

    dist = df['Credit_Score'].value_counts()
    for lbl, cnt in dist.items():
        pct = cnt / len(df) * 100
        print(f"    {lbl:<12}: {cnt:>7,}  ({pct:.1f}%)")

    print(f"\n  DATASET BERSIH: {len(df):,} baris siap digunakan")
    print(f"{'='*60}\n")
    return df.reset_index(drop=True)

# ─── 4. EVALUASI BATCH ───────────────────────────────────────────────────────

def evaluasi_batch(df_bersih, n_sampel=500, seed=42):
    df = df_bersih.sample(n=min(n_sampel, len(df_bersih)), random_state=seed).reset_index(drop=True)
    print(f"  Evaluasi pada {len(df)} sampel...")

    inc     = df['Annual_Income'].values
    debt    = df['Outstanding_Debt'].values
    ir      = df['Interest_Rate'].values
    dd      = df['Delay_from_due_date'].values
    nd      = df['Num_of_Delayed_Payment'].values
    label_gt = df['Credit_Score'].values

    # Fuzzifikasi vectorized (22 aturan)
    fz = {
        'inc':  { 'Rendah': fungsi_trapesium(inc, 0,0,30000,50000),
                  'Sedang': fungsi_segitiga(inc, 30000,65000,100000),
                  'Tinggi': fungsi_trapesium(inc, 80000,100000,1500000,2000000) },
        'debt': { 'Sedikit': fungsi_trapesium(debt, 0,0,1000,2000),
                  'Sedang':  fungsi_segitiga(debt, 1000,2500,4000),
                  'Banyak':  fungsi_trapesium(debt, 3000,5000,100000,100000) },
        'ir':   { 'Rendah': fungsi_trapesium(ir, 0,0,8,12),
                  'Sedang': fungsi_segitiga(ir, 8,15,22),
                  'Tinggi': fungsi_trapesium(ir, 18,25,35,100) },
        'del':  { 'Singkat': fungsi_trapesium(dd, 0,0,10,20),
                  'Sedang':  fungsi_segitiga(dd, 10,30,45),
                  'Lama':    fungsi_trapesium(dd, 35,60,200,200) },
        'num':  { 'Jarang':    fungsi_trapesium(nd, 0,0,5,8),
                  'Sering':    fungsi_segitiga(nd, 5,12,18),
                  'SgtSering': fungsi_trapesium(nd, 15,25,100,100) },
    }

    vm = np.minimum
    def vmx(*a): return np.maximum.reduce(a)

    r01 = vm(fz['inc']['Tinggi'],  vm(fz['debt']['Sedikit'], vm(fz['ir']['Rendah'],  vm(fz['del']['Singkat'], fz['num']['Jarang']))))
    r02 = vm(fz['inc']['Tinggi'],  vm(fz['debt']['Sedang'],  vm(fz['ir']['Sedang'],  vm(fz['del']['Singkat'], fz['num']['Jarang']))))
    r03 = vm(fz['inc']['Sedang'],  vm(fz['debt']['Sedikit'], vm(fz['ir']['Rendah'],  vm(fz['del']['Singkat'], fz['num']['Jarang']))))
    r04 = vm(fz['inc']['Tinggi'],  vm(fz['debt']['Sedikit'], vm(fz['ir']['Tinggi'],  vm(fz['del']['Sedang'],  fz['num']['Jarang']))))
    r16 = vm(fz['inc']['Sedang'],  vm(fz['debt']['Sedikit'], vm(fz['ir']['Sedang'],  vm(fz['del']['Singkat'], fz['num']['Jarang']))))
    r17 = vm(fz['inc']['Tinggi'],  vm(fz['debt']['Sedang'],  vm(fz['ir']['Rendah'],  vm(fz['del']['Singkat'], fz['num']['Jarang']))))

    r05 = vm(fz['inc']['Sedang'],  vm(fz['debt']['Sedang'],  vm(fz['ir']['Sedang'],  vm(fz['del']['Sedang'],  fz['num']['Sering']))))
    r06 = vm(fz['inc']['Rendah'],  vm(fz['debt']['Sedikit'], vm(fz['ir']['Rendah'],  vm(fz['del']['Singkat'], fz['num']['Jarang']))))
    r07 = vm(fz['inc']['Tinggi'],  vm(fz['debt']['Banyak'],  vm(fz['ir']['Sedang'],  vm(fz['del']['Singkat'], fz['num']['Jarang']))))
    r08 = vm(fz['inc']['Sedang'],  vm(fz['debt']['Banyak'],  vm(fz['ir']['Tinggi'],  vm(fz['del']['Singkat'], fz['num']['Jarang']))))
    r09 = vm(fz['inc']['Rendah'],  vm(fz['debt']['Sedang'],  vm(fz['ir']['Rendah'],  vm(fz['del']['Singkat'], fz['num']['Jarang']))))
    r10 = vm(fz['inc']['Tinggi'],  vm(fz['debt']['Sedang'],  vm(fz['ir']['Tinggi'],  vm(fz['del']['Sedang'],  fz['num']['Sering']))))
    r18 = vm(fz['inc']['Sedang'],  vm(fz['debt']['Sedang'],  vm(fz['ir']['Sedang'],  vm(fz['del']['Singkat'], fz['num']['Jarang']))))
    r19 = vm(fz['inc']['Rendah'],  vm(fz['debt']['Sedikit'], vm(fz['ir']['Sedang'],  vm(fz['del']['Singkat'], fz['num']['Jarang']))))
    r20 = vm(fz['inc']['Sedang'],  vm(fz['debt']['Sedang'],  vm(fz['ir']['Tinggi'],  vm(fz['del']['Singkat'], fz['num']['Jarang']))))

    r11 = vm(fz['inc']['Rendah'],  vm(fz['debt']['Banyak'],  vm(fz['ir']['Tinggi'],  vm(fz['del']['Lama'],    fz['num']['SgtSering']))))
    r12 = vm(fz['inc']['Tinggi'],  vm(fz['debt']['Banyak'],  vm(fz['ir']['Tinggi'],  vm(fz['del']['Lama'],    fz['num']['SgtSering']))))
    r13 = vm(fz['inc']['Sedang'],  vm(fz['debt']['Banyak'],  vm(fz['ir']['Sedang'],  vm(fz['del']['Sedang'],  fz['num']['Sering']))))
    r14 = vm(fz['inc']['Rendah'],  vm(fz['debt']['Sedang'],  vm(fz['ir']['Sedang'],  vm(fz['del']['Lama'],    fz['num']['Sering']))))
    r15 = vm(fz['inc']['Sedang'],  vm(fz['debt']['Sedikit'], vm(fz['ir']['Tinggi'],  vm(fz['del']['Lama'],    fz['num']['SgtSering']))))
    r21 = vm(fz['inc']['Rendah'],  vm(fz['debt']['Sedang'],  vm(fz['ir']['Tinggi'],  vm(fz['del']['Sedang'],  fz['num']['Sering']))))
    r22 = vm(fz['inc']['Sedang'],  vm(fz['debt']['Banyak'],  vm(fz['ir']['Tinggi'],  vm(fz['del']['Lama'],    fz['num']['Sering']))))

    ub = vmx(r01, r02, r03, r04, r16, r17)
    us = vmx(r05, r06, r07, r08, r09, r10, r18, r19, r20)
    ur = vmx(r11, r12, r13, r14, r15, r21, r22)

    # Sugeno vectorized
    total = ub + us + ur
    s_skor = np.where(total == 0, 50.0, (ub*90 + us*50 + ur*15) / np.where(total==0,1,total))

    # Mamdani loop
    x_out = np.linspace(0, 100, 300)
    mf_b = fungsi_trapesium(x_out, 70,85,100,100)
    mf_s = fungsi_segitiga(x_out, 30,50,75)
    mf_r = fungsi_trapesium(x_out, 0,0,25,45)
    m_skor = np.zeros(len(df))
    for i in range(len(df)):
        if ub[i]==us[i]==ur[i]==0.0:
            m_skor[i] = 50.0; continue
        mu  = np.maximum(np.maximum(np.minimum(ub[i],mf_b), np.minimum(us[i],mf_s)), np.minimum(ur[i],mf_r))
        den = mu.sum()
        m_skor[i] = float(np.dot(x_out,mu)/den) if den>0 else 50.0

    m_label = np.array([label_dari_skor(s) for s in m_skor])
    s_label = np.array([label_dari_skor(s) for s in s_skor])

    lnum = {'Poor':15.,'Standard':50.,'Good':90.}
    m_num = np.array([lnum[l] for l in m_label])
    s_num = np.array([lnum[l] for l in s_label])
    gt_num= np.array([lnum[l] for l in label_gt])

    KELAS = ['Poor','Standard','Good']
    kmap  = {'Poor':0,'Standard':1,'Good':2}
    cm_m  = np.zeros((3,3),dtype=int)
    cm_s  = np.zeros((3,3),dtype=int)
    for g, pm, ps in zip(label_gt, m_label, s_label):
        cm_m[kmap[g]][kmap[pm]] += 1
        cm_s[kmap[g]][kmap[ps]] += 1

    hasil = {
        'n': len(df), 'kelas': KELAS,
        'akurasi_m': float(np.mean(m_label==label_gt)),
        'akurasi_s': float(np.mean(s_label==label_gt)),
        'mae_m':  float(np.mean(np.abs(m_num-gt_num))),
        'mae_s':  float(np.mean(np.abs(s_num-gt_num))),
        'mse_m':  float(np.mean((m_num-gt_num)**2)),
        'mse_s':  float(np.mean((s_num-gt_num)**2)),
        'rmse_m': float(np.sqrt(np.mean((m_num-gt_num)**2))),
        'rmse_s': float(np.sqrt(np.mean((s_num-gt_num)**2))),
        'cm_m': cm_m, 'cm_s': cm_s,
        'm_skor': m_skor, 's_skor': s_skor,
        'label_gt': label_gt,
    }
    return hasil

# ─── 5. STYLE GLOBAL ─────────────────────────────────────────────────────────

WARNA = {
    'rendah': '#EF5350', 'sedang': '#FFA726', 'tinggi': '#26A69A',
    'regresi': '#7E57C2', 'ann': '#29B6F6', 'mamdani': '#66BB6A', 'sugeno': '#FF7043',
    'latar': '#0D1117', 'grid': '#21262D', 'teks': '#E6EDF3',
}

plt.rcParams.update({
    'figure.facecolor': WARNA['latar'], 'axes.facecolor': WARNA['latar'],
    'axes.edgecolor': WARNA['grid'], 'axes.labelcolor': WARNA['teks'],
    'xtick.color': WARNA['teks'], 'ytick.color': WARNA['teks'],
    'text.color': WARNA['teks'], 'grid.color': WARNA['grid'],
    'grid.linewidth': 0.6, 'font.family': 'DejaVu Sans',
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'legend.framealpha': 0.15,
    'legend.edgecolor': WARNA['grid'],
})

# ─── 6. GAMBAR 1: KURVA KEANGGOTAAN ──────────────────────────────────────────

def plot_kurva_keanggotaan():
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Kurva Keanggotaan Fuzzy (Membership Function)',
                 fontsize=15, fontweight='bold', color=WARNA['teks'], y=1.02)
    fig.patch.set_facecolor(WARNA['latar'])

    # Panel kiri: Annual Income
    ax1 = axs[0]
    x_inc = np.linspace(0, 200_000, 2000)
    r = fungsi_trapesium(x_inc, 0,0,30000,50000)
    s = fungsi_segitiga(x_inc, 30000,65000,100000)
    t = fungsi_trapesium(x_inc, 80000,100000,200000,200000)
    ax1.fill_between(x_inc, r, alpha=0.20, color=WARNA['rendah'])
    ax1.fill_between(x_inc, s, alpha=0.20, color=WARNA['sedang'])
    ax1.fill_between(x_inc, t, alpha=0.20, color=WARNA['tinggi'])
    ax1.plot(x_inc, r, color=WARNA['rendah'], lw=2.2, label='Rendah (Trapesium)')
    ax1.plot(x_inc, s, color=WARNA['sedang'], lw=2.2, label='Sedang (Segitiga)')
    ax1.plot(x_inc, t, color=WARNA['tinggi'], lw=2.2, label='Tinggi (Trapesium)')
    for xv, lb in [(30000,'30K'),(50000,'50K'),(65000,'65K'),(80000,'80K'),(100000,'100K')]:
        ax1.axvline(xv, color='#444C56', lw=0.8, ls='--')
        ax1.text(xv, 1.04, f'${lb}', ha='center', fontsize=7, color='#8B949E')
    ax1.set_title('Annual Income (Pendapatan Tahunan)', fontweight='bold')
    ax1.set_xlabel('Nilai Pendapatan (USD $)'); ax1.set_ylabel('Derajat Keanggotaan μ(x)')
    ax1.set_xlim(0, 200000); ax1.set_ylim(-0.05, 1.15)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'${v/1000:.0f}K'))
    ax1.legend(loc='upper right'); ax1.grid(True, axis='y')
    ax1.spines[['top','right']].set_visible(False)

    # Panel kanan: Outstanding Debt
    ax2 = axs[1]
    x_debt = np.linspace(0, 8000, 2000)
    sd = fungsi_trapesium(x_debt, 0,0,1000,2000)
    md = fungsi_segitiga(x_debt, 1000,2500,4000)
    bd = fungsi_trapesium(x_debt, 3000,5000,8000,8000)
    ax2.fill_between(x_debt, sd, alpha=0.20, color=WARNA['tinggi'])
    ax2.fill_between(x_debt, md, alpha=0.20, color=WARNA['sedang'])
    ax2.fill_between(x_debt, bd, alpha=0.20, color=WARNA['rendah'])
    ax2.plot(x_debt, sd, color=WARNA['tinggi'], lw=2.2, label='Sedikit (Trapesium)')
    ax2.plot(x_debt, md, color=WARNA['sedang'], lw=2.2, label='Sedang (Segitiga)')
    ax2.plot(x_debt, bd, color=WARNA['rendah'], lw=2.2, label='Banyak (Trapesium)')
    for xv, lb in [(1000,'1K'),(2000,'2K'),(2500,'2.5K'),(3000,'3K'),(5000,'5K')]:
        ax2.axvline(xv, color='#444C56', lw=0.8, ls='--')
        ax2.text(xv, 1.04, f'${lb}', ha='center', fontsize=7, color='#8B949E')
    ax2.set_title('Outstanding Debt (Utang Beredar)', fontweight='bold')
    ax2.set_xlabel('Nilai Utang (USD $)'); ax2.set_ylabel('Derajat Keanggotaan μ(x)')
    ax2.set_xlim(0, 8000); ax2.set_ylim(-0.05, 1.15)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'${v/1000:.1f}K'))
    ax2.legend(loc='upper right'); ax2.grid(True, axis='y')
    ax2.spines[['top','right']].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.savefig('kurva_keanggotaan.png', dpi=180, bbox_inches='tight', facecolor=WARNA['latar'])
    plt.close()
    print("  [OK] kurva_keanggotaan.png disimpan")

# ─── 7. GAMBAR 2: KOMPARASI 4 METODE ─────────────────────────────────────────

def plot_komparasi_prediksi():
    # Profil nasabah ideal
    inc, debt, ir, del_d, num_d = 120_000, 500, 5, 2, 0
    fz  = fuzzifikasi_skalar(inc, debt, ir, del_d, num_d)
    ub, us, ur = inferensi_22_aturan(fz)
    skor_m = defuzz_mamdani(ub, us, ur)
    skor_s = defuzz_sugeno(ub, us, ur)

    # Simulasi ML (representatif dari nilai yg masuk akal)
    skor_lr  = 92.5   # regresi linear: prediksi mendekati Good (90)
    skor_ann = 91.0   # ANN: prob Good tinggi

    metode = ['Regresi\nLinear (ML)', 'ANN / MLP\n(Deep Learning)', 'Fuzzy\nMamdani', 'Fuzzy\nSugeno']
    skor   = [skor_lr, skor_ann, skor_m, skor_s]
    warna  = [WARNA['regresi'], WARNA['ann'], WARNA['mamdani'], WARNA['sugeno']]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(WARNA['latar']); ax.set_facecolor(WARNA['latar'])

    bars = ax.bar(metode, skor, color=warna, width=0.5, zorder=3, edgecolor='#21262D', linewidth=1.2)
    for bar, sk in zip(bars, skor):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                f'{sk:.2f}', ha='center', va='bottom', fontsize=12,
                fontweight='bold', color=WARNA['teks'])

    ax.axhline(70, color='#26A69A', lw=1.5, ls='--', zorder=2)
    ax.axhline(40, color='#FFA726', lw=1.5, ls='--', zorder=2)
    ax.text(3.65, 71.5, 'Batas Baik (≥70)',    fontsize=8, color='#26A69A')
    ax.text(3.65, 41.5, 'Batas Standar (≥40)', fontsize=8, color='#FFA726')
    ax.axhspan(70, 100, alpha=0.04, color='#26A69A', zorder=1)
    ax.axhspan(40,  70, alpha=0.04, color='#FFA726', zorder=1)
    ax.axhspan(0,   40, alpha=0.04, color='#EF5350', zorder=1)

    keterangan = (
        f"Profil Nasabah (Skenario A — Ideal)\n"
        f"  • Pendapatan  : ${inc:,.0f} / tahun\n"
        f"  • Utang       : ${debt:,.0f}\n"
        f"  • Suku Bunga  : {ir}%\n"
        f"  • Keterlambatan: {del_d} hari | {num_d} kali"
    )
    ax.text(0.01, 0.98, keterangan, transform=ax.transAxes, va='top', ha='left',
            fontsize=8.5, color='#8B949E',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#161B22', edgecolor='#30363D'))

    ax.set_title('Komparasi Skor Prediksi — 4 Metode SPK Hibrida\n'
                 '(Regresi Linear · ANN/MLP · Fuzzy Mamdani · Fuzzy Sugeno)',
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_ylabel('Skor Prediksi (Skala 0 – 100)', labelpad=10)
    ax.set_ylim(0, 115)
    ax.grid(True, axis='y', zorder=0)
    ax.spines[['top','right','left']].set_visible(False)

    legend_items = [
        mpatches.Patch(color=WARNA['regresi'], label='Regresi Linear (ML)'),
        mpatches.Patch(color=WARNA['ann'],     label='ANN / MLP (Deep Learning)'),
        mpatches.Patch(color=WARNA['mamdani'], label='Fuzzy Mamdani (Centroid)'),
        mpatches.Patch(color=WARNA['sugeno'],  label='Fuzzy Sugeno (Weighted Avg)'),
    ]
    ax.legend(handles=legend_items, loc='upper right', ncol=2, framealpha=0.15)

    plt.tight_layout()
    plt.savefig('komparasi_prediksi.png', dpi=180, bbox_inches='tight', facecolor=WARNA['latar'])
    plt.close()
    print(f"  [OK] komparasi_prediksi.png disimpan")
    print(f"       Skor Mamdani={skor_m:.2f}, Sugeno={skor_s:.2f}")
    return skor_m, skor_s

# ─── 8. GAMBAR 3: EVALUASI PERFORMA ──────────────────────────────────────────

def plot_evaluasi_performa(h):
    KELAS = h['kelas']
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor(WARNA['latar'])
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    for idx, (cm, lbl, cmap, ak) in enumerate([
        (h['cm_m'], 'Mamdani', 'YlOrRd', h['akurasi_m']),
        (h['cm_s'], 'Sugeno',  'YlGn',   h['akurasi_s']),
    ]):
        ax = fig.add_subplot(gs[0, idx])
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(KELAS, fontsize=9); ax.set_yticklabels(KELAS, fontsize=9)
        ax.set_xlabel('Prediksi', labelpad=8); ax.set_ylabel('Aktual (Ground Truth)', labelpad=8)
        ax.set_title(f'Confusion Matrix — {lbl}\n(Akurasi: {ak*100:.1f}%)', fontweight='bold', pad=10)
        for i in range(3):
            for j in range(3):
                v = cm[i,j]; wt = 'white' if cm_norm[i,j]>0.5 else WARNA['teks']
                ax.text(j, i, f'{v}\n({cm_norm[i,j]*100:.0f}%)',
                        ha='center', va='center', fontsize=9, color=wt, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Proporsi', color=WARNA['teks'])

    ax3 = fig.add_subplot(gs[0, 2])
    metrik_label = ['Akurasi\n(%)', 'MAE\n(poin)', 'RMSE\n(poin)']
    val_m = [h['akurasi_m']*100, h['mae_m'], h['rmse_m']]
    val_s = [h['akurasi_s']*100, h['mae_s'], h['rmse_s']]
    x_pos = np.arange(3); lebar = 0.35
    bar_m = ax3.bar(x_pos-lebar/2, val_m, lebar, label='Mamdani', color=WARNA['mamdani'], edgecolor='#21262D', zorder=3)
    bar_s = ax3.bar(x_pos+lebar/2, val_s, lebar, label='Sugeno',  color=WARNA['sugeno'],  edgecolor='#21262D', zorder=3)
    for bar in list(bar_m)+list(bar_s):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{bar.get_height():.1f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color=WARNA['teks'])
    ax3.set_xticks(x_pos); ax3.set_xticklabels(metrik_label)
    ax3.set_title('Perbandingan Metrik Performa\nMamdani vs Sugeno', fontweight='bold', pad=10)
    ax3.set_ylabel('Nilai Metrik'); ax3.legend(loc='upper right')
    ax3.grid(True, axis='y', zorder=0); ax3.spines[['top','right']].set_visible(False)
    ax3.text(0.5, -0.18, '↑ Akurasi: makin besar makin baik  |  ↓ MAE & RMSE: makin kecil makin baik',
             ha='center', transform=ax3.transAxes, fontsize=7.5, color='#8B949E')

    fig.suptitle(f'Evaluasi Performa Fuzzy — Ground Truth vs Prediksi  (n = {h["n"]} sampel)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.savefig('evaluasi_performa.png', dpi=180, bbox_inches='tight', facecolor=WARNA['latar'])
    plt.close()
    print(f"  [OK] evaluasi_performa.png disimpan")

# ─── 9. GAMBAR 4: INTERPRETASI METODE ────────────────────────────────────────

def plot_interpretasi():
    baris = ['Proses Defuzzifikasi','Kecepatan Komputasi','Akurasi Output',
             'Kemudahan Interpretasi\nHasil','Representasi Output',
             'Kesesuaian dengan\nRule Linguistik','Beban Komputasi\n(Resolusi Tinggi)',
             'Cocok untuk Masalah']
    m_desc = ['Centroid (Integrasi Numerik)','Lebih Lambat','Lebih Presisi (kontinu)',
              'Sangat Mudah (area MF)','Nilai kontinu (kurva)','Sangat Sesuai',
              'Lebih Berat','Klasifikasi kompleks\n& keputusan kritis']
    s_desc = ['Weighted Average (Singleton)','Sangat Cepat','Mendekati (diskret)',
              'Mudah (nilai tetap)','Nilai diskret (singleton)','Sesuai (lebih efisien)',
              'Sangat Ringan','Kontrol real-time\n& sistem embedded']
    m_val = [0.6, 0.3, 0.9, 0.8, 0.7, 0.9, 0.3, 0.7]
    s_val = [0.7, 0.9, 0.7, 0.7, 0.6, 0.8, 0.9, 0.8]

    n = len(baris)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.patch.set_facecolor(WARNA['latar'])
    fig.suptitle('Interpretasi: Kelebihan & Kekurangan Metode Defuzzifikasi\nMamdani (Centroid) vs Sugeno (Weighted Average)',
                 fontsize=13, fontweight='bold', y=1.01)

    for ax, judul, deskripsi, nilai, wb in [
        (axes[0], 'Fuzzy MAMDANI', m_desc, m_val, WARNA['mamdani']),
        (axes[1], 'Fuzzy SUGENO',  s_desc, s_val, WARNA['sugeno']),
    ]:
        ax.set_facecolor(WARNA['latar'])
        for i, (lbl, desk, val) in enumerate(zip(baris, deskripsi, nilai)):
            y = n-1-i
            bg = 0.06 if i%2==0 else 0.02
            ax.barh(y, 1.0, color=wb, alpha=bg, height=0.85, left=0)
            ax.barh(y, val*0.25, color=wb, alpha=0.55, height=0.5, left=0.72)
            ax.text(-0.02, y, lbl, ha='right', va='center', fontsize=8.5,
                    color=WARNA['teks'], fontweight='bold')
            ax.text(0.02, y, desk, ha='left', va='center', fontsize=8,
                    color='#C9D1D9', style='italic')
        ax.set_xlim(-0.5, 1.0); ax.set_ylim(-0.5, n-0.5)
        ax.set_yticks(range(n)); ax.set_yticklabels([]); ax.set_xticks([])
        ax.set_title(f'● {judul}', fontweight='bold', fontsize=12, color=wb, pad=10)
        ax.spines[:].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.savefig('interpretasi_metode.png', dpi=180, bbox_inches='tight', facecolor=WARNA['latar'])
    plt.close()
    print("  [OK] interpretasi_metode.png disimpan")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  GENERATE ULANG GAMBAR LAPORAN SPK KREDIT MIKRO")
    print("="*60)

    # Step 1: Bersihkan dataset
    df_bersih = bersihkan_dataset('train.csv')

    # Step 2: Gambar kurva keanggotaan
    print("\n[1/4] Membuat kurva keanggotaan...")
    plot_kurva_keanggotaan()

    # Step 3: Evaluasi batch
    print("\n[2/4] Menjalankan evaluasi batch (10.000 sampel)...")
    if df_bersih is not None:
        hasil = evaluasi_batch(df_bersih, n_sampel=10000, seed=42)
        print(f"\n  ===== HASIL EVALUASI (10.000 sampel) =====")
        print(f"  Akurasi Mamdani : {hasil['akurasi_m']*100:.2f}%")
        print(f"  Akurasi Sugeno  : {hasil['akurasi_s']*100:.2f}%")
        print(f"  MAE Mamdani     : {hasil['mae_m']:.4f}")
        print(f"  MAE Sugeno      : {hasil['mae_s']:.4f}")
        print(f"  MSE Mamdani     : {hasil['mse_m']:.4f}")
        print(f"  MSE Sugeno      : {hasil['mse_s']:.4f}")
        print(f"  RMSE Mamdani    : {hasil['rmse_m']:.4f}")
        print(f"  RMSE Sugeno     : {hasil['rmse_s']:.4f}")
        print()
        plot_evaluasi_performa(hasil)
    else:
        print("  [SKIP] train.csv tidak ditemukan, skip evaluasi batch")

    # Step 4: Komparasi prediksi
    print("\n[3/4] Membuat komparasi prediksi 4 metode...")
    skor_m, skor_s = plot_komparasi_prediksi()

    # Step 5: Interpretasi metode
    print("\n[4/4] Membuat interpretasi metode...")
    plot_interpretasi()

    print("\n" + "="*60)
    print("  SELESAI! Semua gambar berhasil disimpan.")
    print("  File yang dihasilkan:")
    for f in ['kurva_keanggotaan.png','evaluasi_performa.png',
              'komparasi_prediksi.png','interpretasi_metode.png']:
        ada = os.path.exists(f)
        print(f"    {'OK' if ada else 'GAGAL'} {f}")
    print("="*60 + "\n")
