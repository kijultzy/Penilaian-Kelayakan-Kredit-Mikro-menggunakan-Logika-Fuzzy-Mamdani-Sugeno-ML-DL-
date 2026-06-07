import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="SPK Kredit Mikro", layout="wide")
st.title("🛡️ SPK Kelayakan Kredit Mikro (Mamdani vs Sugeno)")
st.subheader("Integrasi Hybrid: Regresi Linear (ML) + ANN (DL) + Fuzzy Logic")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("train.csv", low_memory=False)
    kolom = ['Annual_Income', 'Outstanding_Debt', 'Interest_Rate', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Credit_Score']
    df_c = df[kolom].copy().dropna()
    for col in ['Annual_Income', 'Outstanding_Debt', 'Num_of_Delayed_Payment']:
        df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
    df_c['Delay_from_due_date'] = df_c['Delay_from_due_date'].apply(lambda x: 0 if x < 0 else x)
    df_c = df_c.dropna()
    df_c['Target_Numeric'] = df_c['Credit_Score'].map({'Poor': 0, 'Standard': 1, 'Good': 2})
    
    X = df_c.drop(['Credit_Score', 'Target_Numeric'], axis=1)
    y = df_c['Target_Numeric']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Lapis 1: Regresi Linear
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    
    # Model Lapis 2: Deep Learning (ANN)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model_dl = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', max_iter=300, random_state=42)
    model_dl.fit(X_train_s, y_train)
    
    return model_lr, model_dl, scaler

with st.spinner('Melatih model ML & DL dari 100.000 data di background... (Tunggu sebentar ya)'):
    model_lr, model_dl, scaler = load_and_train()

def fungsi_segitiga(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if a < x <= b: return (x - a) / (b - a)
    if b < x < c: return (c - x) / (c - b)
    return 0.0

def fungsi_trapesium(x, a, b, c, d):
    # Logika batas yang sudah diperbaiki
    if x < a or x > d: return 0.0
    if a <= x < b: return (x - a) / (b - a) if b != a else 1.0
    if b <= x <= c: return 1.0
    if c < x <= d: return (d - x) / (d - c) if d != c else 1.0
    return 0.0

st.sidebar.header("📥 Input Data Nasabah")
inc = st.sidebar.number_input("Annual Income ($)", value=45000)
debt = st.sidebar.number_input("Outstanding Debt ($)", value=1500)
ir = st.sidebar.slider("Interest Rate (%)", 1, 35, 12)
del_day = st.sidebar.slider("Delay from Due Date (Hari)", 0, 60, 15)
num_del = st.sidebar.slider("Number of Delayed Payment (Kali)", 0, 25, 4)

# Fuzzifikasi Lapis 3
fz = {
    'inc': {'Rendah': fungsi_trapesium(inc, 0, 0, 30000, 50000), 'Sedang': fungsi_segitiga(inc, 30000, 65000, 100000), 'Tinggi': fungsi_trapesium(inc, 80000, 100000, 150000, 2000000)},
    'debt': {'Sedikit': fungsi_trapesium(debt, 0, 0, 1000, 2000), 'Sedang': fungsi_segitiga(debt, 1000, 2500, 4000), 'Banyak': fungsi_trapesium(debt, 3000, 5000, 100000, 100000)},
    'ir': {'Rendah': fungsi_trapesium(ir, 0, 0, 8, 12), 'Sedang': fungsi_segitiga(ir, 8, 15, 22), 'Tinggi': fungsi_trapesium(ir, 18, 25, 35, 100)},
    'del': {'Singkat': fungsi_trapesium(del_day, 0, 0, 10, 20), 'Sedang': fungsi_segitiga(del_day, 10, 30, 45), 'Lama': fungsi_trapesium(del_day, 35, 60, 200, 200)},
    'num': {'Jarang': fungsi_trapesium(num_del, 0, 0, 5, 8), 'Sering': fungsi_segitiga(num_del, 5, 12, 18), 'S.Sering': fungsi_trapesium(num_del, 15, 25, 100, 100)}
}

# Inferensi (15 Aturan Baku sesuai LaTeX)
r01 = min(fz['inc']['Tinggi'], fz['debt']['Sedikit'], fz['ir']['Rendah'], fz['del']['Singkat'], fz['num']['Jarang'])
r02 = min(fz['inc']['Tinggi'], fz['debt']['Sedang'], fz['ir']['Sedang'], fz['del']['Singkat'], fz['num']['Jarang'])
r03 = min(fz['inc']['Sedang'], fz['debt']['Sedikit'], fz['ir']['Rendah'], fz['del']['Singkat'], fz['num']['Jarang'])
r04 = min(fz['inc']['Tinggi'], fz['debt']['Sedikit'], fz['ir']['Tinggi'], fz['del']['Sedang'], fz['num']['Jarang'])
r05 = min(fz['inc']['Sedang'], fz['debt']['Sedang'], fz['ir']['Sedang'], fz['del']['Sedang'], fz['num']['Sering'])
r06 = min(fz['inc']['Rendah'], fz['debt']['Sedikit'], fz['ir']['Rendah'], fz['del']['Singkat'], fz['num']['Jarang'])
r07 = min(fz['inc']['Tinggi'], fz['debt']['Banyak'], fz['ir']['Sedang'], fz['del']['Singkat'], fz['num']['Jarang'])
r08 = min(fz['inc']['Sedang'], fz['debt']['Banyak'], fz['ir']['Tinggi'], fz['del']['Singkat'], fz['num']['Jarang'])
r09 = min(fz['inc']['Rendah'], fz['debt']['Sedang'], fz['ir']['Rendah'], fz['del']['Singkat'], fz['num']['Jarang'])
r10 = min(fz['inc']['Tinggi'], fz['debt']['Sedang'], fz['ir']['Tinggi'], fz['del']['Sedang'], fz['num']['Sering'])
r11 = min(fz['inc']['Rendah'], fz['debt']['Banyak'], fz['ir']['Tinggi'], fz['del']['Lama'], fz['num']['S.Sering'])
r12 = min(fz['inc']['Tinggi'], fz['debt']['Banyak'], fz['ir']['Tinggi'], fz['del']['Lama'], fz['num']['S.Sering'])
r13 = min(fz['inc']['Sedang'], fz['debt']['Banyak'], fz['ir']['Sedang'], fz['del']['Sedang'], fz['num']['Sering'])
r14 = min(fz['inc']['Rendah'], fz['debt']['Sedang'], fz['ir']['Sedang'], fz['del']['Lama'], fz['num']['Sering'])
r15 = min(fz['inc']['Sedang'], fz['debt']['Sedikit'], fz['ir']['Tinggi'], fz['del']['Lama'], fz['num']['S.Sering'])

# Aturan Tambahan (Sapu Jagat) agar web responsif
r16 = max(fz['inc']['Tinggi'], fz['num']['Jarang'])
r17 = fz['ir']['Sedang']
r18 = max(fz['debt']['Banyak'], fz['num']['S.Sering'], fz['del']['Lama'])

# Penggabungan Kekuatan Aturan (Agregasi)
u_baik = max(r01, r02, r03, r04, r16)
u_standar = max(r05, r06, r07, r08, r09, r10, r17)
u_buruk = max(r11, r12, r13, r14, r15, r18)

# Defuzzifikasi
points = np.linspace(0, 100, 50)
num, den = 0, 0
for x in points:
    mu_b = min(u_baik, fungsi_trapesium(x, 70, 85, 100, 100))
    mu_s = min(u_standar, fungsi_segitiga(x, 30, 50, 75))
    mu_br = min(u_buruk, fungsi_trapesium(x, 0, 0, 25, 45))
    mu_max = max(mu_b, mu_s, mu_br)
    num += x * mu_max
    den += mu_max
skor_mamdani = num / den if den != 0 else 50
skor_sugeno = ((u_baik * 90) + (u_standar * 50) + (u_buruk * 15)) / (u_baik + u_standar + u_buruk) if (u_baik + u_standar + u_buruk) != 0 else 50

# Prediksi Model ML & DL
fitur_df = pd.DataFrame([[inc, debt, ir, del_day, num_del]], columns=['Annual_Income', 'Outstanding_Debt', 'Interest_Rate', 'Delay_from_due_date', 'Num_of_Delayed_Payment'])
pred_ml = model_lr.predict(fitur_df)[0]
pred_dl_prob = model_dl.predict_proba(scaler.transform(fitur_df))[0][0] 

# UI Output Antarmuka
col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 ML Prediction", f"{pred_ml:.2f}")
col2.metric("🧠 ANN Default Risk", f"{pred_dl_prob * 100:.1f}%")
col3.metric("📐 Mamdani Score", f"{skor_mamdani:.2f}")
col4.metric("⚡ Sugeno Score", f"{skor_sugeno:.2f}")

st.markdown("---")
if skor_sugeno >= 70: 
    st.success("🟢 **DISETUJUI (BAIK)**")
elif skor_sugeno >= 40: 
    st.warning("🟡 **PERTIMBANGAN (STANDAR)**")
else: 
    st.error("🔴 **DITOLAK (BURUK)**")