import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- SIDEBAR: INPUT DATA ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1622/1622857.png", width=80)
    st.title("Panel Kontrol")
    st.write("---")
    
    st.subheader("1. Sumber Data")
    input_option = st.radio("Pilih Metode Input:", ("Upload File", "Input Manual"))
    
    if input_option == "Upload File":
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    else:
        st.info("Input data langsung pada tabel di area utama.")

# --- HEADER UTAMA ---
st.title("📈 Sistem Peramalan Penjualan")
st.markdown("Aplikasi cerdas untuk memprediksi penjualan masa depan dengan metode WMA dan Holt's.")

# --- LOGIKA DATA ---
df = pd.DataFrame(columns=["Bulan", "Penjualan"])

if input_option == "Upload File" and uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error membaca file: {e}")
else:
    # Data Default/Template
    initial_data = pd.DataFrame({
        "Bulan": ["Januari", "Februari", "Maret", "April"],
        "Penjualan": [120.0, 135.0, 150.0, 145.0]
    })
    st.subheader("📝 Input Data Historis")
    df = st.data_editor(initial_data, num_rows="dynamic", use_container_width=True)

# Pastikan data siap diproses
if not df.empty and df['Penjualan'].notnull().any():
    
    # --- BAGIAN 2: PARAMETER METODE ---
    st.write("---")
    col_p1, col_p2 = st.columns([1.5, 1])
    
    with col_p1:
        st.subheader("⚙️ Konfigurasi Peramalan")
        method = st.selectbox("Pilih Metode:", 
                             ["Weighted Moving Average (WMA)", "Double Exponential Smoothing (Holt)"])
    
    with col_p2:
        st.subheader("📅 Jangka Waktu")
        horizon = st.number_input("Jumlah bulan ke depan:", min_value=1, max_value=24, value=3)

    # Inisialisasi variabel hasil
    forecast_history = []
    forecast_future = []
    
    # Container Parameter Spesifik
    with st.expander("Klik untuk atur Parameter Metode", expanded=True):
        if method == "Weighted Moving Average (WMA)":
            n = st.slider("Periode Moving Average (n):", 2, len(df), min(6, len(df)))

            # Menghitung bobot otomatis berdasarkan Sum-of-Digits (seperti di gambar)
            # Urutan: [1, 2, 3, ..., n]
            auto_weights = [float(i) for i in range(1, n + 1)]
            total_w = sum(auto_weights)
            
            st.info(f"Menggunakan pembobotan otomatis (n={n}). Pembagi: {int(total_w)}")
            
            # Menampilkan bobot agar user tahu apa yang terjadi (opsional)
            cols = st.columns(n)
            for i in range(n):
                cols[i].metric(f"Bbt t-{n-i}", int(auto_weights[i]))

            if st.button("🚀 HITUNG SEKARANG"):
                data_list = df['Penjualan'].astype(float).tolist()
                history = [None] * len(data_list)
                
                # Hitung History
                for i in range(n, len(data_list)):
                    window = data_list[i-n:i]
                    # window[0] adalah data tertua (bobot 1), window[-1] data terbaru (bobot n)
                    weighted_sum = sum(window[j] * auto_weights[j] for j in range(n))
                    history[i] = round(weighted_sum / total_w)
                
                # Hitung Masa Depan (Forecasting)
                future = []
                temp_data = data_list.copy()
                for _ in range(horizon):
                    window = temp_data[-n:]
                    next_val = sum(window[j] * auto_weights[j] for j in range(n)) / total_w
                    future.append(next_val)
                    temp_data.append(next_val)
            
                forecast_history, forecast_future = history, future
            
        else: # Holt's Method
            c1, c2 = st.columns(2)
            alpha = c1.slider("Level Smoothing (Alpha):", 0.0, 1.0, 0.3)
            beta = c2.slider("Trend Smoothing (Beta):", 0.0, 1.0, 0.2)
            
            if st.button("🚀 HITUNG SEKARANG"):
                y = df['Penjualan'].astype(float).tolist()
                l, b = [y[0]], [y[1] - y[0]]
                history = [None, y[0]]
                for i in range(1, len(y)):
                    l_curr = alpha * y[i] + (1 - alpha) * (l[-1] + b[-1])
                    b_curr = beta * (l_curr - l[-1]) + (1 - beta) * b[-1]
                    l.append(l_curr); b.append(b_curr)
                    if i < len(y)-1: history.append(l[-1] + b[-1])
                future = [(l[-1] + (h * b[-1])) for h in range(1, horizon + 1)]
                forecast_history, forecast_future = history, future

    # --- BAGIAN 3: HASIL ---
    if forecast_history or forecast_future:
        # Pengolahan Data untuk Visualisasi
        res_df = df.copy()
        res_df['Peramalan'] = forecast_history
        
        last_month = len(df)
        future_data = []
        for i, val in enumerate(forecast_future):
            future_data.append({"Bulan": f"Mendatang +{i+1}", "Penjualan": np.nan, "Peramalan": val})
        
        display_df = pd.concat([res_df, pd.DataFrame(future_data)], ignore_index=True)
        display_df['Error'] = display_df['Penjualan'] - display_df['Peramalan']

        # Layout Hasil: Grafik & Tabel
        tab1, tab2 = st.tabs(["📊 Visualisasi Grafik", "📋 Tabel Data"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=display_df['Bulan'], y=display_df['Penjualan'], 
                                     name="Data Aktual", mode='lines+markers', line=dict(color='#1e3d59', width=3)))
            fig.add_trace(go.Scatter(x=display_df['Bulan'], y=display_df['Peramalan'], 
                                     name="Peramalan", mode='lines+markers', line=dict(color='#ff6e40', dash='dash')))
            fig.update_layout(hovermode="x unified", template="plotly_white", height=500,
                              margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.dataframe(display_df.style.highlight_null(color="#f8f9fa").format(precision=2), use_container_width=True)

        # --- BAGIAN 4: EVALUASI KINERJA ---
        st.write("---")
        st.subheader("🎯 Evaluasi Akurasi (Backtesting)")
        
        valid_eval = display_df.dropna(subset=['Penjualan', 'Peramalan'])
        if not valid_eval.empty:
            mae = mean_absolute_error(valid_eval['Penjualan'], valid_eval['Peramalan'])
            mse = mean_squared_error(valid_eval['Penjualan'], valid_eval['Peramalan'])
            mape = np.mean(np.abs((valid_eval['Penjualan'] - valid_eval['Peramalan']) / valid_eval['Penjualan'])) * 100

            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error")
            m2.metric("MSE", f"{mse:.2f}", help="Mean Squared Error")
            m3.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")
            
            # Tips Interpretasi
            if mape < 10: st.success("Akurasi: Sangat Tinggi (MAPE < 10%)")
            elif mape < 20: st.info("Akurasi: Baik (MAPE 10-20%)")
            else: st.warning("Akurasi: Rendah, pertimbangkan mengubah parameter.")
        else:
            st.warning("Data historis terlalu sedikit untuk evaluasi akurasi.")

else:
    st.warning("Silakan lengkapi data penjualan pada tabel atau upload file untuk memulai analisis.")

# --- FOOTER ---
st.write("---")
st.caption("© 2026 Dashboard Peramalan Penjualan | ICHIIO")