import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from io import BytesIO

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Forecasting Dashboard Multi-Produk",
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
    input_option = st.radio("Pilih Metode Input:", ("Input Manual Dynamic", "Upload File"))
    
    if input_option == "Upload File":
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
        st.caption("Pastikan file memiliki kolom: **Tahun**, **Bulan**, **Produk**, **Penjualan**")
    else:
        st.info("Kamu bisa bebas menambah baris & nama produk pada tabel di bawah.")

# --- HEADER UTAMA ---
st.title("📈 Sistem Peramalan Penjualan Multi-Produk")
st.markdown("Aplikasi prediksi penjualan multi-produk dengan visualisasi perbandingan Aktual vs Ramalan.")

# --- LOGIKA DATA ---
if input_option == "Upload File" and 'uploaded_file' in locals() and uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        df_raw = df_raw.dropna(how='all')
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        df_raw = pd.DataFrame()
else:
    # TEMPLATE DATA DENGAN PRODUK SESUAI SKETSA
    produk_default = ["Susu", "Crispy", "Pia"]
    bulan_default = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", 
                     "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
    
    sample_rows = []
    np.random.seed(42)
    for p in produk_default:
        base_val = 1200 if p == "Susu" else (1800 if p == "Crispy" else 2200)
        for b in bulan_default:
            val = round(base_val + np.random.randint(-200, 300), 2)
            sample_rows.append({"Tahun": 2025, "Bulan": b, "Produk": p, "Penjualan": val})
            
    initial_data = pd.DataFrame(sample_rows)
    
    st.subheader("📝 Input Data Historis (Bebas Tambah Produk & Baris)")
    df_raw = st.data_editor(
        initial_data, 
        num_rows="dynamic", 
        use_container_width=True,
        column_config={
            "Tahun": st.column_config.NumberColumn("Tahun", format="%d"),
            "Produk": st.column_config.TextColumn("Nama Produk / Varian", help="Ketik nama produk secara bebas"),
            "Penjualan": st.column_config.NumberColumn("Penjualan", format="%.2f")
        }
    )

# --- PROSES VALIDASI WAKTU & CLEANING ---
if not df_raw.empty:
    required_cols = ['Tahun', 'Bulan', 'Produk', 'Penjualan']
    if all(col in df_raw.columns for col in required_cols):
        df_clean = df_raw.dropna(subset=['Bulan', 'Penjualan', 'Produk']).copy()
        
        month_map = {
            "Januari": 1, "Februari": 2, "Maret": 3, "April": 4, "Mei": 5, "Juni": 6,
            "Juli": 7, "Agustus": 8, "September": 9, "Oktober": 10, "November": 11, "Desember": 12
        }
        
        df_clean['Month_Num'] = df_clean['Bulan'].map(month_map)
        df_clean = df_clean.dropna(subset=['Month_Num'])
        df_clean['Month_Num'] = df_clean['Month_Num'].astype(int)
        df_clean['Periode'] = df_clean['Bulan'] + " " + df_clean['Tahun'].astype(str)
        df_clean = df_clean.sort_values(['Tahun', 'Month_Num']).reset_index(drop=True)
        
        list_produk = df_clean['Produk'].unique().tolist()
        
        st.write("---")
        st.subheader("⚙️ Konfigurasi Peramalan")
        
        col_p1, col_p2 = st.columns([1.5, 1])
        with col_p1:
            method = st.selectbox("Pilih Metode Peramalan:", ["Weighted Moving Average (WMA)", "Double Exponential Smoothing (Holt)"])
        with col_p2:
            horizon = st.number_input("Jumlah bulan ke depan:", min_value=1, max_value=24, value=3)

        # Parameter per metode
        if method == "Weighted Moving Average (WMA)":
            n = st.slider("Periode Moving Average (n):", 2, 12, 4)
            auto_weights = [float(i) for i in range(1, n + 1)]
            total_w = sum(auto_weights)
        else:
            c1, c2 = st.columns(2)
            alpha = c1.slider("Level Smoothing (Alpha):", 0.0, 1.0, 0.3)
            beta = c2.slider("Trend Smoothing (Beta):", 0.0, 1.0, 0.2)

        calc_trigger = st.button("🚀 HITUNG & TAMPILKAN SEMUA PRODUK")
        
        # --- PROSES PERAMALAN UNTUK TIAP PRODUK ---
        if calc_trigger:
            all_results = []
            eval_metrics = []
            
            inv_month_map = {v: k for k, v in month_map.items()}

            for prod in list_produk:
                df_p = df_clean[df_clean['Produk'] == prod].copy().reset_index(drop=True)
                
                if len(df_p) < 2:
                    continue
                
                forecast_history = []
                forecast_future = []

                if method == "Weighted Moving Average (WMA)":
                    data_list = df_p['Penjualan'].astype(float).tolist()
                    curr_n = min(n, len(data_list))
                    curr_weights = [float(i) for i in range(1, curr_n + 1)]
                    curr_total_w = sum(curr_weights)

                    history = [None] * len(data_list)
                    for i in range(curr_n, len(data_list)):
                        window = data_list[i-curr_n:i]
                        history[i] = sum(window[j] * curr_weights[j] for j in range(curr_n)) / curr_total_w

                    future = []
                    temp_data = data_list.copy()
                    for _ in range(horizon):
                        window = temp_data[-curr_n:]
                        next_val = sum(window[j] * curr_weights[j] for j in range(curr_n)) / curr_total_w
                        future.append(next_val)
                        temp_data.append(next_val)

                    forecast_history, forecast_future = history, future

                else: # Holt's Method
                    y = df_p['Penjualan'].astype(float).tolist()
                    l, b = [y[0]], [y[1] - y[0]]
                    history = [None, y[0]]
                    for i in range(1, len(y)):
                        l_curr = alpha * y[i] + (1 - alpha) * (l[-1] + b[-1])
                        b_curr = beta * (l_curr - l[-1]) + (1 - beta) * b[-1]
                        l.append(l_curr); b.append(b_curr)
                        if i < len(y)-1: history.append(l[-1] + b[-1])
                    future = [(l[-1] + (h * b[-1])) for h in range(1, horizon + 1)]
                    forecast_history, forecast_future = history, future

                # Susun dataframe historis + future
                res_p = df_p[['Periode', 'Produk', 'Penjualan']].copy()
                res_p['Peramalan'] = forecast_history

                last_m_num = int(df_p['Month_Num'].iloc[-1])
                last_year = int(df_p['Tahun'].iloc[-1])

                future_rows = []
                for i in range(horizon):
                    last_m_num += 1
                    if last_m_num > 12:
                        last_m_num = 1
                        last_year += 1
                    future_rows.append({
                        "Periode": f"{inv_month_map[last_m_num]} {last_year}",
                        "Produk": prod,
                        "Penjualan": np.nan,
                        "Peramalan": forecast_future[i]
                    })

                full_p = pd.concat([res_p, pd.DataFrame(future_rows)], ignore_index=True)
                all_results.append(full_p)

                # Hitung evaluasi per produk
                valid_eval = full_p.dropna(subset=['Penjualan', 'Peramalan'])
                if not valid_eval.empty:
                    mae = mean_absolute_error(valid_eval['Penjualan'], valid_eval['Peramalan'])
                    mse = mean_squared_error(valid_eval['Penjualan'], valid_eval['Peramalan'])
                    mape = np.mean(np.abs((valid_eval['Penjualan'] - valid_eval['Peramalan']) / valid_eval['Penjualan'])) * 100
                    eval_metrics.append({"Produk": prod, "MAE": mae, "MSE": mse, "MAPE (%)": mape})

            if all_results:
                final_df = pd.concat(all_results, ignore_index=True)
                
                # --- VISUALISASI ---
                st.write("---")
                st.subheader("📊 Visualisasi Hasil Peramalan Semua Produk")
                
                tab_bar, tab_line = st.tabs(["📊 Perbandingan Produk (Diagram Batang)", "📈 Grafik Tren (Line Chart)"])
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

                # 1. DIAGRAM BATANG (SESUAI SKETSA: Produk di Sumbu X -> Bar Aktual vs Ramalan)
                with tab_bar:
                    summary_product = final_df.groupby('Produk').agg({
                        'Penjualan': 'sum',
                        'Peramalan': 'sum'
                    }).reset_index()

                    fig_bar = go.Figure()

                    fig_bar.add_trace(go.Bar(
                        x=summary_product['Produk'],
                        y=summary_product['Penjualan'],
                        name='Aktual',
                        marker_color='#1f77b4',
                        text=summary_product['Penjualan'].apply(lambda x: f"{x:,.0f}"),
                        textposition='auto'
                    ))

                    fig_bar.add_trace(go.Bar(
                        x=summary_product['Produk'],
                        y=summary_product['Peramalan'],
                        name='Ramalan',
                        marker_color='#ff7f0e',
                        text=summary_product['Peramalan'].apply(lambda x: f"{x:,.0f}"),
                        textposition='auto'
                    ))

                    fig_bar.update_layout(
                        barmode='group',
                        title="Perbandingan Total Penjualan Aktual vs Ramalan Per Produk",
                        xaxis_title="Produk",
                        yaxis_title="Total Penjualan",
                        template="plotly_white",
                        height=550,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                # 2. LINE CHART (OPSIONAL / TREN BULANAN)
                with tab_line:
                    fig_line = go.Figure()
                    for idx, prod in enumerate(list_produk):
                        prod_df = final_df[final_df['Produk'] == prod]
                        color = colors[idx % len(colors)]
                        
                        fig_line.add_trace(go.Scatter(
                            x=prod_df['Periode'], y=prod_df['Penjualan'],
                            name=f"{prod} (Aktual)", mode='lines+markers', line=dict(color=color, width=2.5)
                        ))
                        fig_line.add_trace(go.Scatter(
                            x=prod_df['Periode'], y=prod_df['Peramalan'],
                            name=f"{prod} (Ramalan)", mode='lines+markers', line=dict(color=color, dash='dash', width=2)
                        ))

                    fig_line.update_layout(
                        title="Perbandingan Tren Penjualan & Peramalan Seluruh Produk",
                        hovermode="x unified", template="plotly_white", height=550,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_line, use_container_width=True)

                # --- TABEL HASIL & EVALUASI AKURASI ---
                tab_table, tab_eval = st.tabs(["📋 Tabel Data Peramalan", "🎯 Evaluasi Akurasi Per Produk"])
                
                with tab_table:
                    st.dataframe(final_df.style.highlight_null(color="#f8f9fa").format(precision=2), use_container_width=True)
                    
                with tab_eval:
                    if eval_metrics:
                        eval_df = pd.DataFrame(eval_metrics)
                        st.dataframe(eval_df.style.format({"MAE": "{:.4f}", "MSE": "{:.4f}", "MAPE (%)": "{:.2f}%"}), use_container_width=True)

                # --- DOWNLOAD EXCEL GABUNGAN ---
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    final_df.to_excel(writer, sheet_name="Hasil Peramalan", index=False)
                    if eval_metrics:
                        pd.DataFrame(eval_metrics).to_excel(writer, sheet_name="Evaluasi Akurasi", index=False)

                st.download_button(
                    label="📊 Download Excel Semua Produk",
                    data=excel_buffer.getvalue(),
                    file_name="hasil_peramalan_semua_produk.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    else:
        st.error(f"Format kolom kurang sesuai. Pastikan ada: {', '.join(required_cols)}")
else:
    st.warning("Silakan lengkapi data penjualan pada tabel terlebih dahulu.")

st.write("---")
st.caption("© 2026 Dashboard Peramalan Penjualan Multi-Produk Dynamic | ICHIIO")