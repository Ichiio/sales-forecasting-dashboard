import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from io import BytesIO


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
st.markdown("Aplikasi prediksi penjualan menggunakan metode **WMA (Sum-of-Digits)** dan **Holt's Exponential Smoothing**.")

# --- LOGIKA DATA ---
if input_option == "Upload File" and uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        # Bersihkan baris yang benar-benar kosong
        df = df.dropna(how='all')
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        df = pd.DataFrame()
else:
    # DATA TEMPLATE
    initial_data = pd.DataFrame({
        "Tahun": [2022]*12,
        "Bulan": ["Januari", "Februari", "Maret", "April", "Mei", "Juni", 
                  "Juli", "Agustus", "September", "Oktober", "November", "Desember"],
        "Penjualan": [1642.0, 1689.0, 1698.0, 1740.0, 2218.0, 1396.0, 
                      2167.0, 1594.0, 1204.0, 1698.0, 1159.0, 1861.0]
    })
    st.subheader("📝 Input Data Historis")
    # PERBAIKAN: Format %d agar Tahun tidak berkoma
    df = st.data_editor(
        initial_data, 
        num_rows="dynamic", 
        use_container_width=True,
        column_config={
            "Tahun": st.column_config.NumberColumn("Tahun", format="%d"),
            "Penjualan": st.column_config.NumberColumn("Penjualan", format="%.2f")
        }
    )

# --- PROSES VALIDASI WAKTU & CLEANING ---
if not df.empty:
    # Hapus baris yang Bulan atau Penjualannya kosong untuk mencegah KeyError: nan
    df = df.dropna(subset=['Bulan', 'Penjualan'])
    
    month_map = {
        "Januari": 1, "Februari": 2, "Maret": 3, "April": 4, "Mei": 5, "Juni": 6,
        "Juli": 7, "Agustus": 8, "September": 9, "Oktober": 10, "November": 11, "Desember": 12
    }
    
    # Map bulan ke angka
    df['Month_Num'] = df['Bulan'].map(month_map)
    
    # Hapus jika ada nama bulan yang tidak valid (tidak ada di month_map)
    df = df.dropna(subset=['Month_Num'])
    df['Month_Num'] = df['Month_Num'].astype(int)
    
    # Sorting agar data tidak bertabrakan antar tahun
    df = df.sort_values(['Tahun', 'Month_Num']).reset_index(drop=True)
    df['Periode'] = df['Bulan'] + " " + df['Tahun'].astype(str)

# Eksekusi Perhitungan jika data tersedia
if not df.empty and df['Penjualan'].notnull().any():
    st.write("---")
    col_p1, col_p2 = st.columns([1.5, 1])
    
    with col_p1:
        st.subheader("⚙️ Konfigurasi Peramalan")
        method = st.selectbox("Pilih Metode:", ["Weighted Moving Average (WMA)", "Double Exponential Smoothing (Holt)"])
    
    with col_p2:
        st.subheader("📅 Jangka Waktu")
        horizon = st.number_input("Jumlah bulan ke depan:", min_value=1, max_value=24, value=3)

    forecast_history = []
    forecast_future = []
    
    with st.expander("Klik untuk atur Parameter Metode", expanded=True):
        if method == "Weighted Moving Average (WMA)":
            n = st.slider("Periode Moving Average (n):", 2, len(df), min(6, len(df)))
            
            # Bobot Sum-of-Digits: data terbaru bobot n, data tertua bobot 1
            auto_weights = [float(i) for i in range(1, n + 1)]
            total_w = sum(auto_weights)
            
            st.info(f"Pembobotan Aktif. Pembagi (Total Bobot): {int(total_w)}")

            if st.button("🚀 HITUNG SEKARANG"):
                data_list = df['Penjualan'].astype(float).tolist()
                history = [None] * len(data_list)
                
                # Hitung History
                for i in range(n, len(data_list)):
                    window = data_list[i-n:i]
                    history[i] = sum(window[j] * auto_weights[j] for j in range(n)) / total_w
                
                # Hitung Future
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

    # --- TAMPILAN HASIL ---
    if forecast_history or forecast_future:
        res_df = df[['Periode', 'Penjualan']].copy()
        res_df['Peramalan'] = forecast_history

        # Logika label bulan masa depan
        last_m_num = int(df['Month_Num'].iloc[-1])
        last_year = int(df['Tahun'].iloc[-1])
        inv_month_map = {v: k for k, v in month_map.items()}
        
        future_rows = []
        for i in range(horizon):
            last_m_num += 1
            if last_m_num > 12:
                last_m_num = 1
                last_year += 1
            future_rows.append({
                "Periode": f"{inv_month_map[last_m_num]} {last_year}",
                "Penjualan": np.nan,
                "Peramalan": forecast_future[i]
            })
        
        display_df = pd.concat([res_df, pd.DataFrame(future_rows)], ignore_index=True)
        display_df['Error'] = display_df['Penjualan'] - display_df['Peramalan']

        tab1, tab2 = st.tabs(["📊 Visualisasi Grafik", "📋 Tabel Data"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=display_df['Periode'], y=display_df['Penjualan'], 
                                     name="Data Aktual", mode='lines+markers', line=dict(color='#1e3d59', width=3)))
            fig.add_trace(go.Scatter(x=display_df['Periode'], y=display_df['Peramalan'], 
                                     name="Peramalan", mode='lines+markers', line=dict(color='#ff6e40', dash='dash')))
            fig.update_layout(hovermode="x unified", template="plotly_white", height=500)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.dataframe(display_df.style.highlight_null(color="#f8f9fa").format(precision=4), use_container_width=True)
            
        # =====================================================
        # DOWNLOAD HASIL
        # =====================================================

        st.write("---")
        st.subheader("📥 Download Hasil")

        # Hanya jika evaluasi tersedia
        valid_export = display_df.dropna(subset=['Penjualan', 'Peramalan'])

        if not valid_export.empty:

            # -----------------------------
            # Hitung Evaluasi
            # -----------------------------

            mae = mean_absolute_error(
                valid_export['Penjualan'],
                valid_export['Peramalan']
            )

            mse = mean_squared_error(
                valid_export['Penjualan'],
                valid_export['Peramalan']
            )

            mape = np.mean(
                np.abs(
                    (
                        valid_export['Penjualan']
                        - valid_export['Peramalan']
                    )
                    /
                    valid_export['Penjualan']
                )
            ) * 100

        # -----------------------------
        # Interpretasi
        # -----------------------------

        if mape < 10:
            kategori = "Sangat Tinggi"
            penjelasan = (
                "Model memiliki tingkat akurasi sangat tinggi "
                "dan sangat layak digunakan untuk peramalan."
            )

        elif mape < 20:
            kategori = "Baik"
            penjelasan = (
                "Model memiliki tingkat akurasi baik "
                "dan cukup andal digunakan dalam pengambilan keputusan."
            )

        elif mape < 50:
            kategori = "Cukup"
            penjelasan = (
                "Model masih dapat digunakan namun perlu "
                "evaluasi lebih lanjut untuk meningkatkan akurasi."
            )

        else:
            kategori = "Buruk"
            penjelasan = (
                "Model kurang akurat sehingga disarankan "
                "menggunakan parameter atau metode lain."
            )

        kesimpulan = (
            f"Berdasarkan hasil evaluasi, metode {method} "
            f"menghasilkan nilai MAPE sebesar {mape:.2f}% "
            f"yang termasuk kategori '{kategori}'. "
            f"{penjelasan}"
        )

        # -----------------------------
        # Data Export
        # -----------------------------

        export_df = display_df.copy()

        export_df['Abs Error'] = (
            export_df['Error']
            .abs()
            .round(4)
        )

        export_df['APE (%)'] = (
            np.abs(
                (
                    export_df['Penjualan']
                    - export_df['Peramalan']
                )
                /
                export_df['Penjualan']
            ) * 100
        ).round(2)

        # Rapikan angka

        for col in [
            'Penjualan',
            'Peramalan',
            'Error',
            'Abs Error'
        ]:
            if col in export_df.columns:
                export_df[col] = (
                    export_df[col]
                    .round(4)
                )

        # -----------------------------
        # Ringkasan Evaluasi
        # -----------------------------

        summary_df = pd.DataFrame({
            "Metrik": [
                "MAE",
                "MSE",
                "MAPE (%)",
                "Kategori Akurasi",
                "Interpretasi",
                "Kesimpulan"
            ],
            "Nilai": [
                round(mae, 4),
                round(mse, 4),
                round(mape, 2),
                kategori,
                penjelasan,
                kesimpulan
            ]
        })

        # =================================================
        # EXCEL
        # =================================================

        excel_buffer = BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:

            export_df.to_excel(
                writer,
                sheet_name="Forecast",
                index=False
            )

            summary_df.to_excel(
                writer,
                sheet_name="Evaluasi",
                index=False
            )

        st.download_button(
            label="📊 Download Excel",
            data=excel_buffer.getvalue(),
            file_name="hasil_peramalan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # =================================================
        # PNG
        # =================================================

        try:

            img_bytes = fig.to_image(
                format="png",
                width=1400,
                height=700
            )

            st.download_button(
                label="🖼️ Download Grafik PNG",
                data=img_bytes,
                file_name="grafik_peramalan.png",
                mime="image/png"
            )

        except Exception:

            st.info(
                "Untuk mengaktifkan export PNG install terlebih dahulu:\n\n"
                "pip install kaleido"
            )
        # --- EVALUASI ---
        st.write("---")
        st.subheader("🎯 Evaluasi Akurasi")
        valid_eval = display_df.dropna(subset=['Penjualan', 'Peramalan'])
        if not valid_eval.empty:
            mae = mean_absolute_error(valid_eval['Penjualan'], valid_eval['Peramalan'])
            mse = mean_squared_error(valid_eval['Penjualan'], valid_eval['Peramalan'])
            mape = np.mean(np.abs((valid_eval['Penjualan'] - valid_eval['Peramalan']) / valid_eval['Penjualan'])) * 100
            
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{mae:.4f}")
            m2.metric("MSE", f"{mse:.4f}")
            m3.metric("MAPE", f"{mape:.2f}%")

            if mape < 10: st.success("Akurasi: Sangat Tinggi (MAPE < 10%)")
            elif mape < 20: st.info("Akurasi: Baik (MAPE 10-20%)")
            else: st.warning("Akurasi: Rendah, pertimbangkan mengubah parameter.")
        else:
            st.warning("Data historis terlalu sedikit untuk evaluasi akurasi.")

else:
    st.warning("Silakan lengkapi data penjualan pada tabel atau upload file untuk memulai analisis.")

st.write("---")
st.caption("© 2026 Dashboard Peramalan Penjualan | ICHIIO")