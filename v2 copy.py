import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import Holt
from sklearn.metrics import mean_squared_error

# ----------------------------
# 1. Import Data dari Excel
# ----------------------------
try:
    # Ganti 'data_saham.xlsx' dengan nama file Anda
    # Misal kolom berisi tanggal dan harga penutupan: 'Tanggal', 'Harga_TLKM'
    df = pd.read_excel('Book1lkk.xlsx')
except FileNotFoundError:
    print("File 'Book1lkk.xlsx' tidak ditemukan. Membuat data dummy untuk demonstrasi.")
    # Membuat data dummy jika file tidak ada
    data = {'Tanggal': pd.to_datetime(pd.date_range(start='1/1/2018', periods=24)),
            'Harga_TLKM': 24 + np.arange(24) * 1.5 + np.random.randn(24) * 5}
    df = pd.DataFrame(data)


# Pastikan data terurut berdasarkan tanggal
# Mengasumsikan kolom pertama adalah tanggal dan kedua adalah harga
df = df.sort_values(by=df.columns[0])

# Ambil hanya kolom harga
harga = df.iloc[:, 1].values

# ----------------------------
# 2. Inisialisasi pencarian parameter
# ----------------------------
best_alpha = None
best_beta = None
best_rmse = float("inf")

# Menentukan rentang nilai alpha dan beta yang akan diuji
alpha_values = np.arange(0.1, 1.0, 0.1)
beta_values = np.arange(0.1, 1.0, 0.1)

# ----------------------------
# 3. Grid Search untuk mencari alpha dan beta terbaik
# ----------------------------
print("--- Memulai Proses Grid Search ---")
print("Mencari kombinasi Alpha dan Beta dengan RMSE terkecil...")
print("-" * 40)

for alpha in alpha_values:
    for beta in beta_values:
        # Fitting model Holt dengan kombinasi alpha dan beta saat ini
        model = Holt(harga).fit(
            smoothing_level=alpha,
            smoothing_slope=beta,
            optimized=False  # Penting: set False agar alpha & beta dari loop digunakan
        )
        
        # Ambil nilai hasil fitting dari model
        fitted_values = model.fittedvalues
        
        # Hitung RMSE antara data aktual dan data hasil fitting
        rmse = np.sqrt(mean_squared_error(harga, fitted_values))
        
        # --- [PERUBAHAN UTAMA] ---
        # Tampilkan hasil dari setiap percobaan yang dilakukan
        print(f"Mencoba -> Alpha: {alpha:.1f}, Beta: {beta:.1f} | Menghasilkan RMSE: {rmse:.4f}")
        
        # Cek apakah RMSE saat ini lebih kecil dari RMSE terbaik sejauh ini
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
            best_beta = beta

# ----------------------------
# 4. Hasil parameter optimal
# ----------------------------
print("-" * 40)
print("✅ Proses pencarian selesai.")
print("\n--- HASIL PARAMETER OPTIMAL ---")
print(f"Alpha (α) Terbaik = {best_alpha:.1f}")
print(f"Beta (β) Terbaik  = {best_beta:.1f}")
print(f"RMSE Terkecil      = {best_rmse:.4f}")