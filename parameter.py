import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import Holt
from sklearn.metrics import mean_squared_error

# ----------------------------
# 1. Import Data dari Excel
# ----------------------------
# Ganti 'data_saham.xlsx' dengan nama file Anda
# Misal kolom berisi tanggal dan harga penutupan: 'Tanggal', 'Harga_TLKM'
df = pd.read_excel('Book1lkk.xlsx')

# Pastikan data terurut berdasarkan tanggal
df = df.sort_values(by=df.columns[0])

# Ambil hanya kolom harga (misalnya kolom kedua)
harga = df.iloc[:, 1].values

# ----------------------------
# 2. Inisialisasi pencarian parameter
# ----------------------------
best_alpha = None
best_beta = None
best_rmse = float("inf")

alpha_values = np.arange(0.1, 1.0, 0.1)
beta_values = np.arange(0.1, 1.0, 0.1)

# ----------------------------
# 3. Grid Search untuk mencari alpha dan beta terbaik
# ----------------------------
for alpha in alpha_values:
    for beta in beta_values:
        model = Holt(harga).fit(smoothing_level=alpha,
                                smoothing_slope=beta,
                                optimized=False)
        fitted_values = model.fittedvalues
        rmse = np.sqrt(mean_squared_error(harga, fitted_values))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
            best_beta = beta

# ----------------------------
# 4. Hasil parameter optimal
# ----------------------------
print(f"Parameter optimal Holt:")
print(f"Alpha = {best_alpha}")
print(f"Beta  = {best_beta}")
print(f"RMSE (fitted) = {best_rmse:.2f}")
