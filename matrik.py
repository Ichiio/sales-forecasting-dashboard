import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Membaca file excel
nama_file = 'data_peramalan.xlsx'
df = pd.read_excel(nama_file)

# 2. Ekstraksi data dari kolom
y_true = df['Aktual'].values
y_arima = df['ARIMA'].values
y_holt = df['HOLT'].values

# Fungsi pembantu untuk menghitung semua metrik sekaligus
def hitung_metrik(aktual, prediksi):
    mae = mean_absolute_error(aktual, prediksi)
    mse = mean_squared_error(aktual, prediksi)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# 3. Hitung metrik untuk kedua model
mae_arima, mse_arima, rmse_arima = hitung_metrik(y_true, y_arima)
mae_holt, mse_holt, rmse_holt = hitung_metrik(y_true, y_holt)

# 4. Tampilkan dalam bentuk tabel perbandingan
hasil = pd.DataFrame({
    'Metrik': ['MAE', 'MSE', 'RMSE'],
    'ARIMA': [mae_arima, mse_arima, rmse_arima],
    'Holt': [mae_holt, mse_holt, rmse_holt]
})

print("--- Perbandingan Performa Model ---")
print(hasil.to_string(index=False))

# 5. Kesimpulan sederhana
print("\nKesimpulan:")
if rmse_arima < rmse_holt:
    print("Model ARIMA lebih akurat karena memiliki RMSE lebih kecil.")
else:
    print("Model Holt lebih akurat karena memiliki RMSE lebih kecil.")