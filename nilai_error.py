from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Contoh data: ganti ini dengan data Anda sendiri
# Misalnya data dari hasil peramalan ARIMA
nilai_aktual = np.array([3970, 3985, 4002, 4011, 4028, 4042, 4056, 4065, 4074, 4082, 4090, 4105])
prediksi_arima = np.array([3943.21, 3936.83, 3930.74, 3924.78, 3918.63, 3912.56, 3906.48, 3900.53, 3894.31, 3888.16, 3882.09, 3875.89])

# Hitung MAE, MSE, dan RMSE
mae = mean_absolute_error(nilai_aktual, prediksi_arima)
mse = mean_squared_error(nilai_aktual, prediksi_arima)
rmse = np.sqrt(mse)

# Tampilkan hasil
print(f"MAE  = {mae:.2f}")
print(f"MSE  = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")
