import joblib
import pandas as pd
import requests

# 1. Buka daftar kolom model
model_columns = joblib.load('model_columns.pkl')

# 2. Masukkan data input sederhana dalam bentuk teks/angka biasa
input_biasa = pd.DataFrame([
    {
        'model_year': 2021,
        'milage': 25000,
        'fuel_type': 'Gasoline',
        'transmission': 'Automatic',
    }
])

# 3. Otomatis samakan kolom dengan model (161 kolom)
input_encoded = (
    pd.get_dummies(input_biasa)
    .reindex(columns=model_columns, fill_value=0)
    .values.tolist()
)

# 4. Kirim ke Flask API
payload = {'data': input_encoded}
response = requests.post('http://127.0.0.1:5000/predict', json=payload)

# 5. Tampilkan hasil prediksi
print('=== HASIL PREDIKSI HARGA LAND ROVER ===')
print(response.json())