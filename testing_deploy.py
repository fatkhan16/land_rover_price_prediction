from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model dan daftar fitur yang telah disimpan
model = joblib.load('land_rover_model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Menerima input data format JSON dari pengguna
    data = request.json['data']
    
    # Mengonversi JSON menjadi DataFrame dan menyelaraskan kolom sesuai data training
    df_input = pd.DataFrame(data)
    df_encoded = pd.get_dummies(df_input).reindex(columns=model_columns, fill_value=0)
    
    # Melakukan prediksi
    prediction = model.predict(df_encoded)
    
    # Mengembalikan hasil prediksi dalam bentuk JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)