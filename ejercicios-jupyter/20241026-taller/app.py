from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

app = Flask(__name__)

# Cargar el modelo entrenado y otros recursos
model = joblib.load('model_arrest.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos de entrada en formato JSON
    data = request.get_json()

    # Convertir a DataFrame
    df = pd.DataFrame([data])

    # Preprocesar los datos
    df = preprocess_data(df, scaler, label_encoders)

    # Hacer predicción
    prediction = model.predict(df)

    # Devolver la predicción como JSON
    return jsonify({'prediction': prediction[0]})

def preprocess_data(df, scaler, label_encoders):
    # Convertir características categóricas a numéricas
    for col, le in label_encoders.items():
        df[col] = le.transform(df[col])

    # Escalar características numéricas
    df_scaled = scaler.transform(df)

    return df_scaled

if __name__ == '__main__':
    app.run(debug=True)  # Agregar debug=True para obtener más información sobre errores
