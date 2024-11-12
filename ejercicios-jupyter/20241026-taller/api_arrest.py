from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Cargar el modelo, etiquetas y características
model = joblib.load('arrest_prediction_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
top_features = joblib.load('top_features.pkl')

# Definir el modelo de datos de entrada basado en las características más importantes
class CrimePredictionInput(BaseModel):
    Primary_Type: str
    Location_Description: str
    Beat: int
    District: int
    Community_Area: float
    Domestic: bool

# Definir el modelo de datos de salida
class CrimePredictionOutput(BaseModel):
    Arrest: bool

# Inicializar la aplicación FastAPI
app = FastAPI()

# Definir el endpoint para hacer predicciones de crimen
@app.post('/predict/')
async def predict_crime_arrest(data: CrimePredictionInput):
    try:
        # Convertir los datos de entrada en un DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Indicar los nombres de las columnas coincidan con los del modelo
        input_data.columns = ['Primary Type', 'Location Description', 'Beat', 'District', 'Community Area', 'Domestic']

        # Codificar las variables categóricas usando los LabelEncoders entrenados
        for col in input_data.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])

        # Seleccionar solo las características más importantes
        input_data = input_data[top_features]

        # Hacer la predicción del arresto
        predicted_arrest = model.predict(input_data)[0]

        return CrimePredictionOutput(Arrest=bool(predicted_arrest))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__name__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)
