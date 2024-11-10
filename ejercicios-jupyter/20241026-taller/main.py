from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Paso 1: Cargar el modelo entrenado
model = joblib.load('crime_prediction_model.pkl')

# Cargar y definir las columnas usadas durante el entrenamiento
data = pd.read_csv('chicago_crimes.csv')
data = data.drop(columns=['Case Number', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Latitude', 'Longitude', 'Description'])

# Crear un DataFrame auxiliar para el mapeo
auxiliary_df = data[['Primary Type']].drop_duplicates().reset_index(drop=True)

# Codificar las variables categóricas usando codificación de frecuencias
freq_encoding = {}
for col in data.select_dtypes(include=['object']).columns:
    freq_encoding[col] = data[col].value_counts().to_dict()
    data[col] = data[col].map(freq_encoding[col])
    if col == 'Primary Type':
        auxiliary_df[col + '_Encoded'] = auxiliary_df[col].map(freq_encoding[col])

# Crear un diccionario para mapear los códigos numéricos a descripciones
primary_type_mapping = pd.Series(auxiliary_df['Primary Type'].values, index=auxiliary_df['Primary Type_Encoded'].values).to_dict()

# Definir las columnas utilizadas en el modelo
X_columns = data.drop(columns=['Primary Type']).columns

# Paso 2: Definir los modelos de datos para la entrada y salida de la API
class CrimePredictionInput(BaseModel):
    DATE_OF_OCCURRENCE: str
    BLOCK: str
    IUCR: str
    LOCATION_DESCRIPTION: str
    ARREST: str
    DOMESTIC: str
    BEAT: int
    WARD: int

class CrimePredictionOutput(BaseModel):
    PRIMARY_DESCRIPTION: str

# Paso 3: Inicializar la aplicación FastAPI
app = FastAPI()

# Paso 4: Definir el endpoint para hacer predicciones de crimen
@app.post('/predict/')
async def predict_crime(data: CrimePredictionInput):
    try:
        # Convertir los datos de entrada en un DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Codificar las variables categóricas usando codificación de frecuencias
        for col in input_data.columns:
            if col in freq_encoding:
                input_data[col] = input_data[col].map(freq_encoding[col])

        # Asegurarse de que todas las columnas necesarias estén presentes
        for column in X_columns:
            if column not in input_data.columns:
                input_data[column] = 0

        # Reordenar las columnas para que coincidan con el modelo
        input_data = input_data[X_columns]

        # Hacer la predicción
        prediction = model.predict(input_data)

        # Mapear la predicción numérica a la descripción del crimen
        primary_description = primary_type_mapping.get(int(prediction[0]), "Descripción no disponible")

        return CrimePredictionOutput(PRIMARY_DESCRIPTION=primary_description)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

