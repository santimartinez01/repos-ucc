import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Paso 1: Cargar los datos
data = pd.read_csv('chicago_crimes.csv')

# Paso 2: Preprocesamiento de datos
# Eliminar columnas innecesarias o que no sean útiles para la predicción
data = data.drop(columns=['Case Number', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Latitude', 'Longitude', 'Description'])

# Codificar las variables categóricas usando codificación de frecuencias
for col in data.select_dtypes(include=['object']).columns:
    freq_encoding = data[col].value_counts().to_dict()
    data[col] = data[col].map(freq_encoding)

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop(columns=['Primary Type'])
y = data['Primary Type']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 3: Entrenamiento del modelo con parámetros reducidos
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Paso 4: Evaluación del modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Paso 5: Guardar el modelo entrenado
joblib.dump(model, 'crime_prediction_model.pkl')
