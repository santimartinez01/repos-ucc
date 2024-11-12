import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Cargar los datos
data = pd.read_csv('chicago_crimes.csv')

# Seleccionar columnas relevantes
columns_to_keep = ['Primary Type', 'Location Description', 'Domestic', 'Beat', 'District', 'Community Area', 'Arrest']
data = data[columns_to_keep]

# Codificar las variables categóricas
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop(columns=['Arrest'])
y = data['Arrest']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Seleccionar las características más importantes
importances = model.feature_importances_
feature_names = X.columns
important_features = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("Importancia de características:\n", important_features)

# Filtrar las características más importantes
top_features = ['Primary Type', 'Location Description', 'Beat', 'District', 'Community Area', 'Domestic']
X_train = X_train[top_features]
X_test = X_test[top_features]

# Re-entrenar el modelo con las características más importantes
model.fit(X_train, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Guardar el modelo, etiquetas y caracteristicas
joblib.dump(model, 'arrest_prediction_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(top_features, 'top_features.pkl')
