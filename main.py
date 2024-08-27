import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
#from sklearn.metrics import classification_report
from model_fun import log_regression4, sigmoid_function, manual_scale, manual_split, calculate_accuracy


# Cargar el DataFrame (reemplaza 'Sleep_health.csv' con el archivo real)
df = pd.read_csv('Sleep_health.csv')

# Crear la nueva columna 'Quality of Sleep Binary'
df['Quality of Sleep Binary'] = df['Quality of Sleep'].apply(lambda x: 1 if x > 6 else 0)

# Seleccionar características y variable objetivo
X = df[['Sleep Duration', 'Stress Level']]
y = df['Quality of Sleep Binary']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test, train_index, test_index = manual_split(X, y, test_size=0.2, random_state=0)

# Escalar características
X_train, X_test = manual_scale(X_train, X_test)

# Convertir a matrices numpy
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Entrenar el modelo
epochs = 1000
alpha = 1  # Ajusta la tasa de aprendizaje si es necesario
best_params = log_regression4(X_train, y_train, alpha, epochs)

# Realizar las predicciones para el conjunto de prueba
X_test_with_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]  # Añadir columna de 1s para el término de sesgo
pred_probabilities = sigmoid_function(X_test_with_bias.dot(best_params))
y_pred = (pred_probabilities >= 0.5).astype(int)  # Convertir probabilidades a clases binarias

# Calcular la precisión manualmente
accuracy_manual = calculate_accuracy(y_test, y_pred)

# Mostrar la precisión manualmente calculada
print(f'Model Accuracy: {accuracy_manual:.4f}')

# Mostrar el informe de clasificación (si se necesita)
#print('Classification Report:\n', classification_report(y_test, y_pred_flat)) 

# Seleccionar un índice para predecir
index_ = 22

# Preparar los datos para la predicción
X_to_predict = [list(X_test[index_])]
X_to_predict_with_bias = np.c_[np.ones((len(X_to_predict), 1)), X_to_predict]

# Realizar la predicción
pred_probability = sigmoid_function(X_to_predict_with_bias.dot(best_params))
predicted_class = (pred_probability >= 0.5).astype(int)

# Obtener el valor original de la calidad del sueño usando el índice del conjunto de prueba
original_val = df.loc[test_index[index_], 'Quality of Sleep']

# Mostrar resultados
print(f'Predicted Probability: {pred_probability[0][0]:.4f}')
print(f'Predicted Quality: {int(predicted_class[0][0])}')
print(f'Actual Quality: {y_test[index_]}')
print(f'Original Quality Value: {original_val}')
