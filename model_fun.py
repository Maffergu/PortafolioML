
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import random


def sigmoid_function(X):
    """
    Calcula la función sigmoide para cada valor en X.
    """
    return 1 / (1 + math.e**(-X))


def log_regression4(X, y, alpha, epochs):
    """
    Entrena un modelo de regresión logística usando descenso de gradiente.
    """
    y_ = np.reshape(y, (len(y), 1))  # Reshape y para matriz columna
    N = len(X)  # Número de muestras
    theta = np.random.randn(len(X[0]) + 1, 1)  # Inicializa los parámetros
    X_vect = np.c_[np.ones((len(X), 1)), X]  # Añade columna de 1s para el sesgo
    avg_loss_list = []  # Lista para pérdidas promedio
    loss_last_epoch = 9999999  # Valor inicial alto para la pérdida

    for epoch in range(epochs):
        sigmoid_x_theta = sigmoid_function(X_vect.dot(theta))  # Calcula probabilidades
        grad = (1/N) * X_vect.T.dot(sigmoid_x_theta - y_)  # Calcula el gradiente
        best_params = theta  # Guarda los mejores parámetros
        theta = theta - (alpha * grad)  # Actualiza los parámetros
        hyp = sigmoid_function(X_vect.dot(theta))  # Recalcula probabilidades
        avg_loss = -np.sum(np.dot(y_.T, np.log(hyp) + np.dot((1-y_).T, np.log(1-hyp)))) / len(hyp)  # Calcula pérdida
        avg_loss_list.append(avg_loss)  # Añade pérdida a la lista
        loss_step = abs(loss_last_epoch - avg_loss)  # Cambio en la pérdida
        loss_last_epoch = avg_loss  # Actualiza pérdida
        if loss_step < 0.001:  # Criterio de parada
            break

    return best_params


def manual_split(X, y, test_size=0.2, random_state=None):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    if random_state is not None:
        random.seed(random_state)  # Establece semilla para reproducibilidad

    indices = list(X.index)  # Obtiene los índices
    random.shuffle(indices)  # Mezcla índices

    split_index = int(len(indices) * (1 - test_size))  # Calcula índice de división

    train_index = indices[:split_index]  # Índices de entrenamiento
    test_index = indices[split_index:]  # Índices de prueba

    X_train = X.loc[train_index]  # Conjunto de entrenamiento de características
    X_test = X.loc[test_index]  # Conjunto de prueba de características
    y_train = y.loc[train_index]  # Conjunto de entrenamiento de etiquetas
    y_test = y.loc[test_index]  # Conjunto de prueba de etiquetas

    return X_train, X_test, y_train, y_test, train_index, test_index
                

def manual_scale(X_train, X_test):
    """
    Escala las características de entrenamiento y prueba.
    """
    mean = X_train.mean()  # Media de características
    std = X_train.std()  # Desviación estándar de características

    X_train_scaled = (X_train - mean) / std  # Escala entrenamiento
    X_test_scaled = (X_test - mean) / std  # Escala prueba

    return X_train_scaled, X_test_scaled
                

def calculate_accuracy(y_test, y_pred):
    """
    Calcula la precisión del modelo.
    """
    y_pred_flat = [value[0] for value in y_pred]  # Aplana las predicciones
    correct_predictions = np.sum(np.array(y_pred_flat) == np.array(y_test))  # Cuenta predicciones correctas
    total_predictions = len(y_test)  # Número total de predicciones
    accuracy = correct_predictions / total_predictions  # Calcula precisión

    return accuracy