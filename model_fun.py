
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import random


def sigmoid_function(X):
    return 1 / (1 + math.e**(-X))

def log_regression4(X, y, alpha, epochs):
    y_ = np.reshape(y, (len(y), 1))  # shape (150,1)
    N = len(X)
    theta = np.random.randn(len(X[0]) + 1, 1)  # initialize theta
    X_vect = np.c_[np.ones((len(X), 1)), X]  # Add x0 (column of 1s)
    avg_loss_list = []
    loss_last_epoch = 9999999
    for epoch in range(epochs):
        sigmoid_x_theta = sigmoid_function(X_vect.dot(theta))  # shape: (150,5).(5,1) = (150,1)
        grad = (1/N) * X_vect.T.dot(sigmoid_x_theta - y_)  # shapes: (5,150).(150,1) = (5, 1)
        best_params = theta
        theta = theta - (alpha * grad)
        hyp = sigmoid_function(X_vect.dot(theta))  # shape (150,5).(5,1) = (150,1)
        avg_loss = -np.sum(np.dot(y_.T, np.log(hyp) + np.dot((1-y_).T, np.log(1-hyp)))) / len(hyp)
        avg_loss_list.append(avg_loss)
        loss_step = abs(loss_last_epoch - avg_loss)
        loss_last_epoch = avg_loss
        if loss_step < 0.001:
            break

    return best_params


def manual_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    indices = list(X.index)
    random.shuffle(indices)

    split_index = int(len(indices) * (1 - test_size))

    train_index = indices[:split_index]
    test_index = indices[split_index:]

    X_train = X.loc[train_index]
    X_test = X.loc[test_index]
    y_train = y.loc[train_index]
    y_test = y.loc[test_index]

    return X_train, X_test, y_train, y_test, train_index, test_index

def manual_scale(X_train, X_test):
    mean = X_train.mean()
    std = X_train.std()

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled

def calculate_accuracy(y_test, y_pred):
    # Asegurar que y sea un array de 1 dimensión
    y_pred_flat = [value[0] for value in y_pred]

    # Contar la cantidad de predicciones correctas
    correct_predictions = np.sum(np.array(y_pred_flat) == np.array(y_test))

    # Calcular precisión
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions

    return accuracy