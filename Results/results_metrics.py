from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np 
import pandas as pd 

def EvaluationMetrics(model, x_train, y_train, x_test, y_test):
    mse_2 = mean_squared_error(y_test, result_linear)
    r2 = r2_score(y_test, result_linear)

    print('Mean squared error:', mse_2)
    print('R2 Score:', r2)

    #print("R2 Score en datos de treinamento :", model.score(x_train, y_train))
    #print("R2 Score en datos de prueba:", model.score(x_test, y_test))

def CrossValidation(model, x_train, y_train):
    # Realizar la validación cruzada en el modelo
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')

    # Calcular el Mean Mean Squared Error y Standard Deviation of Mean Squared Error en la validación cruzada
    mean_mse_cv = -np.mean(cv_scores)
    std_mse_cv = np.std(cv_scores)
    print('Mean Cross-Validated Mean Squared Error:', mean_mse_cv)
    print('Standard Deviation of Cross-Validated Mean Squared Error:', std_mse_cv)