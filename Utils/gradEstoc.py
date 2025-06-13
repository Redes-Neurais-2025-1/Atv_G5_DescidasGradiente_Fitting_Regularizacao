import pandas as pd
import numpy as np

def gradiente_estocástico (theta_values, features, targets, learning_rate):
    theta_0 = theta_values[0]
    theta_1 = theta_values[1]
    armazenar_custo = []
    armazenar_theta_0 = []
    armazenar_theta_1 = []

    for i in range(len(features)):
        armazenar_theta_0.append(theta_0)
        armazenar_theta_1.append(theta_1)
        pred = 
        erro = 
        custo = (1/2)*(erro**2)
        armazenar_custo.append(custo)
        theta_0 = theta_0 - learning_rate * erro
        theta_1 = theta_1 - (learning_rate * erro) * features[i]

    return [theta_0, theta_1], armazenar_custo, [armazenar_theta_0, armazenar_theta_1]