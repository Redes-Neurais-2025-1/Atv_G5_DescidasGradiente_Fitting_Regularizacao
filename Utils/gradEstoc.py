import pandas as pd
import numpy as np

def gradiente_estocástico(theta_values, features, targets, learning_rate):
    """
    Aplica o algoritmo de Gradiente Descendente Estocástico (SGD) para atualização dos parâmetros
    theta_0 e theta_1 em regressão linear.

    Args:
        theta_values (list or array): Lista com os valores iniciais dos parâmetros [theta_0, theta_1].
        features (array-like): Vetor de variáveis independentes (x).
        targets (array-like): Vetor de variáveis dependentes (y).
        learning_rate (float): Taxa de aprendizado.

    Returns:
        tuple:
            - list: Valores finais dos parâmetros [theta_0, theta_1].
            - list: Lista com o custo (erro quadrático) em cada iteração.
            - list: Histórico dos valores de theta_0 e theta_1 ao longo das iterações.
    """
    theta_0 = theta_values[0]
    theta_1 = theta_values[1]
    armazenar_custo = []
    armazenar_theta_0 = []
    armazenar_theta_1 = []

    for i in range(len(features)):
        armazenar_theta_0.append(theta_0)
        armazenar_theta_1.append(theta_1)
        pred =  # <- implementar
        erro =  # <- implementar
        custo = (1 / 2) * (erro ** 2)
        armazenar_custo.append(custo)
        theta_0 = theta_0 - learning_rate * erro
        theta_1 = theta_1 - (learning_rate * erro) * features[i]

    return [theta_0, theta_1], armazenar_custo, [armazenar_theta_0, armazenar_theta_1]
