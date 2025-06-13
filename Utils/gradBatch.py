import pandas as pd
import numpy as np

def calcular_custo(theta_values, features, targets, samples):
    """
    Calcula a função de custo (Erro Quadrático Médio) para regressão linear.

    Args:
        theta_values (list or array): Lista com dois valores, [theta_0, theta_1].
        features (array-like): Vetor de variáveis independentes (x).
        targets (array-like): Vetor de variáveis dependentes (y).
        samples (int): Número de amostras no conjunto de dados.

    Returns:
        float: Valor da função de custo.
    """
    theta_0 = theta_values[0]
    theta_1 = theta_values[1]
    sigma = 0

    for i in range(samples):
        pred =  # <- implementar
        erro =  # <- implementar
        sigma += erro**2

    return (1 / (2 * samples)) * sigma


def calcular_gradiente(theta_values, features, targets, samples):
    """
    Calcula o gradiente da função de custo com relação aos parâmetros theta_0 e theta_1.

    Args:
        theta_values (list or array): Lista com dois valores, [theta_0, theta_1].
        features (array-like): Vetor de variáveis independentes (x).
        targets (array-like): Vetor de variáveis dependentes (y).
        samples (int): Número de amostras no conjunto de dados.

    Returns:
        list: Gradientes [grad_theta_0, grad_theta_1] que indicam a direção de ajuste dos thetas.
    """
    theta_0 = theta_values[0]
    theta_1 = theta_values[1]
    sigma_0 = 0
    sigma_1 = 0

    for i in range(samples):
        pred =  # <- implementar predição
        erro =  # <- implementar erro
        sigma_0 += erro
        sigma_1 += erro * features[i]

    grad_theta_0 =  # <- implementar
    grad_theta_1 =  # <- implementar

    return [grad_theta_0, grad_theta_1]


def atualizar_theta(theta_values, features, targets, samples, learning_rate, num_iters):
    """
    Atualiza os parâmetros theta utilizando Gradiente Descendente.

    Args:
        theta_values (list or array): Valores iniciais dos parâmetros [theta_0, theta_1].
        features (array-like): Vetor de variáveis independentes (x).
        targets (array-like): Vetor de variáveis dependentes (y).
        samples (int): Número de amostras no conjunto de dados.
        learning_rate (float): Taxa de aprendizado para atualização dos parâmetros.
        num_iters (int): Número de iterações do gradiente descendente.

    Returns:
        tuple:
            - list: Valores finais dos parâmetros [theta_0, theta_1].
            - list: Lista com os custos calculados a cada iteração.
            - list: Lista com o histórico dos valores de theta ao longo das iterações.
    """
    theta_0 = theta_values[0]
    theta_1 = theta_values[1]
    armazenar_custo = []
    armazenar_theta_0 = []
    armazenar_theta_1 = []

    for i in range(num_iters):
        armazenar_custo.append(calcular_custo([theta_0, theta_1], features, targets, samples))
        armazenar_theta_0.append(theta_0)
        armazenar_theta_1.append(theta_1)
        grad = calcular_gradiente([theta_0, theta_1], features, targets, samples)
        theta_0 = theta_0 - learning_rate * grad[0]
        theta_1 = theta_1 - learning_rate * grad[1]

    return [theta_0, theta_1], armazenar_custo, [armazenar_theta_0, armazenar_theta_1]
