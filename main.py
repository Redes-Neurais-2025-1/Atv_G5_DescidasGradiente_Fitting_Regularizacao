import numpy as np
import matplotlib.pyplot as plt

from Utils.gradBatch import calcular_custo, calcular_gradiente, atualizar_theta
from Utils.gradEstoc import gradiente_estocástico
from Utils.gradMiniBatch import split_batch, executar_minibatch

def main():
    # Conjunto 1: Relação linear simples com ruído leve
    x1 = np.linspace(0, 10, 300)
    y1 = 2.5 * x1 + 7 + np.random.randn(300) * 1

    # Conjunto 2: Relação linear com ruído mais forte
    x2 = np.linspace(0, 10, 300)
    y2 = -1.2 * x2 + 4 + np.random.randn(300) * 5

    # Conjunto 3: Relação não linear (parabólica)
    x3 = np.linspace(-5, 5, 300)
    y3 = 0.8 * x3**2 - 3 * x3 + 2 + np.random.randn(300) * 3

    x = x1 #ou x2, ou x3
    y = y1 #ou y2, ou y3

    # Hiperparâmetros
    theta_inicial = 
    learning_rate = 
    num_epochs = 100
    tipo_gradiente = 'batch'  # 'batch', 'estocastico', 'mini-batch'
    batch_size = 10  # usado apenas no mini-batch

    # Armazenar histórico
    custo_hist = []
    theta_0_hist = []
    theta_1_hist = []

    if tipo_gradiente == 'batch':
        theta_0 = theta_inicial[0]
        theta_1 = theta_inicial[1]
        for _ in range(num_epochs):
            custo = calcular_custo([theta_0, theta_1], x, y, len(x))
            grad = calcular_gradiente([theta_0, theta_1], x, y, len(x))
            theta_0 -= learning_rate * grad[0]
            theta_1 -= learning_rate * grad[1]
            custo_hist.append(custo)
            theta_0_hist.append(theta_0)
            theta_1_hist.append(theta_1)

    elif tipo_gradiente == 'estocastico':
        theta = theta_inicial
        for _ in range(num_epochs):
            theta, custos, [ths_0, ths_1] = gradiente_estocástico(theta, x, y, learning_rate)
            custo_hist.extend(custos)
            theta_0_hist.extend(ths_0)
            theta_1_hist.extend(ths_1)

    elif tipo_gradiente == 'mini-batch':
        theta = theta_inicial
        for _ in range(num_epochs):
            X_batches, Y_batches = split_batch(x, y, num_batches=batch_size)
            theta, custos, [ths_0, ths_1] = executar_minibatch(theta, X_batches, Y_batches, learning_rate, atualizar_theta)
            custo_hist.extend(custos)
            theta_0_hist.extend(ths_0)
            theta_1_hist.extend(ths_1)

    print(f"Coeficientes finais: theta_0 = {theta_0_hist[-1]}, theta_1 = {theta_1_hist[-1]}")

    # Gráfico do custo
    plt.plot(custo_hist)
    plt.title(f"Evolução do custo - Gradiente: {tipo_gradiente}")
    plt.xlabel("Iterações")
    plt.ylabel("Custo")
    plt.grid(True)
    plt.show()

    # Gráfico da reta ajustada
    plt.scatter(x, y, label="Dados")
    y_pred = theta_0_hist[-1] + theta_1_hist[-1] * x
    plt.plot(x, y_pred, color='red', label="Reta ajustada")
    plt.title("Ajuste do Modelo")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
