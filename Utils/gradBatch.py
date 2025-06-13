import pandas as pd
import numpy as np

def calcular_custo(theta_values, features, targets, samples):
    theta_0 = theta_values[0]
    theta_1 = theta_values[1]
    sigma = 0
    
    for i in range(samples):
        pred = 
        erro = 
        sigma += erro**2

    return (1/(2*samples)) * sigma

##Calcular o gradiente
def calcular_gradiente(theta_values, features, targets, samples):
    theta_0 = theta_values[0]
    theta_1 = theta_values[1]
    sigma_0 = 0
    sigma_1 = 0

    for i in range(samples):
        pred = 
        erro = 
        sigma_0 += erro
        sigma_1 += erro * features[i]
        
    grad_theta_0 = 
    grad_theta_1 = 

    return [grad_theta_0, grad_theta_1]


##Atualização do theta
def atualizar_theta(theta_values, features, targets, samples, learning_rate, num_iters):
    theta_0 = theta_values[0]
    theta_1 = theta_values[1]
    sigma_0 = 0
    sigma_1 = 0
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