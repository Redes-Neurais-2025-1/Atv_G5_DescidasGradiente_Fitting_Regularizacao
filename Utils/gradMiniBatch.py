def split_batch(features, targets, num_batches):
    x_split = []
    y_split = []
    n = 0

    for i in range(len(features)):
        vazio_x = []
        vazio_y = []
        for j in range(num_batches):
            if n <= len(features) - 1:
                vazio_x.append(features[n])
                vazio_y.append(targets[n])
                n += 1
                
        if len(vazio_x) != 0:
            x_split.append(vazio_x)
            y_split.append(vazio_y)
        
    return x_split, y_split

def executar_minibatch(theta_values, X_batches, Y_batches, learning_rate, atualizar_theta_fn):
    theta_vector = theta_values
    armazenar_custo = []
    armazenar_theta_0 = []
    armazenar_theta_1 = []

    for i in range(len(X_batches)):
        armazenar_theta_0.append(theta_vector[0])
        armazenar_theta_1.append(theta_vector[1])
        samples = len(X_batches[i])

        theta_vector, custo_hist, _ = atualizar_theta_fn(
            theta_vector, X_batches[i], Y_batches[i], samples, learning_rate, 1
        )
        custo = custo_hist[0]
        armazenar_custo.append(custo)

    return theta_vector, armazenar_custo, [armazenar_theta_0, armazenar_theta_1]
