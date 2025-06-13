# Atividade — Regressão Linear com Gradientes

## Objetivo

O propósito desta atividade é implementar e comparar diferentes algoritmos de otimização baseados em Gradiente Descendente para ajustar modelos de regressão linear sobre conjuntos de dados simulados.

## Tarefas

1. **Completar as funções de gradiente**:
   - `calcular_custo`
   - `calcular_gradiente`
   - `atualizar_theta` (batch)
   - `gradiente_estocástico`
   - `split_batch` e `executar_minibatch` (mini-batch)

2. **Executar experimentos** para estimar os parâmetros (coeficientes `theta_0` e `theta_1`) usando **ao menos dois métodos** entre:
   - Gradiente Descendente Batch
   - Gradiente Estocástico
   - Gradiente Mini-Batch

3. **Aplicar os métodos em três conjuntos de dados diferentes**:
   - Relação linear simples com pouco ruído
   - Relação linear com alto ruído
   - Relação não linear (ex: parabólica)

4. **Classificar cada modelo ajustado** com base em sua **variância** e **viés (bias)**, nas seguintes categorias:
   - Baixa variância e baixo viés
   - Alta variância e alto viés
   - Alta variância e baixo viés
   - Baixa variância e alto viés

5. **Responder ao questionário**

## Entrega

- Gerar um arquivo **PDF** contendo:
  - Os **parâmetros utilizados** em cada experimento (valores de `theta_inicial`, `learning_rate`, `batch_size`, `num_epochs` e tipo de gradiente).
  - Capturas de tela ou gráficos da **reta de regressão ajustada** para cada conjunto.
  - Justificativas da classificação (bias/variância) com base no comportamento dos modelos.
  - Resposta do questionário.

- Anexar o PDF na pasta `documento`.

## Estrutura Sugerida

```
Projeto/
│
├── Utils/
│   ├── gradBatch.py
│   ├── gradEstoc.py
│   └── gradMiniBatch.py
│
├── main.py
├── README.md         
└── documento/
    └── Resposta.pdf
    └── Questionário.pdf
    └── Resposta.docx
    └── Questionário.docx
```

## Observações

- Certifique-se de que todas as funções estejam devidamente comentadas com **docstrings**.
- Você pode reutilizar ou adaptar os conjuntos `x1`, `x2` e `x3` fornecidos no `main.py`.
- A comparação entre os métodos é essencial para entender como **ruído** e **forma do dado** afetam a generalização do modelo.

---
