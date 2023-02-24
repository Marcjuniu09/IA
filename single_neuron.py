# Funcionamento de um neurônio artificial simples (perceptron), mostrando como ele aprende a classificar dados de entrada.
# Utilizado tando a regressão linear quanto a regressão logística para classificar os dados de entrada e ver qual é a melhor opção.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import math
# definindo a função sigmoide
def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a


# Gerando um conjunto de dados aleatórios
np.random.seed(42)
ages = np.random.randint(low=15, high=70, size=40)
labels = []
for age in ages:
    if age < 30:
        labels.append(0)
    else:
        labels.append(1)
        
for i in range(0, 3):
    r = np.random.randint(0, len(labels) - 1)
    if labels[r] == 0:
        labels[r] = 1
    else:
        labels[r] = 0   

# Resultado com regressão linear

model = LinearRegression()
model.fit(ages.reshape(-1, 1), labels)
#y = m.x + b
m = model.coef_[0]
b = model.intercept_
#0.5 = m.x + b
#0.5 - b = m.x
#(0.5 - b) / m = x
limiar_idade = (0.5 - b) / m
plt.plot(ages, ages * m + b, color = 'blue')
plt.plot([limiar_idade, limiar_idade], [0, 0.5], '--', color = 'green')
plt.scatter(ages, labels, color="red")
plt.show()

#Para este conjunto de dados a regressão linear não é a melhor opção

# Resultado com regressão logística

x = np.arange(-10., 10., 0.2)
sig = sigmoid(x)

model = LogisticRegression()
model.fit(ages.reshape(-1, 1), labels)

#y = m.x + b
m = model.coef_[0][0]
b = model.intercept_[0]

x = np.arange(0, 70, 0.1)
sig = sigmoid(m*x + b)

limiar_idade = 0 - (b / m)

plt.scatter(ages, labels, color="red")
plt.plot([limiar_idade, limiar_idade], [0, 0.5], '--', color = 'green')