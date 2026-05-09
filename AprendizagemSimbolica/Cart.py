import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.tree import plot_tree

colunas = ['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade', 'classe']
dados = pd.read_csv('01_treino_sinais_vitais_sem_label.txt', names=colunas, header=None)
# dados2 = pd.read_csv('02_treino_sinais_vitais_com_label.txt', names=colunas, header=None)

print(dados.head())
print(dados.shape)
print(dados.describe())

# print()

# print(dados2.head())
# print(dados2.shape)
# print(dados2.describe())

features = ['qPA', 'pulso', 'resp']
X = dados[features]                     #Cria um novo conjunto de dados só com as 3 colunas acima
y_reg = dados['gravidade']              #rEGRESSÃO
y_clf = dados['classe']                 #Classificação

# X2 = dados2[features]
# y_reg2 = dados2['gravidade']
# y_clf2 = dados2['classe']

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf,
    tam_teste = 0.5,
    amostras_estratificada = y_clf,   #Garante a proporção das classes
    seed = 0
)