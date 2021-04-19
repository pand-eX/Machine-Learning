# Importação das bibliotecas
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd
from scipy import stats

#carregamos de disco para visualizar apenas
iristemp = pd.read_csv("C:/Users/Cliente/Desktop/Python/UDEMY/Machine_learning/Iris.csv")
iristemp

# Carregamento da base de dados e visualização 
iris = datasets.load_iris()
stats.describe(iris.data)

#classes e dados
iris.target
iris.data

# Criação dos previsores (variável independente - X) e class (variável dependente - y)
previsores = iris.data
classe = iris.target

# Divisão da base de dados entre treinamento e teste (70% para treinamento e 30% para teste)
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe,
                                                                    test_size=0.3, random_state=0)
len(X_treinamento)

# Criação do modelo, treinamento
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_treinamento, y_treinamento)

#obtenção das previsões
previsoes = knn.predict(X_teste)
previsoes

#obtenção da matriz de confusão 
confusao = confusion_matrix(y_teste, previsoes)
confusao

#taxas de acerto e erro
taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_erro = 1 - taxa_acerto
taxa_acerto