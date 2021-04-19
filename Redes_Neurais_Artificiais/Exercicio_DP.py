import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

base = pd.read_csv("C:/Users/Cliente/Desktop/Python/UDEMY/Redes_Neurais_Artificiais/soybean.csv")
base.head()
base

# Criação da variável X que presenta os atributos previsores
X = base.iloc[:, 0:35].values
X

# Criação da variável y que contém as respostas
y = base.iloc[:, 35].values
y


labelencoder = LabelEncoder()
for x in range(35):
    X[:, x] = labelencoder.fit_transform(X[:, x])

# Divisão da base em treino e teste (70% para treinamento e 30% para teste)
X_trainamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                    test_size= 0.3,
                                                                    random_state= 0)
                                                                
# Criação do classificador Random Forest
floresta = RandomForestClassifier(n_estimators= 100)
floresta.fit(X_trainamento, y_treinamento)

#previsoes
previsoes = floresta.predict(X_teste)
previsoes

#agora é possível fazer a matriz de confusion
matriz = confusion_matrix(y_teste, previsoes)
matriz

#agora vamos calcular a taxa de acerto
taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_acerto
taxa_erro = 1 - taxa_acerto
taxa_erro 