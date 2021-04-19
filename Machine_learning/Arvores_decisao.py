import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import graphviz
from sklearn.tree import export_graphviz
from yellowbrick.classifier import ConfusionMatrix

# Carregamento da base de dados e definição dos previsores (variáveis independentes - X) e classe (variável dependente - y)
credito = pd.read_csv("C:/Users/Cliente/Desktop/Python/UDEMY/Machine_learning/Credit.csv")
credito.shape 
credito.head()

previsores = credito.iloc[:, 0:20].values
classe = credito.iloc[:, 20].values

# Conversão dos atributos categóricos para atributos numéricos, passando o índice de cada atributo categórico
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder.fit_transform(previsores[:, 6])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder.fit_transform(previsores[:, 9])
previsores[:, 11] = labelencoder.fit_transform(previsores[:, 11])
previsores[:, 13] = labelencoder.fit_transform(previsores[:, 13])
previsores[:, 14] = labelencoder.fit_transform(previsores[:, 14])
previsores[:, 16] = labelencoder.fit_transform(previsores[:, 16])
previsores[:, 18] = labelencoder.fit_transform(previsores[:, 18])
previsores[:, 19] = labelencoder.fit_transform(previsores[:, 19])

# Divisão da base de dados entre treinamento e teste. Usamos 30% para testar e 70% para treinar. Random_state = 0 para sempre obter a mesma divisão da base quando o código for executado
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe,
                                                                    test_size = 0.3, random_state = 0)

# Criação e treinamento do modelo
arvore = DecisionTreeClassifier()
arvore.fit(X_treinamento, y_treinamento)

# Exportação da árvore de decisão para o formato .dot, para posterior visualização
export_graphviz(arvore, out_file = "tree.dot")

# Obtenção das previsões
previsoes = arvore.predict(X_teste)
previsoes

#matriz de confusão
confusao = confusion_matrix(y_teste, previsoes)
confusao

# Visualização da matriz de confusão
# Warning interno da biblioteca yellowbrick, já esta na última versão (sem solução para o warning no momento)

v = ConfusionMatrix(GaussianNB())
v.fit(X_treinamento, y_treinamento)
v.score(X_teste, y_teste)
v.poof()

#taxa de acerto
taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_acerto

#taxa de erro
taxa_erro = 1 - taxa_acerto
taxa_erro

