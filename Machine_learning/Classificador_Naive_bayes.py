# Importação das bibliotecas
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from yellowbrick.classifier import ConfusionMatrix

# Carregamento da base de dados e definição dos previsores (variáveis independentes - X) e a classe (variável dependente - y)
credito = pd.read_csv("C:/Users/Cliente/Desktop/Python/UDEMY/Machine_learning/Credit.csv")
credito.shape
credito.head()

# Formato de matriz
previsores = credito.iloc[:,0:20].values
classe = credito.iloc[:,20].values

# Transformação dos atributos categóricos em atributos numéricos, passando o índice de cada coluna categórica
# Precisamos criar um objeto para cada atributo categórico, pois na sequência vamos executar o processo de encoding novamente para o registro de teste
# Se forem utilizados objetos diferentes, o número atribuído a cada valor poderá ser diferente, o que deixará o teste inconsistente
LabelEncoder1 = LabelEncoder()
previsores[:,0] = LabelEncoder1.fit_transform(previsores[:,0])

LabelEncoder2 = LabelEncoder()
previsores[:,2] = LabelEncoder2.fit_transform(previsores[:,2])

LabelEncoder3 = LabelEncoder()
previsores[:,3] = LabelEncoder3.fit_transform(previsores[:,3])

LabelEncoder4 = LabelEncoder()
previsores[:,5] = LabelEncoder4.fit_transform(previsores[:,5])

LabelEncoder5 = LabelEncoder()
previsores[:,6] = LabelEncoder5.fit_transform(previsores[:,6])

LabelEncoder6 = LabelEncoder()
previsores[:,8] = LabelEncoder6.fit_transform(previsores[:,8])

LabelEncoder7 = LabelEncoder()
previsores[:,9] = LabelEncoder7.fit_transform(previsores[:,9])

LabelEncoder8 = LabelEncoder()
previsores[:,11] = LabelEncoder8.fit_transform(previsores[:,11])

LabelEncoder9 = LabelEncoder()
previsores[:,13] = LabelEncoder9.fit_transform(previsores[:,13])

LabelEncoder10 = LabelEncoder()
previsores[:,14] = LabelEncoder10.fit_transform(previsores[:,14])

LabelEncoder11 = LabelEncoder()
previsores[:,16] = LabelEncoder11.fit_transform(previsores[:,16])

LabelEncoder12 = LabelEncoder()
previsores[:,18] = LabelEncoder12.fit_transform(previsores[:,18])

LabelEncoder13 = LabelEncoder()
previsores[:,19] = LabelEncoder13.fit_transform(previsores[:,19])

# Divisão da base de dados entre treinamento e teste (30% para testar e 70% para treinar)

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe, test_size = 0.3,
                                                                                        random_state = 0)
X_teste  

# Criação e treinamento do modelo (geração da tabela de probabilidades)
naive_bayes = GaussianNB()
naive_bayes.fit(X_treinamento, y_treinamento)

# Previsões utilizando os registros de teste
previsoes = naive_bayes.predict(X_teste)
previsoes

#geração da matriz de confusão e cálculo da taxa de acerto e erro
confusao = confusion_matrix(y_teste, previsoes)
confusao

taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_error = 1 - taxa_acerto
taxa_acerto

# Visualização da matriz de confusão
# Warning interno da biblioteca yellowbrick, já esta na última versão (sem solução para o warning no momento)
v = ConfusionMatrix(GaussianNB())
v.fit(X_treinamento, y_treinamento)
v.score(X_teste, y_teste)
v.poof()

# Previsão com novo registro, transformando os atributos categóricos em numéricos
novo_credito = pd.read_csv("C:/Users/Cliente/Desktop/Python/UDEMY/Machine_learning/NovoCredit.csv")
novo_credito.shape
novo_credito

# Usamos o mesmo objeto que foi criado antes, para manter o padrão dos dados
# Chamamos somente o método "transform", pois a adaptação aos dados (fit) já foi feita anteriormente

novo_credito = novo_credito.iloc[:,0:20].values
novo_credito[:,0] = LabelEncoder1.transform(novo_credito[:,0])
novo_credito[:, 2] = LabelEncoder2.transform(novo_credito[:, 2])
novo_credito[:, 3] = LabelEncoder3.transform(novo_credito[:, 3])
novo_credito[:, 5] = LabelEncoder4.transform(novo_credito[:, 5])
novo_credito[:, 6] = LabelEncoder5.transform(novo_credito[:, 6])
novo_credito[:, 8] = LabelEncoder6.transform(novo_credito[:, 8])
novo_credito[:, 9] = LabelEncoder7.transform(novo_credito[:, 9])
novo_credito[:, 11] = LabelEncoder8.transform(novo_credito[:, 11])
novo_credito[:, 13] = LabelEncoder9.transform(novo_credito[:, 13])
novo_credito[:, 14] = LabelEncoder10.transform(novo_credito[:, 14])
novo_credito[:, 16] = LabelEncoder11.transform(novo_credito[:, 16])
novo_credito[:, 18] = LabelEncoder12.transform(novo_credito[:, 18])
novo_credito[:, 19] = LabelEncoder13.transform(novo_credito[:, 19])

#resultado da previsão
naive_bayes.predict(novo_credito)