#Deep Learning 
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

dataset = pd.read_csv("C:/Users/Cliente/Desktop/Python/UDEMY/Redes_Neurais_Artificiais/Credit2.csv", sep=";")
dataset

#separação dos variáveis, ignoro primeira pois não tem valor semântico / terems q remover a coluna a 1 coluna ID.
#o python e indexado em 0 mas ignora o ultimo valor.  pegando a coluna de checking_status até num_dependents / 1a10 ignora a ultima fica de 1 a 9.
X = dataset.iloc[:,1:10].values # : > dois pontos significa todas as linhas de 1:10 de 1 até 10.. 
y = dataset.iloc[:, 10].values # aqui nos estamos falando a coluna que ele vai pegar a 10 y = CLASSE . ele pega a classe.

#temos um array e não mais um data frame
X

#label encoder coluna "checking_status" > ela tem 4 valores diferente precisamos transoformas em numero e a >
#atribui valores de 0 a 3 ## transformando em categorias.
LabelEncoder = LabelEncoder()
X[:,0] = LabelEncoder.fit_transform(X[:,0])
X

#one hot encoder coluna credit_history / não tem valores categoricos na base de dados. transformando tudo para númericos
#deve adicionar 5 colunas
onehotencoder = make_column_transformer((OneHotEncoder(categories="auto", sparse=False), [1]), remainder="passthrough")
X = onehotencoder.fit_transform(X)
X

#Excluimos a variável para evitar a dummy variable trap / as colunas ficando autocorrelacionadas 
X = X[:,1:]
X

#Laber encoder com a classe / ela e categorica e vamos transferir para númerico como e good ou bad vai ser 0 e 1
labelenconder_Y = LabelEncoder()
y = labelenconder_Y.fit_transform(y)
y

#separação em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(len(X_train), len(X_test), len(y_train), len(y_test)) #print do comprimento so para ver como vai sair a impressao

#Feature Scalling,/ padronizar os valores númericos usando o Z-Score
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_test
#terminou a cima o processo de transformação agora e a parte de implementar a rede neural >>
#aqui é so a tupologia # arquitetura
Classifier = Sequential()
Classifier.add(Dense(units = 6, kernel_initializer= "uniform", activation= "relu", input_dim = 12 ))
Classifier.add(Dense(units=6, kernel_initializer= "uniform", activation= "relu"))
Classifier.add(Dense(units=1, kernel_initializer= "uniform", activation= "sigmoid")) #como nossa classe e 1 a saída tbm será.
Classifier.compile(optimizer= "adam", loss = "binary_crossentropy", metrics= ["accuracy"])
#aqui ja vamos rodar os treinamento. 
Classifier.fit(X_train, y_train, batch_size = 10, epochs= 100) #batch_size = é o número de registro q vão submeter a rede neural para atualização de pesos.

#agora vamos fazer um previsão usando o X_test # propabilidade em valor de verdadeiro ou Falso / teste lógico
y_pred = Classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred

#com o resutaldo a cima podemos gerar a matriz de confusão.
cm = confusion_matrix(y_test, y_pred)
cm