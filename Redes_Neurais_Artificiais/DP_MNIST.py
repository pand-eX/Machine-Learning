#Deep Learning > MNIST
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist

#obtenção ds dados e "divisão automática entre treinamento e teste / usaremos o MNIST para isso."
(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

# Visualização de imagens específicas
plt.imshow(X_treinamento[21], cmap = "gray")
plt.title(y_treinamento[21])
plt.show()

# Mudança de dimensão, originalmente está em 28x28 e precisamos 784 / aqui é uma matriz e precisamo dos dados em um vetor
X_treinamento = X_treinamento.reshape((len(X_treinamento), np.prod(X_treinamento.shape[1:])))
X_teste = X_teste.reshape((len(X_teste), np.prod(X_teste.shape[1:])))
X_teste[0]

# Transformação dos dados para float para podermos normalizar os dados
X_treinamento = X_treinamento.astype("float32")
X_teste = X_teste.astype("float32")

# Normalização (255 é o valor máximo de um pixel) se olhar para a formual a cima sabemos q o maximo e 255 de pixel..
X_treinamento /= 255
X_teste /= 255

# Transformação para o formato dummy (temos 10 classes) 0a9... criams um vetor onde nao tem nada e 0 e onde ta o carácter e 1
y_treinamento = np_utils.to_categorical(y_treinamento, 10)
y_teste = np_utils.to_categorical(y_teste, 10)
y_teste[0]

# Estrutura da rede neural: 784 - 64 - 64 - 64 - 10 / / matriz de 28x28 = 764 atributos pixels...
# Dropout é utilizado para zerar uma porcentagem dos neurônios, para evitar o overfitting / para rede não ficar superajustada.
#quando os 64 neurônios se juntar com os outros do neurônios da camada seguinte 20% dessas transmissões de dados vao ser zeradas.
modelo = Sequential()#objeto sequential .. ja conhecemos . o activation = ativação do relu muito utilizado para redes de imagens, rede do tipo convolucionais
modelo.add(Dense(units= 64, activation= "relu", input_dim = 784))
modelo.add(Dropout(0.2))
modelo.add(Dense(units= 64, activation= "relu"))
modelo.add(Dropout(0.2))
modelo.add(Dense(units= 64, activation= "relu"))
modelo.add(Dropout(0.2))
#camada de saida, softmax probabilidade
modelo.add(Dense(units= 10, activation= "softmax"))#queremos prevêr 10 caracteres do 0ao9 e o softmax q vai dar o percentual da aderência de cada caractere

# Visualização da estrutura da rede neural
modelo.summary()

# Configuração dos parâmetros da rede neural e treinamento (utilizando base de dados de validação)
# Na variável historico temos os histórico das execuções (erro e accuracy)
modelo.compile(optimizer= "adam", loss = "categorical_crossentropy",
                metrics = ["accuracy"])
historico = modelo.fit(X_treinamento, y_treinamento, epochs= 20,
                        validation_data=(X_teste, y_teste))

# Gráfico para visualizar os erros e accuracy
historico.history.keys()
#evolução do erro, azul
plt.plot(historico.history["val_loss"])
#performance da rede
plt.plot(historico.history["val_accuracy"])
plt.show()

# Obtenção das previsões 
previsoes = modelo.predict(X_teste)
previsoes

# valor máximo (com a probabilidade maior por serem 10 saídas) e geração da matriz de confusão
y_teste_matrix = [np.argmax(t) for t in y_teste]
y_previsoes_matrix = [np.argmax(t) for t in previsoes]
confusao = confusion_matrix(y_teste_matrix, y_previsoes_matrix)
confusao

# Previsão com um novo registro, convertendo o array para o formato de matriz
#número 4
y_treinamento[20]

#passo a mesma posição para o modelo prever
novo = X_treinamento[20]
#de matriz para vetor
novo = np.expand_dims(novo, axis = 0)
#previsao
pred = modelo.predict(novo)
#maior valor
pred = [np.argmax(pred) for t in pred]
pred