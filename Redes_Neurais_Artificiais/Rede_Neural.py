# Importação das bibliotecas
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np

## Carregamento da base de dados e criação dos previsores (variáveis independentes - X) e da classe (variável dependente - y)
base = datasets.load_iris()
previsores = base.data
classe = base.target
classe

# Transformação da classe para o formato "dummy", pois temos uma rede neural com 3 neurônios na camada de saída / dividir o problema em 3 coluna distintas
classe_dummy = np_utils.to_categorical(classe)
classe_dummy

# Divisão da base de dados entre treinamento e teste (30% para testar e 70% para treinar)
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                    classe_dummy,
                                                                    test_size=0.3,
                                                                    random_state=0)

# Criação da estrutura da rede neural com a classe Sequential (sequência de camadas)
modelo = Sequential()#primeiro definimos que o nosso modelo será Sequential quer dizer > uma após a outra / sequência...
#usaremos 3 CAMADAS observe que existe 3 modelos. o Dense significa que são todas conectadas.um neurônios é conectado com o neurônio da camada seguinte.
#a units = são unidade ocultas 5 e o input_dim = 4 são as caracteristica do nosso dados, IRIS. dados de entradas....(tipo da planta)
#a units = 5 vc pode alterar eles você pode querer testar outra configuração das redes neurais com 6 ou 8 neurônios, agora o input_dim = 4 não ela é a estruta dos dados.
#primeira camada oculta, 5 neuronios, 4 neuronios de entrada
modelo.add(Dense(units = 5, input_dim = 4))

#segunda camada oculta
modelo.add(Dense(units = 4))
#terceira camada oculta
# Função softmax porque temos um problema de classificação com mais de duas classes > Ela é a função de ATIVAÇÃO
#(é gerada uma probabilidade em cada neurônio) 
modelo.add(Dense(units= 3, activation="softmax"))

# Visualização da estrutura da rede neural
modelo.summary()

# Configuração dos parâmetros da rede neural (adam = algoritmo para atualizar os pesos e loss = cálculo do erro)
#agora vamos compilar e treinar a rede. Usaremos o adam para otimização dos pesos / e o accuracy para métrica de acerto.
modelo.compile(optimizer= "adam", loss = "categorical_crossentropy",
                metrics=["accuracy"])

# Treinamento, dividindo a base de treinamento em uma porção para validação (validation_data) / aqui ele ja faz tudo em um só teste e treina
modelo.fit(X_treinamento, y_treinamento, epochs = 1000,
            validation_data = (X_teste, y_teste))


# Previsões e mudar a variável para True ou False de acordo com o threshold 0.5 / como ela vai retornar entre 0 e 1 e bom ir pra T / F
previsoes = modelo.predict(X_teste)
previsoes = (previsoes > 0.5)
previsoes

# Como é um problema com três saídas, precisamos buscar a posição que possui o maior valor (são retornados 3 valores)
y_teste_matrix = [np.argmax(t) for t in y_teste]
y_previsao_matrix = [np.argmax(t) for t in previsoes]

# Geração da matriz de confusão
confusao = confusion_matrix(y_teste_matrix, y_previsao_matrix)
confusao
        