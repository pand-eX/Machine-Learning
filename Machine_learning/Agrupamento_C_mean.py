# Agrupamento com c-means / a diferença do C para o K e q ele não vai atribuir uma instância de forma absoluta a um cluster
#igual o k fazia ele dizia q a 1 instância pertencia ao 2 grupo a 3 instancias ao 1 grupo e etc.
#o c,means vai fazer uma atribuição ao cluster de forma percentual exemplo : a 1 instância de iris e 70% de aderência
# ao cluster 1 e 20% ao cluster 2 e 10% ao cluster 3.

from sklearn import datasets
import numpy as np 
from sklearn.metrics import confusion_matrix
import skfuzzy

#carregando os dados
iris = datasets.load_iris()

# Aplicação do algoritmo definindo três cluster (c = 3) e passando a matriz transposta (iris.data.T). Os outros parâmetros são obrigatórios e são os default indicados na documentação
#o T / faz a transposição da matriz 
r = skfuzzy.cmeans(data = iris.data.T, c = 3, m = 2, error = 0.005,
                    maxiter = 1000, init= None)#os parametros estão na documentação do agrupador.

# Obtendo as porcentagens de um registros pertencer a um cluster, que está na posição 1 da matriz retornada
previsoes_porcentagem = r[1]

# Visualização da probabilidade de um registro pertencer a cada um dos cluster (o somatório é 1.0 que indica 100%)
#criando assim uma forma de visualização que fica mais amigável 150 é o numero total dos dados 50~50~50
#como são 3 cluster cada um tem um percentual o outro elemento varia conforme o X do nosso lastro 
for x in range(150):
    print(previsoes_porcentagem[0][x] ,previsoes_porcentagem[1][x] ,previsoes_porcentagem[2][x])

# Geração de matriz de contingência para comparação com as classes originais da base de dados / CONFUSÃO
previsoes = previsoes_porcentagem.argmax(axis = 0) #Axis é o eixo 0 que são as linhas
resultados = confusion_matrix(iris.target, previsoes)
resultados