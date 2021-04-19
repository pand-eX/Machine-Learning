#Agrupamento ~ dados não supervisionados. usar a distancias euclidianas, apartir de um numero de cluster 
#agrupar sem usar a classe, e compara se o grupamento k-mean tem uma similaridade com as classes do iris
#utilizaremos o KMEANS pq ele agrupa sem gerar ruídos. 
from sklearn import datasets
import numpy as np 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

iris = datasets.load_iris()
# visualização de quantos registros existem por classe / usaremos o UNIQUE para contar a quantidade de registro na classe
unicos, quantidade = np.unique(iris.target, return_counts = True)
unicos
quantidade

# Agrupamento com k-means, utilizando 3 clusters (de acordo com a base de dados)
cluster = KMeans(n_clusters = 3) #o 3 e o numero de classe que nos ja sabemos 3
cluster.fit(iris.data) #o agrupamento n estamos usando o target so usando as 4 colunas numericas q tem largura e comprimento das pétalas.

# Visualização dos três centroides / metodo q utilozou no processo de agrupamento.
centroides = cluster.cluster_centers_
centroides

# Visualização dos grupos que cada registro foi associado / visualisando os grupo que ele criou não e previsão e agrupamento.
previsoes = cluster.labels_
previsoes

# Contagem dos registros por classe
unicos2, quantidade2 = np.unique(previsoes, return_counts = True)
unicos2
quantidade2

# Geração da matriz de contingência para comparar os grupos com a base de dados / CONFUSÃO
resultados = confusion_matrix(iris.target, previsoes) #utilizando a classe real dos dados original. previsoes = agrupamentos
resultados

# Geração do gráfico com os clusters gerados, considerando para um (previsoes 0, 1 ou 2)
# Usamos somente as colunas 0 e 1 da base de dados original para termos 2 dimensões / usando grafico de dispersão

plt.scatter(iris.data[previsoes ==0, 0], iris.data[previsoes ==0, 1],
            c = "green", label = "Setosa")

plt.scatter(iris.data[previsoes ==1,0], iris.data[previsoes == 1, 1],
            c = "red", label = "Versicolor")

plt.scatter(iris.data[previsoes ==2, 0], iris.data[previsoes == 2, 1],
            c = "blue", label = "Virginica")

plt.legend()
plt.show()