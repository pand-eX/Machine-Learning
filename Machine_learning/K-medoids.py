#k-medoids é um cluster principal caracteristica dele é que o centro onde s cluster são inicializados ao invés de serem
#pontos aleatórios como no k-means o k-medoids usam pontos reais de dados 

from sklearn import datasets
from sklearn.metrics import confusion_matrix
import numpy as np 
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer

#carregamento dos dados
iris = datasets.load_iris()
#criando o objeto cluster / porém estão pegando as 2 primeiras colunas de iris(:, 0:2) para melhor intemdimento dos graficos do cluster, voce pode alterar para 4 se quisers
# Configuração dos parâmetros do k-medoids, utilizando somente as duas primeiras colunas da base de dados por causa da visualização apenas
# 3, 12 e 20 são índices aleatórios de registros da base de dados (inicialização)
cluster = kmedoids(iris.data[:, 0:2], [3,12,20]) #a sintaxe do pyto e de [:, 0:2] de 0 até 2 ignorando o último valor pegando 0 e 1 
#visualização dos pontos escolhidos (3,12,20) aqui estou falando q vou usar o registro para iniciar dessas posições. apartir dali
cluster.get_medoids()

# Aplicação do algoritmo para o agrupamento, obtenção da previsões (grupo de cada registro) e visualização dos medoides
cluster.process()
previsoes = cluster.get_clusters()
medoides = cluster.get_medoids()
#lista de 3 elementos, com os indices dos registros do cluster
previsoes #aqui são 3 lista olhando a imagen vc vê / é diferente você não vê o vetor de 0 e 1 ou vetor de propabilidade como era o c-means

#visualização do agrupamento / gerar o grafico do cluster com o centroides
v = cluster_visualizer()
v.append_clusters(previsoes, iris.data[:,0:2])
v.append_cluster(medoides, data = iris.data[:,0:2], marker='*', markersize= 20)
v.show()

# Código para criar duas listas, uma com os grupos reais da base de dados e outra com os valores dos grupos
# Utilizado posteriormente para visualização da matriz de contingência
lista_previsoes = []
lista_real = []
for i in range(len(previsoes)):
    for j in range(len(previsoes[i])):
        lista_previsoes.append(i)
        lista_real.append(iris.target[previsoes[1][j]])

# Geração da matriz de contingência, comparando os grupos reais com os grupos previstos / isso e preciso para fazer a tabela REAL de contigÊncia"CONFUSÃO"
lista_previsoes = np.asarray(lista_previsoes)
lista_real = np.asarray(lista_real)
resultados = confusion_matrix(lista_real, lista_previsoes)
resultados
