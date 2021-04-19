from igraph import Graph
from igraph import plot

# Definição do grafo com as arestas
grafo1 = Graph(edges = [(0,1),(1,2),(2,3),(3,0)], directed = True)

# Definição do rótulo de cada vértice
grafo1.vs["label"] = range(grafo1.vcount())
print(grafo1)

#gráfico
plot(grafo1, bbox=(0,0,300,300)) #defindo margens e o tamanho do gráfico

#Criando o segundo Grafo
grafo2 = Graph(edges = [(0,1),(1,2),(2,3),(3,0),(0,3),(3,2),(2,1),(1,0)], directed = True)
grafo2.vs["label"] = range(grafo2.vcount())
plot(grafo2, bbox = (0,0,400,400))

#Grafo com laço
grafo3 = Graph(edges = [(0,1),(1,2),(2,3),(3,0),(1,1)])
grafo3.vs["label"] = range(grafo3.vcount())
plot(grafo3, bbox = (0,0,300,300))

#Criação do quarto grafo
grafo4 = Graph(edges = [(0,1),(1,2),(2,3),(3,0),(1,1)], directed = True)
#adicionamos vertice isolado
grafo4.add_vertex(5)
grafo4.vs["label"] = range(grafo4.vcount())
plot(grafo4, bbox=(0,0,300,300))