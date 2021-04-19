from igraph import Graph
from igraph import plot

#Grafo direcionado
grafo1 = Graph(edges = [(0,1),(1,2),(2,3),(3,0)], directed = True)
grafo1.vs["label"] = range(grafo1.vcount())
print(grafo1)

#grafo não direcionado
grafo2 = Graph(edges = [(0,1),(1,2),(2,3),(3,0)], directed = False)
grafo2.vs["label"] = range(grafo2.vcount())
print(grafo2)

#Gráfico
plot(grafo2, bbox=(0,0,300,300))

#Adicionando vértices e aresta por meieo das funções add_vertices e add_vertex
grafo3 = Graph(directed = False)
grafo3.add_vertices(10)
#grafo3.add_vertex(16) 
grafo3.add_edges([(0,1),(2,2),(2,3),(3,0)])
grafo3.vs["label"] = range(grafo3.vcount())
print(grafo3)
plot(grafo3, bbox = (0,0,300,300))

#criando o quarto grafo, configurando rótulos e personalizados para os vértices
grafo4 = Graph(directed = False)
grafo4.add_vertices(5)
grafo4.add_edges([(0,1),(1,2),(2,3),(3,4),(4,0),(0,2),(2,1)])
grafo4.add_vertex(5)
grafo4.add_vertex(6)
grafo4.vs["label"] = ["A","B","C","D","E","F","G"]
plot(grafo4, bbox=(0,0,300,300))