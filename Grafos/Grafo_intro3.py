from igraph import Graph
from igraph import plot

#grafo
grafo4 = Graph(directed = False)
grafo4.add_vertices(5)
grafo4.add_edges([(0,1),(1,2),(2,3),(3,4),(4,0),(0,2),(2,1)])
grafo4.add_vertex(5)
grafo4.add_vertex(6)
grafo4.vs["label"] = ["A","B","C","D","E","F","G"]
grafo4.vs["name"] = ["A","B","C","D","E","F","G"]

#impressão da matriz adjacência
print(grafo4.get_adjacency())

#linha
grafo4.get_adjacency()[0,]

#coluna
grafo4.get_adjacency()[0,1]

# Estrutura de repetição para percorrer cada vértice, visualizando o nome e o rótulo
for v in grafo4.vs:
    print(v)
plot(grafo4, bbox=(0,0,300,300))

# Criação de grafo com pesos entre as relações / #aqui e propriedades para os Vértices "VS"
grafo5 = Graph(edges = [(0,1),(2,3),(0,2),(0,3)], directed = True)
grafo5.vs["label"] = ["Fernando", "Pedro", "Jose", "Antonio"]
grafo5.vs["peso"] = [40,30,30,25]
print(grafo5)

# Percorrer os vértices para visualizar os pesos
for v in grafo5.vs: #aqui vc percorre todo o grafo
    print(v)
grafo5.vs[0] #pocição escolhida[0]

# Definição do tipo de amizade e do peso das relações / Aqui a propriedade e para relação entre os vértices "Arestas" / "ES"
grafo5.es["TipoAmizade"] = ["Amigo", "Inimig", "Inimigo", "Amigo"]
grafo5.es["weight"] = [1,2,1,3] #essa propriedade e interna ela já existe, isso torna o grafo ponderado
print(grafo5)

# Percorrer as Arestas, tipo de amizade
for e in grafo5.es:
    print(e)

#propriedades e valores de uma posição
grafo5.es[0]

# tipos de amizade
grafo5.es["TipoAmizade"]
print(grafo5)

#mudanças ds tips da relações em grafos já existente
#existe 4 propriedades que podem ter nos grafos W / U / N / T > Grafo direcionado / ou não direcinado / tem um nome / tem um tipo
# grafo pode ter propriedade tanto para o vertices tanto para aresta, eu posso criar ou usar as internas . 
grafo5.vs["type"] = "Humanos"
grafo5.vs["name"] = "Amizades"
print(grafo5)
plot(grafo5, bbox=(0,0,400,400))
