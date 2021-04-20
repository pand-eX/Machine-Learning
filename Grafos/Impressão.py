from igraph import Graph
from igraph import plot

# Criação de grafo com pesos entre as relações
#repare que no label e peso estamos colocando o vs = vértices pq faz sentido em uma rede social
#e no tipo de amizade e o weight estamos colocando as arestas  = ES
grafo5 = Graph(edges = [(0,1),(2,3),(0,2),(0,3)], directed = True)
grafo5.vs["label"] = ["Fernando", "Pedro", "Jose", "Antonio"]
grafo5.vs["peso"] = [40,30,30,25]
grafo5.es["TipoAmizade"] = ["Amigo", "Inimigo", "Inimigo", "Amigo"]
grafo5.es["weight"] = [1,2,1,3] #o peso é o que torna o grafo ponderado

# Visualizar informações sobre os vértices
for v in grafo5.vs:
    print(v)

# Visualizar informações sobre os aresta
for e in grafo5.es:
    print(e)

# Definição de cores para os vértices
grafo5.vs["cor"] = ["blue", "red", "yellow", "green"] #aqui não deixa o gráfico com cor aqui estamos criando uma propriedade de cor
plot(grafo5, bbox = (0,0,300,300),vertex_color = grafo5.vs["cor"]) #agora aqui sim passamos o vertex_color q faz a mudança das cor no vértices

# pesos para as arrestas/Impressão da aresta mais forte ou mais fina conforme o peso da amizade
#edge_width são as larguras da arestas
plot(grafo5, bbox=(300,300), edge_width = grafo5.es["weight"],
        vertex_color = grafo5.vs["cor"])

# Pesos para os vértices / vertex_size é o parametro da impressão "//usando o peso já definido no grafo5"
plot(grafo5, bbox=(300,300), vertex_size = grafo5.vs["peso"],
        vertex_color = grafo5.vs["cor"])


# Curvatura
plot(grafo5, bbox=(300,300), vertex_size = grafo5.vs["peso"],
        edge_width = grafo5.es["weight"],
        vertex_color = grafo5.vs["cor"],
        edge_curved = 0.4)

#Formato / lembrando que isso so muda a impressão não estamos mudando as propriedade dos vértices.
plot(grafo5, bbox=(300,300), vertex_size = grafo5.vs["peso"],
        edge_width = grafo5.es["weight"],
        vertex_color = grafo5.vs["cor"],
        edge_curved = 0.4, vertex_shape = "square")#você pode colocar até imagem, consultar a documentação se quiser.