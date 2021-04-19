#produzir regras de associação

import pandas as pd 
from apyori import apriori

#leitura dos dados
dados = pd.read_csv("C:/Users/Cliente/Desktop/Python/UDEMY/Machine_learning/transacoes.txt", header=None)
dados

#transformação para o formato de lista, que é exigido pela biblioteca apyori - 6 é a quantidade de itens na base de dados
transacoes = []
for i in range(6):
    transacoes.append([str(dados.values[i,j]) for j in range(0,3)])
transacoes

# Execução do algoritmo apriori para geração das regras de associação, definindo os parâmetros de suporte e confiança
#aqui fica a mineração das regras
regras = apriori(transacoes, min_support = 0.5, min_confidence = 0.5, min_length=2)

# Criação de nova variável para armazenar somente as regras de associação
resultados = list(regras)
print(resultados[0])
resultados

# Criação de nova variável, percorrendo a variável anterior para uma melhor visualização dos resultados
resultados2 = [list(x) for x in resultados]
resultados2

# Criação de outra variável para a visualização das regras ficar mais fácil para o usuário, adicionando as regras encontradas na variável resultados2
resultados3 = []
for j in range(0,7):
    resultados3.append([list(x) for x in resultados2[j][2]])
resultados3