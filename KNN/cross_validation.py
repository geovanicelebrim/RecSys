import os.path
from numpy.random import shuffle

path = "ml-100k/"

"""
@brief Divide o conjunto de dados em treino e teste

Dado o conjunto de dados, este é dividido em dois subconjuntos: treino e teste, 
a uma porcentagem dada como parâmetro. 

@param arquivo Nome do arquivo de entrada
@param porcentagem_treino Porcentagem do conjunto de treino em relação ao conjunto de dados
@return treino Dados do conjunto de treino
@return teste Dados do conjunto de teste

"""
def treino_teste_split(arquivo, porcentagem_treino=.8):

	dados = open(arquivo, 'r', encoding="utf-8").readlines()

	conjunto_de_treino = path + "treino.txt"
	conjunto_de_teste = path + "teste.txt"

	if not os.path.exists(conjunto_de_teste) or not os.path.exists(conjunto_de_teste):
		tamanho = len(dados)
		posicao = int(tamanho*porcentagem_treino)
		shuffle(dados)
		treino, teste = dados[:posicao], dados[posicao:]
		
		with open(conjunto_de_treino, 'w') as f:
			for linha in treino:
				f.write(linha)

		with open(conjunto_de_teste, 'w') as f:
			for linha in teste:
				f.write(linha)
	else:
		treino = open(conjunto_de_treino)
		teste = open(conjunto_de_teste)

	return treino, teste

"""
@brief Divide o conjunto de dados em treino e teste usando k-fold

@param
@return

"""
def kfold_split():
	return

"""
@brief Calcula a média absoluta do erro

Dados o gabarito e predito, calcula a média absoluta dos erros

@param gabarito Gabarito do teste
@param predito Lista das predições do teste
@return mae Média absoluta do erro

"""
def calcular_mae(gabarito, predito):
	mae = 0
	t = 0
	for i in range(len(predito)):
		if predito[i] > 0:
			mae += abs(predito[i] - gabarito[i])
			t = t + 1
	mae /= t
	return mae
