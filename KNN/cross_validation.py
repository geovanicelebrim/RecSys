import os.path
from random import shuffle
from numpy import zeros

path = "ml-100k/"

"""
@brief Divide o conjunto de dados em subconjuntos de treino e teste

@param indicesTreino Indices das notas de treino
@param indicesTeste Indices das notas de teste

@return matrizTreino Matriz de treino
@return matrizTeste Matriz de teste

"""
def criarDataset(dados, usuarios, itens, indicesTreino, indicesTeste):

	matrizTreino = zeros((len(usuarios), len(itens)))
	matrizTeste = zeros((len(usuarios), len(itens)))

	for indice in indicesTreino:
		tokens = dados[indice].split()

		usuario = int(tokens[0]) - 1
		item 	= int(tokens[1]) - 1
		nota 	= int(tokens[2])

		matrizTreino[usuario, item] = nota

	for indice in indicesTeste:
		tokens = dados[indice].split()

		usuario = int(tokens[0]) - 1
		item 	= int(tokens[1]) - 1
		nota 	= int(tokens[2])

		matrizTeste[usuario, item] = nota

	return matrizTreino, matrizTeste

"""
@brief Divide o conjunto de dados em treino e teste

Dado o conjunto de dados, este é dividido em dois subconjuntos: treino e teste, 
a uma porcentagem dada como parâmetro. 

@param arquivo Nome do arquivo de entrada
@param porcentagemTreino Porcentagem do conjunto de treino em relação ao conjunto de dados
@return treino Dados do conjunto de treino
@return teste Dados do conjunto de teste

"""
def treinoTesteSplit(arquivo, porcentagemTreino=.8):

	dados = open(arquivo, 'r', encoding="utf-8").readlines()

	conjuntoDeTreino = path + "treino.txt"
	conjuntoDeTeste = path + "teste.txt"

	if not os.path.exists(conjuntoDeTreino) or not os.path.exists(conjuntoDeTeste):
		tamanho = len(dados)
		posicao = int(tamanho*porcentagemTreino)
		for i in range(10):
			shuffle(dados)
		treino, teste = dados[:posicao], dados[posicao:]
		
		with open(conjuntoDeTreino, 'w') as f:
			for linha in treino:
				f.write(linha)

		with open(conjuntoDeTeste, 'w') as f:
			for linha in teste:
				f.write(linha)
	else:
		treino = open(conjuntoDeTreino)
		teste = open(conjuntoDeTeste)

	return treino, teste

"""
@brief Calcula a média absoluta do erro

Dados o gabarito e predito, calcula a média absoluta dos erros

@param gabarito Gabarito do teste
@param predito Lista das predições do teste
@return mae Média absoluta do erro

"""
def calcularMAE(gabarito, predito):
	mae = 0
	t = 0
	for i in range(len(predito)):
		if predito[i] > 0:
			mae += abs(gabarito[i] - predito[i])
			t = t + 1
	mae /= t
	return mae


def calcularRMSE(gabarito, predito):
	rmse = 0
	t = 0
	for i in range(len(predito)):
		if predito[i] > 0:
			rmse += (gabarito[i] - predito[i])**2
			t = t + 1
	rmse = (rmse/t)**.5
	return rmse


def calcularPrecisionRecall(gabarito, predito):
	TP = 0
	TN = 0
	FP = 0
	FN = 0

	for i in range(gabarito.shape[0]):
		for j in range(gabarito.shape[1]):
			if gabarito[i, j] == 0 and predito[i, j] == 0:
				TN = TN + 1
			elif gabarito[i, j] == 0 and predito[i, j] == 1:
				FP = FP + 1
			elif gabarito[i, j] == 1 and predito[i, j] == 0:
				FN = FN + 1
			elif gabarito[i, j] == 1 and predito[i, j] == 1:
				TP = TP + 1

	precision = TP / (TP + FP)
	recall = TP / (TP + FN)

	return (precision, recall)


def calcularF1Score(gabarito, predito):
	precision, recall = calcularPrecisionRecall(gabarito, predito)

	f1 = (2 * precision * recall) / (precision + recall)

	return (precision, recall, f1)