import os.path
from random import shuffle

path = "ml-100k/"

"""
@brief Divide o conjunto de dados em partes de tamanhos iguais

@param arquivo Nome do arquivo de entrada
@param divisoes Número de divisões do conjunto de dados
@return dados_divididos Divisões do conjunto de dados

"""
def dividir_base(arquivo, divisoes=5):

	dados = open(arquivo, 'r', encoding="utf-8").readlines()

	tamanho = len(dados)

	if (tamanho % divisoes) != 0:
		print("O número de divisões não gera partes de tamanhos iguais!")
		return None

	tamanho_parte = tamanho/ divisoes

	nome_base = path + "base_%s.txt"

	dados_divididos = []

	arquivos_existem = True

	for i in range(divisoes):

		nome = nome_base % (i+1,)

		if os.path.exists(nome):
			f = open(nome_base % (i+1,), 'r')

			arquivos_existem = arquivos_existem and (len(f.readlines()) == tamanho_parte)
		else:
			arquivos_existem = False

	if not arquivos_existem:

		shuffle(dados)

		for i in range(divisoes):
			dados_divididos.append( dados[ int(i*tamanho_parte) :  int((i+1)*tamanho_parte)] )

			with open(nome_base % (i+1,), 'w') as f:
				for linha in dados_divididos[i]:
					f.write(linha)

	else:
		for i in range(divisoes):
			with open(nome_base % (i+1,), 'r') as f:
				dados_divididos.append(f.readlines())

	return dados_divididos

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
			mae += abs(gabarito[i] - predito[i])
			t = t + 1
	mae /= t
	return mae

def calcular_rmse(gabarito, predito):
	rmse = 0
	t = 0
	for i in range(len(predito)):
		if predito[i] > 0:
			rmse += (gabarito[i] - predito[i])**2
			t = t + 1
	rmse = (rmse/t)**.5
	return rmse
