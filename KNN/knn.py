# -*- coding: utf-8 -*-

import os.path
import matplotlib.pyplot as plt
from math import sqrt
from scipy.sparse import lil_matrix
from numpy import array, savetxt, loadtxt, zeros, nonzero, dot, mean, std, sum, intersect1d, argsort, fromiter, delete
from numpy.linalg import norm 
from numpy.random import shuffle
from sklearn.metrics.pairwise import cosine_similarity as cosine, euclidean_distances as euclidean
from random import randint

path = "ml-100k/"
u_user = path + "u.user"
u_item = path + "u.item"
u_data = path + "u.data"

"""
@brief Gera matriz usuario-item

Dado o conjunto de dados com as notas dos usuários para os itens, 
monta uma matriz mn, onde m é o número de usuários, e n, o número de itens. 
Cada valor v_ij da matriz é uma nota dada pelo usuário i ao item j

@param usuarios Dados dos usuários
@param itens Dados dos itens
@param avaliacoes Notas dos usuários aos itens
@return mui Matriz usuário-item

"""
def gerar_matriz_usuario_item(usuarios, itens, avaliacoes):
	mui = lil_matrix((len(usuarios), len(itens)))
	
	for avaliacao in range(len(avaliacoes)):
		usuario, item, nota, tempo = avaliacoes[avaliacao].strip().split("\t")
		linha = int(usuario) - 1
		coluna = int(item) - 1
		mui[linha, coluna] = float(nota)

	return mui


"""
@brief Divide o conjunto de dados em treino e teste

Dados o conjunto de dados, este é dividido em dois subconjuntos: treino e teste, 
a uma porcentagem dada como parâmetro. 

@param dados Conjunto de dados
@param mui Matriz usuário-item
@param porcentagem_treino Porcentagem do conjunto de treino em relação ao conjunto de dados
@return mtreino Matriz esparsa (usuário-item) de treino
@return treino Dados do conjunto de treino
@return mteste Matriz esparsa (usuário-item) de teste
@return teste Dados do conjunto de teste

"""
def treino_teste_split(dados, mui, porcentagem_treino=.8):

	conjunto_de_treino = path + "treino.txt"
	conjunto_de_teste = path + "teste.txt"

	if not os.path.exists(conjunto_de_teste) or not os.path.exists(conjunto_de_teste):
		tamanho = dados.shape[0]
		posicao = int(tamanho*porcentagem_treino)
		shuffle(dados)
		treino, teste = dados[:posicao], dados[posicao:]

		savetxt(conjunto_de_treino, treino, fmt="%d", delimiter="\t")
		savetxt(conjunto_de_teste, teste, fmt="%d", delimiter="\t")

	else:
		treino = loadtxt(conjunto_de_treino, delimiter="\t")
		teste = loadtxt(conjunto_de_teste, delimiter="\t")

	mtreino = lil_matrix((mui.shape[0], mui.shape[1]))
	mteste = lil_matrix((mui.shape[0], mui.shape[1]))

	for t in treino:
		linha, coluna = t[0] - 1, t[1] - 1
		mtreino[linha, coluna] = mui[linha, coluna]

	for t in teste:
		linha, coluna = t[0] - 1, t[1] - 1
		mteste[linha, coluna] = mui[linha, coluna]

	return (mtreino, treino, mteste, teste)


"""
@brief Cria os dados que alimentam o knn

Dados os conjuntos de dados, gera os dados de treino e teste necessários para o knn.

@param usuarios Dados dos usuários
@param itens Dados dos itens
@param avaliacoes Dados das avaliacoes
@return mui Matriz usuário-item
@return mtreino Matriz esparsa (usuário-item) de treino
@return treino Dados do conjunto de treino
@return mteste Matriz esparsa (usuário-item) de teste
@return teste Dados do conjunto de teste

"""
def criar_dataset():
	with open(u_user, encoding="utf-8") as a:
		usuarios = a.readlines()	

	with open(u_item, encoding="utf-8") as a:
		itens = a.readlines()

	with open(u_data, encoding="utf-8") as a:
		avaliacoes = a.readlines()

	mui = gerar_matriz_usuario_item(usuarios, itens, avaliacoes) # matriz usuário-item

	m = zeros((len(avaliacoes), 3), dtype="int") # matriz de avaliacoes
	for a in range(len(avaliacoes)):
		user, item, rating, timestamp = avaliacoes[a].strip().split()
		m[a, 0] = int(user) 
		m[a, 1] = int(item) 
		m[a, 2] = int(rating)

	mtreino, treino, mteste, teste = treino_teste_split(m, mui) 

	return (mui, mtreino, treino, mteste, teste)


"""
@brief Normaliza os valores na matriz usuário-item

Dada uma matriz usuário-item, normaliza as notas usando z-score.

@param matriz Matriz usuário-item
@return Matriz normalizada

"""
def normalizar_zscore(matriz):
	for i in range(matriz.shape[0]):
		colunas = nonzero(matriz[i,:])[1]
		media = lil_matrix.sum(matriz[i,colunas]) / len(colunas)
		desvio = std(matriz[i,colunas].toarray())
		for j in colunas:
			matriz[i, j] = (matriz[i, j] - media) / desvio


"""
@brief Calcula a similaridade entre os usuários usando cosseno

Calcula a similaridade entre os usuários através do cálculo do cosseno.
Dados dois usuários, u e v, o cálculo é feito considerando apenas os itens avaliados por ambos os usuários. 

@param matriz Matriz usuário-item
@return similaridades Matriz de similaridade usuário-usuário

"""
def similaridade_cosseno(matriz):
	similaridades = zeros((matriz.shape[0], matriz.shape[0]))
	
	for i in range(matriz.shape[0]):
		for j in range(matriz.shape[0]):
			u = nonzero(matriz[i,:])[1] # índices dos itens avaliados pelo usúario u
			v = nonzero(matriz[j,:])[1] # índices dos itens avaliados pelo usuário v
			
			intersecao = intersect1d(u, v) # índices das itens avaliados por u e v

			u = matriz[i, intersecao].toarray()[0]
			v = matriz[j, intersecao].toarray()[0]

			if len(intersecao) > 0:
				cosseno = dot(u, v) / (norm(u) * norm(v))
				similaridades[i, j] = cosseno

	return similaridades


"""
@brief Calcula a similaridade de Pearson entre os usuários 
@param matriz Matriz usuário-item
@return similaridades Matriz de similaridade usuário-usuário

"""
def similaridade_pearson(matriz):
	similaridades = zeros((matriz.shape[0], matriz.shape[0]))
	
	for i in range(matriz.shape[0]):
		for j in range(matriz.shape[0]):
			u = nonzero(matriz[i,:])[1] # índices dos itens avaliados pelo usúario i
			v = nonzero(matriz[j,:])[1] # índices dos itens avaliados pelo usuário j
			
			intersecao = intersect1d(u, v) # índices das itens avaliados por i e j

			if len(intersecao) > 0:
				u = matriz[i, intersecao].toarray()[0] # notas dos itens avaliados por i
				v = matriz[j, intersecao].toarray()[0] # notas dos itens avaliados por j

				u = u - mean(u) # subtrai a média das notas de i
				v = v - mean(v) # subsrai a média das notas de j

				if norm(u) > 0 and norm(v) > 0:
					pearson = dot(u, v) / (norm(u) * norm(v)) # calcula o cosseno entre i e j

				similaridades[i, j] = pearson

	return similaridades


"""
@brief Executa o algoritmo do k-nearest neighbours

Para cada entrada do conjunto de teste, é feita a predição da nota de um usuário para o item.
A predição é feita através do algoritmo KNN. As notas dos usuários são normalizadas utilizando z-score.

@param k Número de vizinhos
@param func_similaridade Função de similaridade
@return mae Média do erro

"""
def knn(k=5, func_similaridade=cosine):

	mui, mtreino, treino, mteste, teste = criar_dataset()
	
	similaridade = func_similaridade(mtreino)

	gabarito = teste[:, 2]
	predito = zeros((teste.shape[0]))

	for i in range(teste.shape[0]):
		usuario, item = teste[i, 0] - 1, teste[i, 1] - 1
		vizinhos = nonzero(mtreino[:, item])[0] # indices dos vizinhos (incluindo o próprio usuário)

		if len(vizinhos) < k+1:
			pass
		else:
			vizinhosMaisProximos = vizinhos[argsort(similaridade[usuario, vizinhos])[-(k+1):-1]] # indices dos k vizinhos mais próximos
			similaridades = similaridade[usuario, vizinhosMaisProximos] # similaridade dos k vizinhos mais próximos
			notas = mtreino[vizinhosMaisProximos, item].toarray() # notas dos vizinhos

			nota = 0
			for j in range(k):
				nota += notas[j] * similaridades[j]
			soma = sum(abs(similaridades))
			if soma > 0:
				nota /= soma

			predito[i] = nota

	mae = 0
	t = 0
	for i in range(len(predito)):
		if predito[i] > 0:
			print("Predito: %.2f Gabarito: %.2f\n", predito[i], gabarito[i])
			mae = mae + abs(gabarito[i] - predito[i])
			t = t + 1
	mae /= t	
	return mae


def main():	
	print(knn(k=15, func_similaridade=similaridade_pearson))
	return

if __name__ == "__main__":
	main()