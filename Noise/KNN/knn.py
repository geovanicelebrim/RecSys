from cross_validation import *
from similarities import *
from numpy import *
from random import choice, shuffle

"""
@brief Perturba uma porcentagem da base

@param matriz Matriz original
@param porcentagem Porcentagem da base que sera perturbada. Default 10%

@return novaMatriz Matriz perturbada
@return matrizIndices Indices das notas perturbadas

"""
def perturbarBase(matriz, porcentagem=.1):

	print("Perturbando ", porcentagem * 100 , "%", " dos dados\n")

	indices = [k for k in range(len(matriz[matriz>0]))]

	matrizIndices = full((matriz.shape), -1, dtype=int)
	matrizIndices[matriz[:]>0] = 0

	for k in range(10):
		shuffle(indices)

	posicao = int(len(indices)*porcentagem)

	novaMatriz = matriz.copy()

	u, i = nonzero(matriz)

	for k in indices[:posicao]:
		nota = matriz[u[k], i[k]]
		escala = [1, 2, 3, 4, 5]
		escala.remove(nota)
		novaNota = choice(escala)
		novaMatriz[u[k], i[k]] = novaNota
		matrizIndices[u[k], i[k]] = 1

	return (novaMatriz, matrizIndices)

"""
@brief Detecta o ruido utilizando algoritmo de O'Mahony

@param matriz Matriz analisada
@param threshold Limiar de diferenca entre a nota original e a predita
@param k Numero de vizinhos mais proximos para o calculo da predicao
@param funcSimilaridade Funcao de similaridade do KNN

@return matrizIndices Indices dos ruidos detectados
"""

def omahonyDeteccao(matriz, threshold=.25, k=20, funcSimilaridade=pearson):

	print("Detectando ruidos com algoritmo de OMahony\n")

	similaridades = funcSimilaridade(matriz)
	print()

	u, i = nonzero(matriz)

	gabarito = matriz[u, i]
	predito  = zeros(len(gabarito))
	
	matrizIndices = full((matriz.shape), -1, dtype=int)
	matrizIndices[u, i] = 0

	j = 0
	for usuario, item in zip(u, i):

		print("\rCalculando predições: ", ((j+1)*100)//len(predito), "%", end="")

		vizinhos = nonzero(matriz[:, item])[0]
		vizinhos = vizinhos[similaridades[usuario, vizinhos]>0.]

		if len(vizinhos) > (k+1):
			pass		
		else:
			knn 	= vizinhos[argsort(similaridades[usuario, vizinhos])[-(k+1):-1]]
			sims	= similaridades[usuario, knn]
			notas	= matriz[knn, item]

			num = dot(notas, sims)
			den = sum(abs(sims))

			if den > 0:
				predito[j] = num/den

			if abs(gabarito[j] - predito[j])/4 > threshold:
				matrizIndices[usuario, item] = 1

		j = j + 1

	print()

	return matrizIndices


"""
@brief Trata o ruido detectado utilizando algoritmo de O'Mahony

@param matriz Matriz analisada
@param threshold Limiar de diferenca entre a nota original e a predita
@param k Numero de vizinhos mais proximos para o calculo da predicao
@param funcSimilaridade Funcao de similaridade do KNN

@return novaMatriz Matriz sem ruido

"""

def omahonyTratamento(matriz, threshold=.25, k=20, funcSimilaridade=pearson):

	print("Tratando ruidos com algoritmo OMahony\n")

	similaridades = funcSimilaridade(matriz)
	print()

	u, i = nonzero(matriz)

	gabarito = matriz[u, i]
	predito  = zeros(len(gabarito))

	novaMatriz = matriz.copy()

	j = 0
	for usuario, item in zip(u, i):

		print("\rCalculando predições: ", ((j+1)*100)//len(predito), "%", end="")

		vizinhos = nonzero(novaMatriz[:, item])[0]
		vizinhos = vizinhos[similaridades[usuario, vizinhos]>0.]

		if len(vizinhos) < (k+1):
			pass		
		else:
			knn 	= vizinhos[argsort(similaridades[usuario, vizinhos])[-(k+1):-1]]
			sims	= similaridades[usuario, knn]
			notas	= novaMatriz[knn, item]

			num = dot(notas, sims)
			den = sum(abs(sims))

			if den > 0:
				predito[j] = num/den

			if abs(gabarito[j] - predito[j])/4 > threshold:
				novaMatriz[usuario, item] = round(predito[j])

		j = j + 1

	print()

	return novaMatriz


"""
@brief Detecta o ruido utilizando algoritmo de Toledo

@param matriz Matriz ruidosa
@param userBased Verifica se a deteccao sera feita com base na classificacao dos usuarios ou dos itens

@return ruidos Indices dos ruidos detectados 
@return thresholds Limiares de diferenca utilizados para o tratamento do ruido

"""

def toledoDeteccao(matriz, userBased=True):

	print("Detectando ruido com algorito Toledo\n")

	classesUsuarios = [0 for i in range(matriz.shape[0])]
	classesItens 	= [0 for i in range(matriz.shape[1])]

	userSets = [[0, 0, 0] for i in range(matriz.shape[0])]
	itemSets = [[0, 0, 0] for i in range(matriz.shape[1])]

	ruidos = full(matriz.shape, -1, dtype=int)
	ruidos[matriz[:]>0] = 0
	
	if userBased:
		kvp = [[0, 0, 0] for i in range(matriz.shape[0])]
	
	else:
		kvp = [[0, 0, 0] for i in range(matriz.shape[1])]

	u, i = nonzero(matriz)	

	for usuario, item in zip(u, i):

		media  = mean(matriz[usuario, matriz[usuario, :] > .0])
		desvio = std(matriz[usuario, matriz[usuario, :] > .0])

		ku = media - desvio
		vu = media + desvio
		thru = desvio

		if matriz[usuario, item] < ku:
			userSets[usuario][0] += 1

		elif matriz[usuario, item] >= ku and matriz[usuario, item] < vu:
			userSets[usuario][1] += 1

		else:
			userSets[usuario][2] += 1

		media  = mean(matriz[matriz[:, item] > .0, item])
		desvio = std(matriz[matriz[:, item] > .0, item])

		ki = media - desvio
		vi = media + desvio
		thri = desvio

		if matriz[usuario, item] < ki:
			itemSets[item][0] += 1

		elif matriz[usuario, item] >= ki and matriz[usuario, item] < vi:
			itemSets[item][1] += 1

		else:
			itemSets[item][2] += 1

		if userBased:
			kvp[usuario] = [ku, vu, thru]

		else:
			kvp[item] = [ki, vi, thri]
	
	for usuario in range(len(userSets)):

		w = userSets[usuario][0]
		a = userSets[usuario][1]
		s = userSets[usuario][2]

		if w >= (a + s):
			classesUsuarios[usuario] = 1

		elif a >= (w + s):
			classesUsuarios[usuario] = 2

		elif s >= (w + a):
			classesUsuarios[usuario] = 3

	for item in range(len(itemSets)):

		w = itemSets[item][0]
		a = itemSets[item][1]
		s = itemSets[item][2]

		if w >= (a + s):
			classesItens[item] = 1

		elif a >= (w + s):
			classesItens[item] = 2

		elif s >= (w + a):
			classesItens[item] = 3

	j = 0
	
	for usuario, item in zip(u, i):

		print("\rAnalisando notas: ", ((j+1)*100)//len(u), "%", end="")

		if userBased:
			k   = kvp[usuario][0]
			v   = kvp[usuario][1]

		else:
			k   = kvp[item][0]
			v   = kvp[item][1]

		if (classesUsuarios[usuario] == 1) and (classesItens[item] == 1) and (matriz[usuario, item] >= k):
			ruidos[usuario, item] = True

		if (classesUsuarios[usuario] == 2) and (classesItens[item] == 2) and (matriz[usuario, item] < k or matriz[usuario, item] >= v):
			ruidos[usuario, item] = True

		if (classesUsuarios[usuario] == 3) and (classesItens[item] == 3) and (matriz[usuario, item] < v):
			ruidos[usuario, item] = True

		j = j + 1

	thresholds = [kvp[i][2] for i in range(len(kvp))]

	print()

	return (ruidos, thresholds)


"""
@brief Trata o ruido detectado utilizando algoritmo de Toledo

@param matriz Matriz ruidosa
@param ruidos Lista de possiveis ruidos
@param thresholds Limiares de diferenca entre as notas reais e as notas previstas
@param userBased Verifica se o tratamento vai ser feito com base na classificacao dos usuarios ou dos itens
@param k Numero de vizinhos mais proximos para o calculo da predicao
@param funcSimilaridade Funcao de similaridade do KNN. Default Pearson

@return novaMatriz Matriz sem ruido

"""

def toledoTratamento(matriz, ruidos, thresholds, userBased=True, k=20, funcSimilaridade=pearson):

	print("Tratando ruido com algoritmo Toledo\n")

	similaridades = funcSimilaridade(matriz)	
	print()
	
	novaMatriz = matriz.copy()

	u, i = nonzero(ruidos[:]>0) 

	gabarito = matriz[u, i]
	predito  = zeros(len(gabarito))

	j = 0
	for usuario, item in zip(u, i):

		print("\rCalculando predições: ", ((j+1)*100)//len(predito), "%", end="")

		vizinhos = nonzero(novaMatriz[:, item])[0]
		vizinhos = vizinhos[similaridades[usuario, vizinhos]>0.]

		if len(vizinhos) < (k+1):
			pass		
		else:
			knn 	= vizinhos[argsort(similaridades[usuario, vizinhos])[-(k+1):-1]]
			sims	= similaridades[usuario, knn]
			notas	= novaMatriz[knn, item]

			num = dot(notas, sims)
			den = sum(abs(sims))

			if den > 0:
				predito[j] = num/den

			if userBased:
				threshold = thresholds[usuario]
			else:
				threshold = thresholds[item]

			if abs(gabarito[j] - predito[j])/4 > threshold:
				novaMatriz[usuario, item] = round(predito[j])

		j = j + 1

	print()

	return novaMatriz


"""
@brief Realiza da predicao das notas de teste

@param matrizTreino Matriz de treino
@param matrizTeste Matriz de teste
@param k Numero de vizinhos mais proximos para o calculo da predicao. Default 20
@param funcSimilaridade Funcao de similaridade do KNN. Default Pearson

@return mae Media absoluta dos erros
@return rmse Media quadratica dos erros ao quadrado
@return cobertura Porcentagem de notas preditas em relacao ao tamanho do conjunto de testes

"""

def calcularPredicoes(matrizTreino, matrizTeste, k=20, funcSimilaridade=pearson):

	print("Calculando predicoes com KNN\n")
	
	similaridades = funcSimilaridade(matrizTreino)
	print()

	u, i = nonzero(matrizTeste)
	
	gabarito = matrizTeste[u, i]
	predito  = zeros(len(gabarito))

	j = 0
	for usuario, item in zip(u, i):

		print("\rCalculando predições: ", ((j+1)*100)//len(predito), "%", end="")

		vizinhos = nonzero(matrizTreino[:, item])[0]
		vizinhos = vizinhos[similaridades[usuario, vizinhos]>0.]

		if len(vizinhos) < (k+1):
			pass		
		else:
			knn 	= vizinhos[argsort(similaridades[usuario, vizinhos])[-(k+1):-1]]
			sims	= similaridades[usuario, knn]
			notas	= matrizTreino[knn, item]

			num = dot(notas, sims)
			den = sum(abs(sims))

			if den > 0:
				predito[j] = num/den

		j = j + 1

	mae = calcularMAE(gabarito, predito)
	rmse = calcularRMSE(gabarito, predito)
	cobertura = (count_nonzero(predito)/len(gabarito)) * 100

	print()

	return (mae, rmse, cobertura) 
