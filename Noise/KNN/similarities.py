from numpy import full, zeros, sum as soma, mean, intersect1d, nonzero, count_nonzero
from sklearn.metrics.pairwise import cosine_similarity as cosine

"""
@brief Calcula média da matriz esparsa
@param matriz Matriz esparsa de entrada
@return media Média da matriz

"""
def mediaMatriz(matriz):
	return mean(matriz[matriz>0])

"""
@brief Calcula a média do usuário usando shrinkage

Dadas as notas do usuário e a média global, calcula a média do usuário proporcial ao shrinkage

@param notas Notas do usuário
@param mu Média global
@param shrinkage Shrinkage
@return Média do usuário

"""
def mediaUsuario(notas, mu, shrinkage=10):
	ru = count_nonzero(notas)
	b = sum(notas) / ru
	dem = shrinkage + ru
	return (shrinkage/dem)*mu + (ru/dem)*b

"""
@brief Calcula a similaridade usando cosseno

Dada a matriz usuário-item, calcula a similaridade entre dois usuários usando cosseno.
A média global é adiciona aos casos em que não há avaliação do usuário para um item. 

@param matriz Matriz usuário-item
@return Similaridade calculada
"""
def cosseno(matriz):
	mu = mediaMatriz(matriz)
	copia = full(matriz.shape, mu) 
	linhas, colunas = matriz.nonzero() 

	for l, c in zip(linhas, colunas):
		copia[l, c] = matriz[l, c]

	return cosine(copia)


"""
@brief Calcula a similaridade entre os usuários usando cosseno

Calcula a similaridade entre os usuários através do cálculo do cosseno.
Dados dois usuários, u e v, o cálculo é feito considerando apenas os itens avaliados por ambos os usuários. 

@param matriz Matriz usuário-item
@return similaridades Matriz de similaridade usuário-usuário

"""
def cossenoIntersecao(matriz):
	similaridades = zeros((matriz.shape[0], matriz.shape[0]))
	
	for u in range(matriz.shape[0]):

		print("\rCalculando similaridades: ", ((u+1)*100)//matriz.shape[0], "%", end="")

		for v in range(u+1):

			cos = 0.

			iu = nonzero(matriz[u,:]) # índices dos itens avaliados pelo usúario u
			iv = nonzero(matriz[v,:]) # índices dos itens avaliados pelo usuário v
			
			intersecao = intersect1d(iu, iv) # índices das itens avaliados por u e v

			nu = matriz[u, iu] # notas de u
			nv = matriz[v, iv] # notas de v	

			den = (soma(nu**2) * soma(nv**2))**.5 # considera todas as notas de u e todas as notas de v

			if len(intersecao) > 10:
				nu = matriz[u, intersecao] # notas dos itens avaliados por u e v
				nv = matriz[v, intersecao] # notas dos itens avaliados por v e u

				num = soma(nu * nv)				

				if den > 0.:
					cos = num/den

				similaridades[u, v] = cos
				similaridades[v, u] = cos
	
	return similaridades

"""
@brief Calcula a similaridade de Pearson entre os usuários 
@param matriz Matriz usuário-item
@return similaridades Matriz de similaridade usuário-usuário

"""
def pearson(matriz):
	similaridades = zeros((matriz.shape[0], matriz.shape[0]))

	mg = mediaMatriz(matriz)

	for u in range(matriz.shape[0]):

		print("\rCalculando similaridades: ", ((u+1)*100)//matriz.shape[0], "%", end="")

		for v in range(u+1):

			pearson = 0.

			iu = nonzero(matriz[u, :]) # índice dos itens avaliados por u
			iv = nonzero(matriz[v, :]) # índice dos itens avaliados por v

			nu = matriz[u, :] # notas de u
			nv = matriz[v, :] # notas de v	

			intersecao = intersect1d(iu, iv) # índice dos itens em comum

			if len(intersecao) > 10:
				nu = nu[intersecao] # notas de u de itens em comum com v
				nv = nv[intersecao] # notas de v de itens em comum com u

				mu = mediaUsuario(nu, mg)
				mv = mediaUsuario(nv, mg)

				num = soma((nu - mu) * (nv - mv))
				den = (soma((nu - mu)**2) * soma((nv - mv)**2))**.5

				if den > 0.:
					pearson = num/den

				similaridades[u, v] = pearson
				similaridades[v, u] = pearson
	
	return similaridades

"""
@brief Calcula o inverso da diferença quadrática

Dada a matriz usuário-item, calcula a similaridade entre dois ussuários
calculando a o inverso da média dos quadrados da diferença entre as notas da interseção.

@param matriz Matriz usuário-item
@return similaridades Matriz de similaridades

"""
def meanSquaredDifference(matriz):

	similaridades = zeros((matriz.shape[0], matriz.shape[0]))

	for u in range(matriz.shape[0]):

		print("\rCalculando similaridades: ", ((u+1)*100)//matriz.shape[0], "%", end="")

		for v in range(u+1):

			msd = 0.

			iu = nonzero(matriz[u, :]) # índice dos itens avaliados por u
			iv = nonzero(matriz[v, :]) # índice dos itens avaliados por v

			nu = matriz[u, :] # notas de u
			nv = matriz[v, :] # notas de v

			intersecao = intersect1d(iu, iv) # índices da interseção

			if len(intersecao) > 0:
				nu = nu[intersecao] # notas de u de itens em comum com v
				nv = nv[intersecao] # notas de v de itens em comum com u

				num = len(intersecao)
				den = soma((nu - nv)**2)

				if den > 0.:
					msd = num/den

				similaridades[u, v] = msd
				similaridades[v, u] = msd

	return similaridades

"""
@brief Calcula a similaridade Silva-Oliveira

Dada a matriz de usuário-item, calcula a similaridade entre dois usuários 
considerando a porcentagem de notas da interseção cuja diferença absoluta seja menor ou igual a um limiar.

@param matriz Matriz usuário-item
@param limiar Limiar da diferença absoluta das notas. Default 1. 
@return similaridades Matriz de similaridades.

"""
def silvaOliveira(matriz, limiar=1):

	similaridades = zeros((matriz.shape[0], matriz.shape[0]))

	for u in range(matriz.shape[0]):

		print("\rCalculando similaridades: ", ((u+1)*100)//matriz.shape[0], "%", end="")
		
		for v in range(u+1):

			rd = 0.

			iu = nonzero(matriz[u,:]) # índices dos itens avaliados pelo usúario u
			iv = nonzero(matriz[v,:]) # índices dos itens avaliados pelo usuário v
			
			intersecao = intersect1d(iu, iv) # índices das itens avaliados por u e v

			nu = matriz[u, intersecao] # notas dos itens avaliados por u e v
			nv = matriz[v, intersecao] # notas dos itens avaliados por v e u

			num = len([i for i in range(len(intersecao)) if abs(nu[i] - nv[i]) <= limiar]) 

			if len(intersecao) > 10:
				rd = num/len(intersecao)

			similaridades[u, v] = rd
			similaridades[v, u] = rd

	return similaridades
	