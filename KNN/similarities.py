from numpy import zeros, dot, sum as soma, mean, intersect1d, nonzero, count_nonzero

"""
@brief Calcula média da matriz esparsa
@param matriz Matriz esparsa de entrada
@return media Média da matriz

"""
def media_matriz(matriz):
	media = 0
	for i in range(matriz.shape[0]):
		for j in range(matriz.shape[1]):
			media += matriz[i, j]
	media /= matriz.nnz
	return media

"""
@brief Calcula a média do usuário usando shrinkage

Dadas as notas do usuário e a média global, calcula a média do usuário proporcial ao shrinkage

@param notas Notas do usuário
@param mu Média global
@param shrinkage Shrinkage
@return Média do usuário

"""
def media_usuario(notas, mu, shrinkage=30):
	ru = count_nonzero(notas)
	b = sum(notas) / ru
	dem = shrinkage + ru
	return (shrinkage/dem)*mu + (ru/dem)*b

"""
@brief Calcula a similaridade entre os usuários usando cosseno

Calcula a similaridade entre os usuários através do cálculo do cosseno.
Dados dois usuários, u e v, o cálculo é feito considerando apenas os itens avaliados por ambos os usuários. 

@param matriz Matriz usuário-item
@return similaridades Matriz de similaridade usuário-usuário

"""
def cosseno(matriz):
	similaridades = zeros((matriz.shape[0], matriz.shape[0]))
	
	for u in range(matriz.shape[0]):

		print("\rCalculando similaridades: ", ((u+1)*100)//matriz.shape[0], "%", end="")

		for v in range(u+1):

			cos = 0.

			iu = nonzero(matriz[u,:])[1] # índices dos itens avaliados pelo usúario u
			iv = nonzero(matriz[v,:])[1] # índices dos itens avaliados pelo usuário v
			
			intersecao = intersect1d(iu, iv) # índices das itens avaliados por u e v

			nu = matriz[u, intersecao].toarray()[0] # notas dos itens avaliados por u e v
			nv = matriz[v, intersecao].toarray()[0] # notas dos itens avaliados por v e u

			num = soma(nu * nv)
			dem = (soma(nu**2) * soma(nv**2))**.5

			if dem > 0.:
				cos = num/dem

			similaridades[u, v] = cos
			similaridades[v, u] = cos
	
	return similaridades

def r_dice(matriz, limiar=1):

	similaridades = zeros((matriz.shape[0], matriz.shape[0]))

	for u in range(matriz.shape[0]):

		print("\rCalculando similaridades: ", ((u+1)*100)//matriz.shape[0], "%", end="")
		
		for v in range(u+1):

			rd = 0.

			iu = nonzero(matriz[u,:])[1] # índices dos itens avaliados pelo usúario u
			iv = nonzero(matriz[v,:])[1] # índices dos itens avaliados pelo usuário v
			
			intersecao = intersect1d(iu, iv) # índices das itens avaliados por u e v

			nu = matriz[u, intersecao].toarray()[0] # notas dos itens avaliados por u e v
			nv = matriz[v, intersecao].toarray()[0] # notas dos itens avaliados por v e u

			num = len([i for i in range(len(intersecao)) if abs(nu[i] - nv[i]) <= limiar]) 

			if len(intersecao) > 0:
				rd = num/len(intersecao)

			similaridades[u, v] = rd
			similaridades[v, u] = rd

	return similaridades

def c_dice(matriz, limiar=1):

	similaridades = zeros((matriz.shape[0], matriz.shape[0]))

	for u in range(matriz.shape[0]):

		# print("\r", u+1, " de ", matriz.shape[0], end="")
		print("\rCalculando similaridades: ", ((u+1)*100)//matriz.shape[0], "%", end="")

		for v in range(u + 1):

			cd = 0.

			iu = nonzero(matriz[u, :])[1] # índice dos itens avaliados por u
			iv = nonzero(matriz[v, :])[1] # índice dos itens avaliados por v	

			intersecao = intersect1d(iu, iv) # índices das itens avaliados por u e v

			nu = matriz[u, intersecao].toarray()[0] # notas dos itens avaliados por u e v
			nv = matriz[v, intersecao].toarray()[0] # notas dos itens avaliados por v e u

			num = len([i for i in range(len(intersecao)) if abs(nu[i] - nv[i]) <= limiar])

			if len(intersecao) > 0:
				cd = (2.0 * num) / (nu.shape[0] + nv.shape[0])

			similaridades[u, v] = cd
			similaridades[v, u] = cd
		
	return similaridades


"""
@brief Calcula a similaridade de Pearson entre os usuários 
@param matriz Matriz usuário-item
@return similaridades Matriz de similaridade usuário-usuário

"""
def pearson(matriz):
	similaridades = zeros((matriz.shape[0], matriz.shape[0]))

	mg = media_matriz(matriz) # média global

	for u in range(matriz.shape[0]):

		print("\rCalculando similaridades: ", ((u+1)*100)//matriz.shape[0], "%", end="")

		for v in range(u+1):

			pearson = 0.

			iu = nonzero(matriz[u, :])[1] # índice dos itens avaliados por u
			iv = nonzero(matriz[v, :])[1] # índice dos itens avaliados por v

			nu = matriz[u, :].toarray()[0] # notas de u
			nv = matriz[v, :].toarray()[0] # notas de v			

			intersecao = intersect1d(iu, iv) # índices da interseção

			if len(intersecao) > 0:
				nu = nu[intersecao] # notas de u de itens em comum com v
				nv = nv[intersecao] # notas de v de itens em comum com u

				# mu = soma(nu)/len(nu) # média de u
				# mv = soma(nv)/len(nv) # média de v

				mu = media_usuario(nu, mg)
				mv = media_usuario(nv, mg)

				num = soma((nu - mu) * (nv - mv))
				dem = (soma((nu - mu)**2) * soma((nv - mv)**2))**.5

				if dem > 0.:
					pearson = num/dem

				similaridades[u, v] = pearson
				similaridades[v, u] = pearson
	
	return similaridades


def mean_squared_difference(matriz):

	similaridades = zeros((matriz.shape[0], matriz.shape[0]))

	for u in range(matriz.shape[0]):

		print("\rCalculando similaridades: ", ((u+1)*100)//matriz.shape[0], "%", end="")

		for v in range(u+1):

			msd = 0.

			iu = nonzero(matriz[u, :])[1] # índice dos itens avaliados por u
			iv = nonzero(matriz[v, :])[1] # índice dos itens avaliados por v

			nu = matriz[u, :].toarray()[0] # notas de u
			nv = matriz[v, :].toarray()[0] # notas de v

			intersecao = intersect1d(iu, iv) # índices da interseção

			nu = nu[intersecao] # notas de u de itens em comum com v
			nv = nv[intersecao] # notas de v de itens em comum com u

			num = len(intersecao)
			dem = soma((nu - nv)**2)

			if dem > 0.:
				msd = num/dem

			similaridades[u, v] = msd
			similaridades[v, u] = msd

	return similaridades
