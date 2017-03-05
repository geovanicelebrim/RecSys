import matplotlib.pyplot as plt

import numpy as np

from scipy.sparse import lil_matrix

from sklearn.metrics.pairwise import cosine_similarity as cosine



path = "ml-100k/"

u_user = path + "u.user"

u_item = path + "u.item"

u_data = path + "u.data"



def gerar_matriz_usuario_item(usuarios, itens, avaliacoes):

	mui = lil_matrix((len(usuarios), len(itens)))

	

	for avaliacao in range(len(avaliacoes)):

		usuario, item, nota, tempo = avaliacoes[avaliacao].strip().split("\t")

		linha = int(usuario) - 1

		coluna = int(item) - 1

		mui[linha, coluna] = int(nota)



	return mui



def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        plt.text(rect.get_x() + rect.get_width()/2., height, '%d' % height, ha='center', va='bottom') 



def treino_teste_split(dados, mui, porcentagem_treino=.8):

	tamanho = dados.shape[0]

	np.random.shuffle(dados)

	treino, teste = dados[:tamanho*porcentagem_treino], dados[tamanho*porcentagem_treino:]



	mtreino = lil_matrix((mui.shape[0], mui.shape[1]), dtype="int")

	mteste = lil_matrix((mui.shape[0], mui.shape[1]), dtype="int")



	for t in treino:

		linha, coluna = t[0], t[1]

		mtreino[linha, coluna] = mui[linha, coluna]



	for t in teste:

		linha, coluna = t[0], t[1]

		mteste[linha, coluna] = mui[linha, coluna]



	return (mtreino, treino, mteste, teste)



def calcular_media_sparse(matriz):

	media = 0

	for i in range(matriz.shape[0]):

		for j in range(matriz.shape[1]):

			media += matriz[i, j]

	media /= matriz.nnz

	return media



def ex01(mui):

	histograma = np.zeros(mui.shape[0], dtype="int")

	for linha in range(mui.shape[0]):

		histograma[linha] = mui[linha,:].nnz



	histograma = sorted(histograma)



	plt.xlabel("Usuários")

	plt.ylabel("Quantidade de Avaliaçoes")

	plt.title("Histograma Avaliações")

	plt.xlim([0, len(histograma)])

	plt.plot(histograma, '.')

	plt.show()



	return histograma



def ex02(mui):

	N = 6

	histograma = np.zeros(6, dtype="int")

	for linha in range(mui.shape[0]):

		for coluna in range(mui.shape[1]):

			nota = mui[linha, coluna]

			histograma[nota] += 1



	histograma = histograma[1:]



	labels = ["1", "2", "3", "4", "5"]



	width = .5

	ind = np.arange(len(labels)) 

	barchart = plt.bar(ind, histograma, width, color="royalblue")

	plt.xlabel("Notas")

	plt.ylabel("Frequências")

	plt.title("Histograma de Notas")

	plt.xticks(ind + width/2., labels)

	plt.ylim(0, max(histograma)+10000)

	autolabel(barchart)

	plt.show()



	return histograma



def ex03(mui, avaliacoes):

	m = np.zeros((len(avaliacoes), 3), dtype="int") # matriz de avaliacoes

	for a in range(len(avaliacoes)):

		user, item, rating, timestamp = avaliacoes[a].strip().split()

		m[a, 0] = int(user) - 1

		m[a, 1] = int(item) - 1

		m[a, 2] = int(rating)



	mtreino, treino, mteste, teste = treino_teste_split(m, mui)



	media_global = calcular_media_sparse(mtreino)



	media_usuarios = np.full((mui.shape[0]), media_global) 



	for i in range(mui.shape[0]):

		media_usuarios[i] = lil_matrix.sum(mtreino[i,:]) / mtreino[i,:].nnz



	predito = np.zeros((teste.shape[0]))



	for i in range(teste.shape[0]):

		user = teste[i,0]

		predito[i] = media_usuarios[user]



	gabarito = teste[:, 2]



	mae = 1/len(predito) * sum(abs(predito-gabarito))



	print(mae)



def ex04(mui, avaliacoes):

	m = np.zeros((len(avaliacoes), 3), dtype="int") # matriz de avaliacoes

	for a in range(len(avaliacoes)):

		user, item, rating, timestamp = avaliacoes[a].strip().split()

		m[a, 0] = int(user) - 1

		m[a, 1] = int(item) - 1

		m[a, 2] = int(rating)



	mtreino, treino, mteste, teste = treino_teste_split(m, mui)



	media_global = calcular_media_sparse(mtreino)



	media_itens = np.full((mui.shape[1]), media_global) 



	for i in range(mui.shape[1]):

		if not mtreino[:,i].nnz == 0:

			media_itens[i] = lil_matrix.sum(mtreino[:,i]) / mtreino[:,i].nnz



	predito = np.zeros((teste.shape[0]))



	for i in range(teste.shape[0]):

		item = teste[i,1]

		predito[i] = media_itens[item]



	gabarito = teste[:, 2]



	mae = 1/len(predito) * sum(abs(predito-gabarito))



	print(mae)



def ex05(mui, avaliacoes):

	m = np.zeros((len(avaliacoes), 3), dtype="int") # matriz de avaliacoes

	for a in range(len(avaliacoes)):

		user, item, rating, timestamp = avaliacoes[a].strip().split()

		m[a, 0] = int(user) - 1

		m[a, 1] = int(item) - 1

		m[a, 2] = int(rating)



	mtreino, treino, mteste, teste = treino_teste_split(m, mui)



	media_global = calcular_media_sparse(mtreino)



	media_usuarios = np.full((mui.shape[0]), media_global) 



	for i in range(mui.shape[0]):

		media_usuarios[i] = lil_matrix.sum(mtreino[i,:]) / mtreino[i,:].nnz



	media_itens = np.full((mui.shape[1]), media_global) 



	for i in range(mui.shape[1]):

		if not mtreino[:,i].nnz == 0:

			media_itens[i] = lil_matrix.sum(mtreino[:,i]) / mtreino[:,i].nnz



	predito = np.zeros((teste.shape[0]))



	for i in range(teste.shape[0]):

		usuario, item = teste[i,0], teste[i,1]

		predito[i] = ((media_usuarios[usuario]*mtreino[usuario, :].nnz) + (media_itens[item]*mtreino[:,item].nnz)) / (mtreino[usuario, :].nnz + mtreino[:,item].nnz)



	gabarito = teste[:, 2]



	mae = 1/len(predito) * sum(abs(predito-gabarito))



	print(mae)



def main():

	with open(u_user) as a:

		usuarios = a.readlines()	



	with open(u_item, encoding="utf-8") as a:

		itens = a.readlines()



	with open(u_data) as a:

		avaliacoes = a.readlines()



	mui = gerar_matriz_usuario_item(usuarios, itens, avaliacoes)



	# ex01(mui)

	# ex02(mui)

	# ex03(mui, avaliacoes)

	# ex04(mui, avaliacoes)

	ex05(mui, avaliacoes)



if __name__ == "__main__":

	main()