#-*-coding:utf-8-*-

from cross_validation import treino_teste_split, calcular_mae, calcular_rmse, treino_teste_split2
from similarities import cosseno, cosseno_intersecao, pearson, sc_dice
from scipy.sparse import lil_matrix as esparsa
from numpy import nonzero, argsort, dot, zeros, count_nonzero, std
from sklearn.metrics.pairwise import cosine_similarity as cosine
import sys, os.path

path = "ml-100k/"
u_user = path + "u.user"
u_item = path + "u.item"
u_data = path + "u.data"

"""
@brief Cria os dados que alimentam o knn

Dados os conjuntos de dados, gera os dados de treino e teste necessários para o knn.

@return matriz_treino Matriz esparsa (usuário-item) de treino
@return dados_treino Dados do conjunto de treino
@return matriz_teste Matriz esparsa (usuário-item) de teste
@return dados_teste Dados do conjunto de teste
"""
def criar_dataset():

	treino, teste = treino_teste_split(u_data)

	with open(u_user, encoding="utf-8") as a:
		usuarios = a.readlines()	

	with open(u_item, encoding="utf-8") as a:
		itens = a.readlines()

	with open(u_data, encoding="utf-8") as a:
		dados = a.readlines()

	matriz_treino = esparsa((len(usuarios), len(itens)))
	matriz_teste = esparsa((len(usuarios), len(itens)))

	dados_treino = []	
	for linha in treino:
		tokens = linha.split()

		usuario = int(tokens[0]) - 1
		item 	= int(tokens[1]) - 1
		nota 	= int(tokens[2])

		matriz_treino[usuario, item] = nota

		dados_treino.append([usuario, item, nota])

	dados_teste = []
	for linha in teste:
		tokens = linha.split()

		usuario = int(tokens[0]) - 1
		item 	= int(tokens[1]) - 1
		nota 	= int(tokens[2])

		matriz_teste[usuario, item] = nota

		dados_teste.append([usuario, item, nota])

	return (matriz_treino, dados_treino, matriz_teste, dados_teste)

"""
@brief Escreve os dados da execução do algoritmo
@param arquivo Arquivo de saída
@param k Número de vizinhos da execução
@param mae Média absoluta dos erros
@param rmse Média quadrática dos erros
@param predicoes Número de predições executadas

"""
def escrever_estatisticas(arquivo, k, mae, rmse, predicoes):
	
	if not os.path.exists(arquivo):
		output = open(arquivo, "a")
		output.write("k, MAE, RMSE, PREDICOES\n")
		output.close()

	with open(arquivo, "a") as output:
		output.write("%d, %.4f, %.4f, %.2f\n" %(k, mae, rmse, predicoes))

"""
@brief Executa o classificador KNN

Realiza a predição das notas segundo o algoritmo KNN.
A função de similaridade e o número de vizinhos são dados como entrada.

@param func_similaridade Função de similaridade. Default cosine
@param min_k Menor número de vizinhos mais próximos. Default 1
@param max_k Maior número de vizinhos mais próximos. Default 5
@param intervalo Taxa de acréscimo do número de vizinhos
@param arquivo_saida Arquivo de saída contendo os dados da execução do algoritmo
@return Média absoluta dos erros

"""
def classificar(func_similaridade=cosseno, min_k=1, max_k=5, acrescimo=1, arquivo_saida="saida.txt", **parametros):

	matriz_treino, dados_treino, matriz_teste, dados_teste = criar_dataset() 

	similaridades = func_similaridade(matriz_treino, **parametros)
	
	gabarito = [dado[2] for dado in dados_teste]

	print("")

	for k in range(min_k, max_k+1, acrescimo):
		
		print("Calculando para k =", k)

		predito = zeros(len(dados_teste))

		for i in range(len(dados_teste)):

			print("\rCalculando predições: ", ((i+1)*100)//len(dados_teste), "%", end="")

			usuario, item = dados_teste[i][0], dados_teste[i][1]
			vizinhos = nonzero(matriz_treino[:, item])[0] 
			vizinhos = vizinhos[similaridades[usuario, vizinhos]>0.]

			if len(vizinhos) < k+1:
				pass
			else:
				knn 		= vizinhos[argsort(similaridades[usuario, vizinhos])[-(k+1):-1]]
				knn_sims 	= similaridades[usuario, knn]
				notas 		= matriz_treino[knn, item].toarray()

				num = dot(notas.T, knn_sims)
				den = sum(abs(knn_sims))

				if den > 0:
					predito[i] = num/den 

		print("\nCalculando média dos erros...")
		escrever_estatisticas(arquivo_saida, k, calcular_mae(gabarito, predito), calcular_rmse(gabarito, predito), (count_nonzero(predito)/len(gabarito))*100)

from random import shuffle

if __name__ == "__main__":

	dividir_base(u_data, divisoes = 5)

	# classificar(func_similaridade=sc_dice, min_k=60, max_k=60,
	# 			acrescimo=10, arquivo_saida="tests/saida.txt", limiar=1)

	# dados = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
	# divisoes = 5

	# dados_divididos = []

	# tamanho = len(dados)
	# tamanho_parte = tamanho/ divisoes

	# shuffle(dados)

	# for i in range(divisoes):
	# 	dados_divididos.append( dados[  int(i*tamanho_parte) : int((i+1)*tamanho_parte)] )

	# print(dados_divididos)

	# #########################################
	# ############## Testes ###################
	# #########################################
	# similaridades = [cosseno_intersecao, pearson, cosseno, sc_dice]
	# saidas = ["cosseno_int.csv", "pearson.csv", "cosseno.csv", "sc_dice.csv"]
	
	# parametros = [ {}, {}, {}, { 'limiar': 1 } ]
	
	# # Realiza os testes para todos as métricas de similaridade
	# for i in range(len(similaridades)):
		
	# 	print("Calculando para similaridade %s" % (str(similaridades[i])))
		
	# 	classificar(func_similaridade=similaridades[i], min_k=10, max_k=100,
	# 				acrescimo=10, arquivo_saida="tests/"+saidas[i], **parametros[i])

	# # Realiza os testes dos limiares para a métrica SC-Dice
	# for i in range(5):
		
	# 	print("Calculando para limiar = " , i)
		
	# 	classificar(func_similaridade=sc_dice, min_k=60, max_k=60,
	# 				acrescimo=10, arquivo_saida="tests/scdice_"+str(i)+".csv", limiar=i)