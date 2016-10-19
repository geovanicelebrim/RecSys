from cross_validation import treino_teste_split, calcular_mae
from similarities import cosseno, pearson, mean_squared_difference, c_dice, r_dice
from scipy.sparse import lil_matrix as esparsa
from sklearn.metrics.pairwise import cosine_similarity as cosine
from numpy import nonzero, argsort, dot, zeros

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
@brief Executa o classificador KNN

Realiza a predição das notas segundo o algoritmo KNN.
A função de similaridade e o número de vizinhos são dados como entrada.

@param func_similaridade Função de similaridade. Default cosine
@param k Número de vizinhos mais próximos. Default 1
@return Média absoluta dos erros

"""
def classificar(func_similaridade=cosine, k=1,**parametros):
	matriz_treino, dados_treino, matriz_teste, dados_teste = criar_dataset() 

	similaridades = func_similaridade(matriz_treino, **parametros)

	predito = zeros(len(dados_teste))
	gabarito = [dado[2] for dado in dados_teste]

	print("")

	for i in range(len(dados_teste)):

		print("\rCalculando predições: ", ((i+1)*100)//len(dados_teste), "%", end="")

		usuario, item, nota = dados_teste[i][0], dados_teste[i][1], dados_teste[i][2]
		vizinhos = nonzero(matriz_treino[:, item])[0]

		if len(vizinhos) < k+1:
			pass
		else:
			knn 		= vizinhos[argsort(similaridades[usuario, vizinhos])[-(k+1):-1]]
			knn_sims 	= similaridades[usuario, knn]
			notas 		= matriz_treino[knn, item].toarray()

			num = dot(notas.T, knn_sims)
			dem = sum(abs(knn_sims))

			if dem > 0:
				predito[i] = num/dem 

	print("\nCalculando média dos erros...")
	return calcular_mae(gabarito, predito)

def main():
	# print("MAE %.4f" %classificar(func_similaridade=mean_squared_difference, k=60))
	# print("MAE %.4f" %classificar(func_similaridade=c_dice, k=60, limiar=1))
	print("MAE %.4f" %classificar(func_similaridade=r_dice, k=60, limiar=1))
	# classificar(func_similaridade=cosseno)

if __name__ == "__main__":
	main()