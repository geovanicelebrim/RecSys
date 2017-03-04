from cross_validation import *
from knn import *
from sklearn.cross_validation import KFold

"""
Analise do threshold do algoritmo O'Mahony

"""
def exp01():

	kf = KFold(len(dados), n_folds=5, shuffle=True)
		
	resultados = array([[ (th+1) * .125, 0., 0. ] for th in range(8)])

	for treino, teste in kf:

		matrizTreino, matrizTeste = criarDataset(dados, usuarios, itens, treino, teste)

		for i in range(8):

			th = (i+1) * .125

			novaMatriz = omahonyTratamento(matrizTreino, threshold=th, k=60)

			mae, rmse, cobertura = calcularPredicoes(novaMatriz, matrizTeste, k=60)

			resultados[i, 1] += mae
			resultados[i, 2] += rmse

	resultados[:, 1:] = resultados[:, 1:]/5

	savetxt("analiseThreshold.csv", resultados, fmt="%.4f", delimiter="\t", 
		header="threshold\tmae\trmse", comments="")

"""
Comparacao entre O'Mahony, Toledo UserBased e Toledo ItemBased

"""
def exp02():

	kf = KFold(len(dados), n_folds=5, shuffle=True)

	resultados = zeros((4, 2), dtype=float)

	for treino, teste in kf:

		matrizTreino, matrizTeste = criarDataset(dados, usuarios, itens, treino, teste)

		mae, rmse, cobertura = calcularPredicoes(matrizTreino, matrizTeste, k=60)

		resultados[0, 0] += mae
		resultados[0, 1] += rmse

		""" Executa O'Mahony """

		novaMatriz = omahonyTratamento(matrizTreino, threshold=.5, k=60)

		mae, rmse, cobertura = calcularPredicoes(novaMatriz, matrizTeste, k=60)

		resultados[1, 0] += mae
		resultados[1, 1] += rmse

		""" Executa Toledo User-Based """

		possiveisRuidos, thresholds = toledoDeteccao(matrizTreino, userBased=True)

		novaMatriz = toledoTratamento(matrizTreino, possiveisRuidos, thresholds, userBased=True, k=60)

		mae, rmse, cobertura = calcularPredicoes(novaMatriz, matrizTeste, k=60)

		resultados[2, 0] += mae
		resultados[2, 1] += rmse

		""" Executa Toledo Item-Based """

		possiveisRuidos, thresholds = toledoDeteccao(matrizTreino, userBased=False)

		novaMatriz = toledoTratamento(matrizTreino, possiveisRuidos, thresholds, userBased=False, k=60)

		mae, rmse, cobertura = calcularPredicoes(novaMatriz, matrizTeste, k=60)

		resultados[3, 0] += mae
		resultados[3, 1] += rmse

	""" Realiza a media dos resultados """

	resultados = resultados/5

	savetxt("comparacaoModelos.csv", resultados, fmt="%.4f", delimiter="\t", 
		header="mae\trmse", comments="")

"""
Comparacao entre precision, recall e f1 de cada modelo com uma base perturbada segundo uma porcentagem

"""
def exp03():

	kf = KFold(len(dados), n_folds=5, shuffle=True)

	resultadosPrecision	= array([[ (porcentagem + 1) * .05, 0., 0., 0.] for porcentagem in range(5)])
	resultadosRecall 	= array([[ (porcentagem + 1) * .05, 0., 0., 0.] for porcentagem in range(5)])
	resultadosF1 		= array([[ (porcentagem + 1) * .05, 0., 0., 0.] for porcentagem in range(5)])

	for treino, teste in kf:

		matrizTreino, matrizTeste = criarDataset(dados, usuarios, itens, treino, teste)

		for i in range(5):

			porcentagem = (i + 1) * .05

			matrizPerturbada, indicesRuidos = perturbarBase(matrizTreino, porcentagem=porcentagem)

			""" Analise O'Mahony """

			ruidosDetectados = omahonyDeteccao(matrizPerturbada, k=60)

			precision, recall, f1 = calcularF1Score(indicesRuidos, ruidosDetectados)

			resultadosPrecision[i, 1] += precision
			resultadosRecall[i, 1] += recall
			resultadosF1[i, 1] += f1

			""" Analise Toledo User-Based """

			possiveisRuidos, thresholds = toledoDeteccao(matrizPerturbada)

			precision, recall, f1 = calcularF1Score(indicesRuidos, possiveisRuidos)

			resultadosPrecision[i, 2] += precision
			resultadosRecall[i, 2] += recall
			resultadosF1[i, 2] += f1

			""" Analise Toledo Item-Based """

			possiveisRuidos, thresholds = toledoDeteccao(matrizPerturbada, userBased=False)

			precision, recall, f1 = calcularF1Score(indicesRuidos, possiveisRuidos)

			resultadosPrecision[i, 3] += precision
			resultadosRecall[i, 3] += recall
			resultadosF1[i, 3] += f1

	""" Realiza media dos resultados """

	resultadosPrecision[:, 1:] 	= resultadosPrecision[:, 1:]/5
	resultadosRecall[:, 1:] 	= resultadosRecall[:, 1:]/5
	resultadosF1[:, 1:] 		= resultadosF1[:, 1:]/5

	savetxt("precision-memory-based.csv", resultadosPrecision, fmt="%.4f", delimiter="\t", 
		header="porcentagem ruido\tomahony\tuser-based\titem-based", comments="")

	savetxt("recall-memory-based.csv", resultadosRecall, fmt="%.4f", delimiter="\t", 
		header="porcentagem ruido\tomahony\tuser-based\titem-based", comments="")

	savetxt("f1-score-memory-based.csv", resultadosF1, fmt="%.4f", delimiter="\t", 
		header="porcentagem ruido\tomahony\tuser-based\titem-based", comments="")


if __name__ == "__main__":
	
	path = "ml-100k/"
	u_user = path + "u.user"
	u_item = path + "u.item"
	u_data = path + "u.data"

	with open(u_user, encoding="utf-8") as a:
		usuarios = a.readlines()	

	with open(u_item, encoding="utf-8") as a:
		itens = a.readlines()

	with open(u_data, encoding="utf-8") as a:
		dados = a.readlines()

	exp01()
	exp02()
	exp03() 