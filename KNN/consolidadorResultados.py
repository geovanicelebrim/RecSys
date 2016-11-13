import os.path
import csv

pasta = "tests"
divisoes = 5

def main():

	global pasta
	global divisoes
	
	# Consolidar os testes das similaridades
	nomes_similaridades = ["cosseno_int", "pearson", "cosseno", "sc_dice"]

	# Consolidar os dados dos testes de limiar do sc_dice
	# nomes_similaridades = [ "sc_dice_limiar_%d" % (i) for i in range(divisoes) ]

	for similaridade in nomes_similaridades:

		resultados_por_execucao = []
		nome_arquivo_consolidado = "%s/%s_consolidado.csv" % (pasta, similaridade)

		for i in range(divisoes):

			nome_arquivo = "%s/%s_%d.csv" % (pasta, similaridade, i)

			if os.path.exists(nome_arquivo):

				arquivo = open(nome_arquivo, "r")

				resultados_por_execucao.append( list(csv.reader(arquivo, delimiter=','))[1:] )

				arquivo.close()

			else:
				print("Arquivo não encontrado: ", nome_arquivo)
				exit()

		ks = len(resultados_por_execucao[0])

		resultados_por_k = []

		for k in range(ks):

			resultados_por_k.append( [] )

			for divisao in range(divisoes):
				resultados_por_k[k].append(resultados_por_execucao[divisao][k])


		arquivo_consolidado = open(nome_arquivo_consolidado, "a")
		arquivo_consolidado.write("k, MAE, RMSE, PREDICOES\n")

		for resultado in resultados_por_k:

			k = int(resultado[0][0])
			MAEs = [ float(m[1]) for m in resultado ]
			RMSEs = [ float(m[2]) for m in resultado ]
			PREDICOESs =[ float(m[3]) for m in resultado ]

			MAE = sum(MAEs) / len(MAEs)
			RMSE = sum(RMSEs) / len(RMSEs)
			PREDICOES = sum(PREDICOESs) / len(PREDICOESs)


			arquivo_consolidado.write("%d, %.4f, %.4f, %.2f\n" % (k, MAE, RMSE, PREDICOES) )

		arquivo_consolidado.close()


	pass

if __name__ == '__main__':
	main()