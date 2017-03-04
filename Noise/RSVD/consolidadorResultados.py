import os.path
import csv

pasta = "tests"
divisoes = 5

def main():

	global pasta
	global divisoes
	
	# Consolidar os testes
	nomes_similaridades = ["test_lambda", "test_lrate", "test_k"]
	nomes = [ ["lambda", float], ["lrate", float], ["k", int] ]

	for i_similaridade in range(len(nomes_similaridades)):

		similaridade = nomes_similaridades[i_similaridade]

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

		arquivo_consolidado.write("iteracoes, MAE, RMSE, %s\n" % (nomes[i_similaridade][0]))

		for resultado in resultados_por_k:

			valor = resultado[0][1]

			iteracoes = [ int(m[0]) for m in resultado ]
			MAEs = [ float(m[3]) for m in resultado ]
			RMSEs = [ float(m[2]) for m in resultado ]

			k = sum(iteracoes) / len(iteracoes)
			MAE = sum(MAEs) / len(MAEs)
			RMSE = sum(RMSEs) / len(RMSEs)


			arquivo_consolidado.write("%d, %.4f, %.4f, %s\n" % (k, MAE, RMSE, str(nomes[i_similaridade][1](valor) ) ) )

		arquivo_consolidado.close()


	pass

if __name__ == '__main__':
	main()