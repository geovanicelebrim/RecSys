#-*-coding:utf-8-*-
import numpy as np
import os.path
import time
from cross_validation import dividir_base

dados_divididos = None

def treino_teste_split(arquivo, parte, divisoes):

	global dados_divididos

	if dados_divididos == None:
		dados_divididos = dividir_base(arquivo, divisoes)

	treino = [ i for j in dados_divididos[ : parte ] for i in j]
	treino += [ i for j in dados_divididos[ parte+1: ] for i in j]

	teste = dados_divididos[parte]

	return treino, teste

"""
@brief Cria os dados que alimentam o knn

Dados os conjuntos de dados, gera os dados de treino e teste necessários para o knn.

@return matriz_treino Matriz esparsa (usuário-item) de treino
@return dados_treino Dados do conjunto de treino
@return matriz_teste Matriz esparsa (usuário-item) de teste
@return dados_teste Dados do conjunto de teste
"""

def crete_dataset(parte, divisoes):
	path_users = './ml-100k/u.user'
	path_items = './ml-100k/u.item'
	path_data = './ml-100k/u.data'

	if not os.path.exists(path_users) or not os.path.exists(path_items):
		print("O dataset não foi encontrado.")
		exit()

	n_users = len(open(path_users, 'r').readlines())
	n_items = len(open(path_items, 'r').readlines())
	lines_train, lines_test = treino_teste_split(path_data, parte, divisoes)

	train_data = [ list(map(int, l.split())) for l in lines_train ]
	test_data = [ list(map(int, l.split())) for l in lines_test ]

	rating_train = np.zeros((n_users, n_items))
	for line in train_data:
		rating_train[line[0]-1, line[1]-1] = line[2]  

	rating_test = np.zeros((n_users, n_items))
	for line in test_data:
		rating_test[line[0]-1, line[1]-1] = line[2]

	I = rating_test.copy()
	I[I > 0] = 1

	return (I, rating_train, rating_test)

# def criar_dataset(parte, divisoes):

# 	# treino, teste = treino_teste_split(u_data)
# 	treino, teste = treino_teste_split(u_data, parte, divisoes)

# 	with open(u_user, encoding="utf-8") as a:
# 		usuarios = a.readlines()	

# 	with open(u_item, encoding="utf-8") as a:
# 		itens = a.readlines()

# 	with open(u_data, encoding="utf-8") as a:
# 		dados = a.readlines()

# 	matriz_treino = esparsa((len(usuarios), len(itens)))
# 	matriz_teste = esparsa((len(usuarios), len(itens)))

# 	dados_treino = []	
# 	for linha in treino:
# 		tokens = linha.split()

# 		usuario = int(tokens[0]) - 1
# 		item 	= int(tokens[1]) - 1
# 		nota 	= int(tokens[2])

# 		matriz_treino[usuario, item] = nota

# 		dados_treino.append([usuario, item, nota])

# 	dados_teste = []
# 	for linha in teste:
# 		tokens = linha.split()

# 		usuario = int(tokens[0]) - 1
# 		item 	= int(tokens[1]) - 1
# 		nota 	= int(tokens[2])

# 		matriz_teste[usuario, item] = nota

# 		dados_teste.append([usuario, item, nota])

# 	return (matriz_treino, dados_treino, matriz_teste, dados_teste)


"""
Realiza a predição das notas utilizando o produto escalar

@param P matriz de variáveis latentes do usuário
@param Q matriz de variáveis latentes do item

@return dot(P.T, Q) produto escalar entre P e Q
"""
def prediction(P, Q):
	return np.dot(P.T, Q)


"""
Realiza o cálculo do erro RMSE e o MAE

@param I Matriz de valores que é possível prever
@param rating Matriz com o gabarito
@param P matriz de variáveis latentes do usuário
@param Q matriz de variáveis latentes do item

@return rmse Cálculo do RMSE para o conjunto de dados
@return mae Cálculo do MAE para o conjunto de dados
"""
def calc_errors(I,rating,P,Q):
 	matrix = (I * (rating - prediction(P,Q)))
 	n_valid = len((rating[rating>0]))
 	
 	#Calcula RMSE
 	rmse = np.sqrt(np.sum(matrix**2)/n_valid)
 	#Calcula MAE
 	mae = np.sum(abs(matrix))/n_valid
 	return (rmse,mae)


"""
Ajusta o modelo e mostra, para cada época, os erros obtidos 
com os dados de teste

@param I Matriz de valores que é possível prever
@param rating_train Matriz com o gabarito do treino, para ajustar o modelo
@param rating_test Matriz com o gabarito do teste, para testar o modelo
@param lamb Taxa de regularização
@param k Dimensão de características
@param n_epochs Número de épocas
@param lrate Taxa de aprendizado

@return test_rmse Resultado do RMSE da última época
@return test_mae Resultado do MAE da última época
@return elapsed Tempo gasto para executar o modelo
"""
def rsvd(I, rating_train, rating_test, filename='./tests/test_progress.csv', lamb=0.11, k=10, max_iteration=250, lrate=0.005, delta=0.0001):
	n_users, n_items = rating_train.shape

	P = np.random.rand(k,n_users)
	Q = np.random.rand(k,n_items)
	users,items = rating_train.nonzero()

	best_test_rmse = float('inf')
	best_test_mae = float('inf')

	if delta == 0.0:
		I2 = rating_train.copy()
		I2[I2 > 0] = 1
		output = open(filename, "w")
		output.write("iteration,rmse_train,rmse_test,mae_train,mae_test\n")

	old_rmse = float('inf')
	test_rmse = 999
	test_mae = 999
	iteration = 0
	start = time.time()

	while (abs(old_rmse - test_rmse) > delta) and (iteration < max_iteration):
		print("\rIteração: {}, Delta: {:.5f}, RMSE: {:.4f}, MAE: {:.4f}     ".format(iteration, abs(old_rmse - test_rmse), test_rmse, test_mae), end="")
		old_rmse = test_rmse
		iteration += 1
		
		for u, i in zip(users,items):
			e = rating_train[u, i] - prediction(P[:,u],Q[:,i])
			P[:,u] += lrate * ( e * Q[:,i] - lamb * P[:,u])
			Q[:,i] += lrate * ( e * P[:,u] - lamb * Q[:,i])
		test_rmse, test_mae = calc_errors(I,rating_test,P,Q)
		if delta == 0.0:
			train_rmse, train_mae = calc_errors(I2,rating_train,P,Q)

		if test_rmse < best_test_rmse:
			best_test_rmse = test_rmse

		if test_mae < best_test_mae:
			best_test_mae = test_mae

		if delta == 0.0:
			output.write("{},{},{},{},{}\n".format(iteration, train_rmse, test_rmse, train_mae, test_mae))
			output.flush()
	if delta == 0.0:
		output.close()

	print("")
	elapsed = time.time() - start
	return(iteration, best_test_rmse, best_test_mae, elapsed)

"""
Melhores parametros:
best_lamb = 0.11
best_k = 10
best_lrate = 0.0066
"""
if __name__ == '__main__':

	divisoes = 5

	for i in range(divisoes):

		print("Calculando para a parte", i)

		I, rating_train, rating_test = crete_dataset(i, divisoes)

		#######################    TESTE DE PROGRESSO   ###########################
		
		print("Teste do progresso")
		rsvd(I, rating_train, rating_test, filename='./tests/test_progress_%d.csv' % (i,), delta=0.0)

		###########################################################################


		###########################################################################
		# Variando o lambda
		print("Teste do lambda")

		output = open("./tests/test_lambda_%d.csv" % (i,), "w")
		output.write("iteration,lambda,rmse,mae\n")

		# for l in np.arange(0.05, 0.15, 0.01):
		for l in np.arange(0.05, 0.16, 0.01):
			print("Testando para lambda = ", l)
			iteration, rmse, mae, elapsed = rsvd(I, rating_train, rating_test, lamb=l)
			output.write("%d, %f, %.4f, %.4f\n" % (iteration, l, rmse, mae))
			output.flush()

		output.close()

		###########################################################################
		# Variando o lrate
		print("Teste do lrate")

		output = open("./tests/test_lrate_%d.csv" % (i,), "w")
		output.write("iteration,lrate,rmse,mae\n")

		# for l in np.arange(0.005, 0.07, 0.0065):
		for l in np.arange(0.005, 0.0764, 0.0065):
			print("Testando para lrate = ", l)
			iteration, rmse, mae, elapsed = rsvd(I, rating_train, rating_test, lrate=l)
			output.write("%d, %f, %.4f, %.4f\n" % (iteration, l, rmse, mae))
			output.flush()

		output.close()

		###########################################################################
		# Variando o k
		print("Teste do k")

		output = open("./tests/test_k_%d.csv" % (i,), "w")
		output.write("iteration,k,rmse,mae\n")

		# for k in range(10, 40, 5):
		for k in range(10, 41, 5):
			print("Testando para k = ", k)
			iteration, rmse, mae, elapsed = rsvd(I, rating_train, rating_test, k=k)
			output.write("%d, %d, %.4f, %.4f\n" % (iteration, k, rmse, mae))
			output.flush()

		output.close()
