#-*-coding:utf-8-*-
import numpy as np
import os.path
import time
from cross_validation import treino_teste_split


"""
Obtém o dataset e, caso não exista um arquivo de treino e outro de
teste, esses arquivos são criados.

@return I Matriz de valores que é possível prever
@return rating_train Matriz com o gabarito do treino
@return rating_test Matriz com o gabarito do teste
"""
def crete_dataset():
	path_users = './ml-100k/u.user'
	path_items = './ml-100k/u.item'
	path_data = './ml-100k/u.data'

	if not os.path.exists(path_users) or not os.path.exists(path_items):
		print("O dataset não foi encontrado.")
		exit()

	n_users = len(open(path_users, 'r').readlines())
	n_items = len(open(path_items, 'r').readlines())
	lines_train, lines_test = treino_teste_split(path_data)

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
def rsvd(I, rating_train, rating_test, filename='./tests/svd_erros_teste.csv', lamb=0.1, k=15, n_epochs=30, lrate=0.01):
	n_users, n_items = rating_train.shape  # Número de usuários e itens
	P = np.random.rand(k,n_users)
	Q = np.random.rand(k,n_items)
	users,items = rating_train.nonzero()

	best_test_rmse = float('inf')
	best_test_mae = float('inf')

	# output = open(filename, 'w')
	# output.write("epoch,rmse,mae\n")
	start = time.time()
	for epoch in range(n_epochs):
		print("\rProgresso: ", ((epoch+1)*100)//n_epochs, "%", end="")
		for u, i in zip(users,items):
			e = rating_train[u, i] - prediction(P[:,u],Q[:,i])
			P[:,u] += lrate * ( e * Q[:,i] - lamb * P[:,u])
			Q[:,i] += lrate * ( e * P[:,u] - lamb * Q[:,i])
		test_rmse, test_mae = calc_errors(I,rating_test,P,Q)

		if test_rmse < best_test_rmse:
			best_test_rmse = test_rmse

		if test_mae < best_test_mae:
			best_test_mae = test_mae
		# output.write("{},{},{}\n".format(epoch,test_rmse,test_mae))
		# output.flush()
	# output.close()
	print("")
	elapsed = time.time() - start
	# return(test_rmse, test_mae, elapsed)
	return(best_test_rmse, best_test_mae, elapsed)


best_lamb = 0.1
best_k = 30
best_n_epochs = 100
best_lrate = 0.01

"""
Sugestão de parâmetros do paper "Improving regularized singular value decomposition for
collaborative filtering":

lamb = Ideal(0.1) Max(0.15) Min(0.05) Passo(0.001)
k = Ideal(30) Max(40) Min(10) Passo(5)
n_epochs = 40-120 (10)
lrate = (ideal: 0.01) Max(0.07) Min(0.005) Passo(0.0065)
"./svd_erros_teste.csv"
"""
if __name__ == '__main__':
	I, rating_train, rating_test = crete_dataset()

	###########################################################################
	# Variando o lambda
	# output = open("./tests/test_lambda.csv", "w")
	# output.write("lambda,rmse,mae\n")

	# for l in np.arange(0.05, 0.15, 0.01):
	# 	print("Testando para lambda = ", l)
	# 	rmse, mae, elapsed = rsvd(I, rating_train, rating_test, lamb=l)
	# 	output.write("{},{},{}\n".format(l, rmse, mae))
	# 	output.flush()

	# output.close()

	# ###########################################################################
	# # Variando o lrate
	# output = open("./tests/test_lrate.csv", "w")
	# output.write("lrate,rmse,mae\n")

	# for l in np.arange(0.005, 0.07, 0.0065):
	# 	print("Testando para lrate = ", l)
	# 	rmse, mae, elapsed = rsvd(I, rating_train, rating_test, lrate=l)
	# 	output.write("{},{},{}\n".format(l, rmse, mae))
	# 	output.flush()

	# output.close()

	# ###########################################################################
	# # Variando o k
	# output = open("./tests/test_k.csv", "w")
	# output.write("k,rmse,mae\n")

	# for k in range(10, 40, 5):
	# 	print("Testando para k = ", k)
	# 	rmse, mae, elapsed = rsvd(I, rating_train, rating_test, k=k)
	# 	output.write("{},{},{}\n".format(k, rmse, mae))
	# 	output.flush()

	# output.close()

	###########################################################################
	# Variando o epoch
	output = open("./tests/test_epochs.csv", "w")
	output.write("epoch,rmse,mae\n")

	for epoch in range(40, 121, 10):
		print("Testando para epoch = ", epoch)
		rmse, mae, elapsed = rsvd(I, rating_train, rating_test, n_epochs=epoch)
		output.write("{},{},{}\n".format(epoch, rmse, mae))
		output.flush()

	output.close()

	###########################################################################

	# rmse, mae, elapsed = rsvd(I, rating_train, rating_test, './svd_erros_teste.csv')
	# print("Último RMSE: {:.4f}, Último MAE: {:.4f}, Tempo gasto: {:.4f} sec.".format(rmse, mae, elapsed))
