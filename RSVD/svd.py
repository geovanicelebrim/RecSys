#-*-coding:utf-8-*-
import numpy as np
import os.path
import time
from cross_validation import split_dataset


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
	lines_train, lines_test = split_dataset(path_data)

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
def rsvd(I, rating_train, rating_test, filename='./tests/svd_erros_teste.csv', lamb=0.09, k=10, max_iteration=100, lrate=0.005, delta=0.0001):
	n_users, n_items = rating_train.shape

	P = np.random.rand(k,n_users)
	Q = np.random.rand(k,n_items)
	users,items = rating_train.nonzero()

	best_test_rmse = float('inf')
	best_test_mae = float('inf')

	old_rmse = float('inf')
	test_rmse = 999
	test_mae = 999
	iteration = 0
	start = time.time()

	while (abs(old_rmse - test_rmse) >= delta) and (iteration < max_iteration):
		print("\rIteração: {}, Delta: {:.5f}, RMSE: {:.4f}, MAE: {:.4f}     ".format(iteration, abs(old_rmse - test_rmse), test_rmse, test_mae), end="")
		old_rmse = test_rmse
		iteration += 1
		
		for u, i in zip(users,items):
			e = rating_train[u, i] - prediction(P[:,u],Q[:,i])
			P[:,u] += lrate * ( e * Q[:,i] - lamb * P[:,u])
			Q[:,i] += lrate * ( e * P[:,u] - lamb * Q[:,i])
		test_rmse, test_mae = calc_errors(I,rating_test,P,Q)

		if test_rmse < best_test_rmse:
			best_test_rmse = test_rmse

		if test_mae < best_test_mae:
			best_test_mae = test_mae

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
	I, rating_train, rating_test = crete_dataset()
	rsvd(I, rating_train, rating_test)
	exit()