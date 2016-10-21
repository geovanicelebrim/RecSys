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
	treino_teste_split(path_data)
	lines_train = open('./ml-100k/treino.txt').readlines()
	lines_test = open('./ml-100k/teste.txt').readlines()

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
def rsvd(I, rating_train, rating_test, lamb=0.1, k=15, n_epochs=30, lrate=0.01):
	n_users, n_items = rating_train.shape  # Número de usuários e itens
	P = np.random.rand(k,n_users)
	Q = np.random.rand(k,n_items)
	users,items = rating_train.nonzero()

	output = open("./svd_erros_teste.csv", 'w')
	output.write("epoch,rmse,mae\n")
	start = time.time()
	for epoch in range(n_epochs):
		print("\rProgresso: ", ((epoch+1)*100)//n_epochs, "%", end="")
		for u, i in zip(users,items):
			e = rating_train[u, i] - prediction(P[:,u],Q[:,i])
			P[:,u] += lrate * ( e * Q[:,i] - lamb * P[:,u])
			Q[:,i] += lrate * ( e * P[:,u] - lamb * Q[:,i])
		test_rmse, test_mae = calc_errors(I,rating_test,P,Q)
		output.write("{},{},{}\n".format(epoch,test_rmse,test_mae))
	output.close()
	print("")
	elapsed = time.time() - start
	return(test_rmse, test_mae, elapsed)


"""
Sugestão de parâmetros do paper "Improving regularized singular value decomposition for
collaborative filtering":

lrate = .001
lamb = .02
k = 96
"""
if __name__ == '__main__':
	I, rating_train, rating_test = crete_dataset()
	rmse, mae, elapsed = rsvd(I, rating_train, rating_test)
	print("Último RMSE: {:.4f}, Último MAE: {:.4f}, Tempo gasto: {:.4f} sec.".format(rmse, mae, elapsed))
