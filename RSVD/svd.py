#-*-coding:utf-8-*-
import numpy as np
import os.path
import time
from cross_validation import dividir_base
from random import randint

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
	# P,Q = load_model()
	users,items = rating_train.nonzero()

	best_test_rmse = float('inf')
	best_test_mae = float('inf')

	# if delta == 0.0:
	# I2 = rating_train.copy()
	# I2[I2 > 0] = 1
	# output = open(filename, "w")
	# output.write("iteration,rmse_train,rmse_test,mae_train,mae_test\n")

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
		# train_rmse, train_mae = calc_errors(I2,rating_train,P,Q)


	# write_model(P, Q)
	print("")
	# print(train_mae, train_rmse)
	
	elapsed = time.time() - start
	return(iteration, best_test_rmse, best_test_mae, elapsed)

def write_model(P, Q):
	file = open("model_Q", "w")

	for line in range(Q.shape[0]):
		for collum in range(Q.shape[1]):
			if collum + 1 == Q.shape[1]:
				file.write(str(Q[line,collum]))
			else:
				file.write(str(Q[line,collum]) + "\t")
		file.write("\n")
	
	file.close()

	file = open("model_P", "w")

	for line in range(P.shape[0]):
		for collum in range(P.shape[1]):
			if collum + 1 == P.shape[1]:
				file.write(str(P[line,collum]))
			else:
				file.write(str(P[line,collum]) + "\t")
		file.write("\n")
	
	file.close()
	pass

def load_model():
	file = open("model_Q", "r")
	lines = file.readlines()
	Q = np.zeros((len(lines), len(lines[0].split("\t"))))
	file.close()

	for l in range(Q.shape[0]):
		for c in range(Q.shape[1]):
			Q[l,c] = float(lines[l].split("\t")[c])

	file = open("model_P", "r")
	lines = file.readlines()
	P = np.zeros((len(lines), len(lines[0].split("\t"))))
	file.close()

	for l in range(P.shape[0]):
		for c in range(P.shape[1]):
			P[l,c] = float(lines[l].split("\t")[c])

	return (P,Q)

def noise_rsvd(rating_train):

	P,Q = load_model()

	###### CALCULA OS ERROS #######
	I2 = rating_train.copy()
	I2[I2 > 0] = 1

	rating = rating_train
	new_rating = np.zeros(rating.shape)
	pred = prediction(P,Q)

	noise = 0
	valid = 0

	for l in range(rating.shape[0]):
		for c in range(rating.shape[1]):
			if rating[l,c] != 0.0:
				valid += 1
				new_rating[l,c] = rating[l,c]
				if abs(rating[l,c] - pred[l,c]) > 2:
					noise += 1
					new_rating[l,c] = pred[l,c]

	print("Valid: ", valid)
	print("Noise: ", noise)
	n_valid = len((rating[rating>0]))
	
	matrix = (I2 * (rating - prediction(P,Q)))
	#Calcula RMSE
	rmse = np.sqrt(np.sum(matrix**2)/n_valid)
	#Calcula MAE
	mae = np.sum(abs(matrix))/n_valid

	print(mae, rmse)
	return new_rating

def noise_detection(rating_train):
	
	index_original_noise, train = generate_noise(rating_train)

	index_predicted_noise = np.full((rating_train.shape), -1, dtype=int)

	P,Q = load_model()

	###### CALCULA OS ERROS #######
	rating = train
	pred = prediction(P,Q)

	for l in range(rating.shape[0]):
		for c in range(rating.shape[1]):
			if rating[l,c] != 0.0:
				if abs(rating[l,c] - pred[l,c]) > 1:
					index_predicted_noise[l,c] = 1
				else:
					index_predicted_noise[l,c] = 0

	confusion(index_original_noise, index_predicted_noise)
	

def confusion(original, predicted):
	true_noise = 0
	false_noise = 0
	true_non_noise = 0
	false_non_noise = 0

	for l in range(original.shape[0]):
		for c in range(original.shape[1]):
			if   (original[l,c] == 0) and (predicted[l,c] == 0):
				true_non_noise += 1
			elif (original[l,c] == 0) and (predicted[l,c] == 1):
				false_noise += 1
			elif (original[l,c] == 1) and (predicted[l,c] == 0):
				false_non_noise += 1
			elif (original[l,c] == 1) and (predicted[l,c] == 1):
				true_noise += 1

	print("Confusion Matrix:")
	print("\t\tTrue\tFalse")
	print("Noise:\t\t", true_noise, "\t", false_noise)
	print("Non Noise:\t", true_non_noise, "\t", false_non_noise)

	pass

# if value = -1, this prediction is not valide
# if value = 0, this prediction not contains noise
# if value = 1, this prediction contains noise
def generate_noise(rating_train, nNoise=1000):

	index_noise = np.full((rating_train.shape), -1, dtype=int)
	index_noise[rating_train > 0.0] = 0

	valid_index = {}
	count = 0;
	for l in range(rating_train.shape[0]):
		for c in range(rating_train.shape[1]):
			if rating_train[l,c] > 0.0:
				valid_index[count] = l,c
				count += 1

	if nNoise > len(valid_index):
		print("Impossivel gerar essa quantidade de ruido.")
		exit(1)

	new_train = rating_train.copy()
	for x in range(nNoise):
		i = randint(0,len(valid_index) - 1)
		while index_noise[valid_index[i][0], valid_index[i][1]] == 1:
			i = randint(0,len(valid_index) - 1)
		
		index_noise[valid_index[i][0], valid_index[i][1]] = 1

		if new_train[valid_index[i][0], valid_index[i][1]] < 3:
			new_train[valid_index[i][0], valid_index[i][1]] = float(randint(3,5))
		elif new_train[valid_index[i][0], valid_index[i][1]] > 3:
			new_train[valid_index[i][0], valid_index[i][1]] = float(randint(1,3))
		else:
			gNoise = float(randint(1,5))
			while gNoise == 3:
				gNoise = float(randint(1,5))
			new_train[valid_index[i][0], valid_index[i][1]] = gNoise

	return (index_noise, new_train)


"""
Melhores parametros:
lamb = 0.11
k = 10
lrate = 0.0066
"""
if __name__ == '__main__':

	I, rating_train, rating_test = crete_dataset(0, 5)

	noise_detection(rating_train)
	# index, train = generate_noise(rating_train, 10)

	# for l in range(index.shape[0]):
	# 	for c in range(index.shape[1]):
	# 		if index[l,c] == 1:
	# 			print("Before: ", rating_train[l,c], "\tAfter: ", train[l,c])

	# print("Resultado SVD sem tirar ruido:")
	# rsvd(I, rating_train, rating_test)
	# print("------------------------------")
	# print("Operações para remover ruido:")
	# new_rating = noise_rsvd(rating_train)

	# noise_rsvd(new_rating)
	# print("------------------------------")
	# print("Resultado SVD após tirar ruido:")
	# rsvd(I, new_rating, rating_test)