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

def create_dataset(parte, divisoes):
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
Realiza o cálculo do erro RMSE e o MAE

@param I Matriz de valores que é possível prever
@param rating Matriz com o gabarito
@param prediction matriz de predição

@return rmse Cálculo do RMSE para o conjunto de dados
@return mae Cálculo do MAE para o conjunto de dados
"""
def errors(I,rating,prediction):
 	matrix = (I * (rating - prediction))
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
def rsvd(I, rating_train, rating_test, filename='./tests/test_progress.csv', lamb=0.11, k=10, max_iteration=250, lrate=0.005, delta=0.0001, persist=None):
	n_users, n_items = rating_train.shape

	################## PARA AGILIZAR O PROCESSO DO ALGORITMO O MODELO PODE SER CARREGADO DO ARQUIVO USANDO 'load_model() ###################
	P = np.random.rand(k,n_users)
	Q = np.random.rand(k,n_items)
	# P,Q = load_model()
	########################################################################################################################################
	
	users,items = rating_train.nonzero()

	best_test_rmse = float('inf')
	best_test_mae = float('inf')

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

	if test_rmse < best_test_rmse:
		best_test_rmse = test_rmse

	if test_mae < best_test_mae:
		best_test_mae = test_mae

	############ PARA ESCREVER O MODELO NO ARQUIVO, DESCOMENTE A LINHA ABAIXO #############
	if persist is not None:
		write_model(P, Q)
	#######################################################################################
	print("")
	
	elapsed = time.time() - start
	return(iteration, best_test_rmse, best_test_mae, elapsed)


"""
Escreve um modelo de predição obtido com o RSVD em um arquivo, para que 
este possa ser rápidamente reutilizado.

@param P matriz de variáveis latentes do usuário
@param Q matriz de variáveis latentes do item
"""
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


"""
Carrega um modelo de predição obtido com o RSVD de um arquivo

@return P, Matriz de variáveis latentes do usuário
@return Q, Matriz de variáveis latentes do item
"""
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


def classify_possible_noise(rating_train, user_based_threshold=True):
	user_classes = [ -1 for i in range(rating_train.shape[0]) ]
	item_classes = [ -1 for i in range(rating_train.shape[1]) ]

	kv = None
	threshold = None

	if user_based_threshold:
		kv = [ [0, 0] for i in range(rating_train.shape[0]) ]
		threshold = [ 0 for i in range(rating_train.shape[0]) ]
	else:
		kv = [ [0, 0] for i in range(rating_train.shape[1]) ]
		threshold = [ 0 for i in range(rating_train.shape[1]) ]

	possible_noise = np.ones((rating_train.shape[0], rating_train.shape[1]), dtype=bool)

	# count_c = 0
	# count_a = 0
	# count_b = 0
	# count_v = len(user_classes)

	for u in range(rating_train.shape[0]):

		nz = rating_train[u].nonzero()[0]

		# ratings do usuário
		ratings = [ rating_train[u, i] for i in nz]

		# mean = sum(ratings)/len(ratings)
		mean = np.mean(ratings)
		# mean = np.average(ratings)
		std = np.std(ratings)

		k = mean - std
		v = mean + std

		w = len([ rating_train[u, i] for i in nz if rating_train[u, i] < k ])
		a = len([ rating_train[u, i] for i in nz if k <= rating_train[u, i] and rating_train[u, i] < v ])
		s = len([ rating_train[u, i] for i in nz if rating_train[u, i] >= v ])

		if user_based_threshold:
			kv[u] = [k, v]
			threshold[u] = std

		if w >= (a + s):
			user_classes[u] = 0
			# count_c += 1
			# count_v -= 1

			# print(u, user_classes[u])
		elif a >= (w + s):
			user_classes[u] = 1

			# count_a += 1
			# count_v -= 1

		elif s >= (w + a):
			user_classes[u] = 2

			# count_b += 1
			# count_v -= 1
		pass

	# print(count_c, count_a, count_b, count_v)

	# count_c = 0
	# count_a = 0
	# count_b = 0
	# count_v = len(item_classes)

	for i in range(rating_train.shape[1]):

		nz = rating_train[:,i].nonzero()[0]

		# ratings do usuário
		ratings = [ rating_train[u, i] for u in nz]

		mean = np.mean(ratings) if (len(ratings) > 0) else 0
		# mean = np.average(ratings) if (len(ratings) > 0) else 0
		std = np.std(ratings) if (len(ratings) > 0) else 0

		k = mean - std
		v = mean + std

		w = len([ rating_train[u, i] for u in nz if rating_train[u, i] < k ])
		a = len([ rating_train[u, i] for u in nz if k <= rating_train[u, i] and rating_train[u, i] < v ])
		s = len([ rating_train[u, i] for u in nz if rating_train[u, i] >= v ])

		if not user_based_threshold:
			kv[i] = [k, v]
			threshold[i] = std

		if w >= (a + s) and mean > 0:
			item_classes[i] = 0
			# count_c += 1
			# count_v -= 1

		elif a >= (w + s) and mean > 0:
			item_classes[i] = 1

			# count_a += 1
			# count_v -= 1

		elif s >= (w + a) and mean > 0:
			item_classes[i] = 2

			# count_b += 1
			# count_v -= 1
		pass

	count_pn = 0
	
	for u in range(rating_train.shape[0]):
		for i in range(rating_train.shape[1]):

			if rating_train[u, i] > 0:

				k, v = 0, 0

				if user_based_threshold:
					k, v = kv[u]
				else:
					k, v = kv[i]

				if user_classes[u] == 0 and item_classes[i] == 0 and rating_train[u, i] >= k:
					possible_noise[u, i] = True

					count_pn += 1
				elif user_classes[u] == 1 and item_classes[i] == 1 and (rating_train[u, i] < k or rating_train[u, i] >= v):
					possible_noise[u, i] = True

					count_pn += 1
				elif user_classes[u] == 2 and item_classes[i] == 2 and rating_train[u, i] < v:
					possible_noise[u, i] = True

					count_pn += 1
				pass

			pass

		pass
	# print(count_c, count_a, count_b, count_v)
	# print(count_pn)

	return (possible_noise, threshold, count_pn)

"""
Uma vez realizada a predição normalmente usando apenas o RSVD e escrevendo
no arquivo o modelo de predição obtido, esta função utiliza o modelo
construído para detectar ruídos presentes na base de TREINO, considerando
um threshold. Além de detectar, esses valores são corrigidos pelo valor 
da predição.

@param rating_train, Matriz de treino que será corrigida
@param threshold, Limiar de diferença não normalizada entre as notas
[1 <= threshold <= 5]

@return new_rating, Matriz de treino corrigida, sem ruído.
"""
def noise_rsvd(rating_train, threshold=1, possible_noise=None, threshold_vec=None, user_based_pv=True):

	P,Q = load_model()

	###### OBTEM OS VALORES VÁLIDOS DA BASE DE DADOS #######
	I = rating_train.copy()
	I[I > 0] = 1
	########################################################
	
	new_rating = rating_train.copy()
	pred = prediction(P,Q)

	noise = 0
	valid = 0

	####### COMPARA A MATRIZ ORIGINAL DE TREINO COM A SUA PREDIÇÃO E IDENTIFICA O RUÍDO #######
	for l in range(rating_train.shape[0]):
		for c in range(rating_train.shape[1]):
			if rating_train[l,c] != 0.0:
				valid += 1
				# new_rating[l,c] = rating_train[l,c]

				if possible_noise is None:
					if abs(rating_train[l,c] - pred[l,c]) > threshold:
						noise += 1
						new_rating[l,c] = pred[l,c]

						pass
					pass

				elif possible_noise[l, c] == True:

					if user_based_pv:
						if abs(rating_train[l,c] - pred[l,c]) > threshold_vec[l]:
							noise += 1
							new_rating[l,c] = pred[l,c]

					else:
						if abs(rating_train[l,c] - pred[l,c]) > threshold_vec[c]:
							noise += 1
							new_rating[l,c] = pred[l,c]

					pass
				pass
			pass
		pass
	###########################################################################################

	# print("Valid cases: ", valid)
	# print("Noise decected: ", noise)
	
	# old_mae, old_rmse = errors(I,rating_train,pred)
	# print("Old train MAE: ", old_mae, "\tOld train RMSE: ", old_rmse)

	# new_mae, new_rmse = calc_errors(I, new_rating, P, Q)
	# print("New train MAE: ", new_mae, "\tNew train RMSE: ", new_rmse)

	return new_rating


"""
Considerando que existe um modelo no arquivo (lembrando que para cada base
existe um modelo que melhor se adequa, portanto, os testes devem ser feitos
com modelos de uma mesma base.), esta função, a partir de uma base, constroi
uma nova base com um determinado valor de ruído. Após isso, seu objetivo é
submeter a base ao modelo com a finalidade de obter a acurácia da detecção 
do ruído gerado.

Ao final, é apresentado uma matriz de confusão com os resultados alcansados.

@param rating_train, base de treino original
@param nNoise, número de dados que serão afetados pelo ruído
@param threshold, Limiar de diferença não normalizada entre as notas
[1 <= threshold <= 5]
"""
def noise_detection(rating_train, nNoise=1000, threshold=3):
	#### OBTÉM UMA NOVA BASE PERTURBADA E UMA MATRIZ COM OS INDICES QUE FORAM PERTURBADOS ####
	index_original_noise, train = generate_noise(rating_train, nNoise)
	#### MATRIZ DE INDICES QUE O MODELO ACREDITA QUE FOI OU NÃO PERTURBADO ####
	index_predicted_noise = np.full((rating_train.shape), -1, dtype=int)

	P,Q = load_model()

	pred = prediction(P,Q)

	##### COMPARA O TREINO (BASE PERTURBADA) COM A PREDIÇÃO PARA DETERMINAR O QUE É RUÍDO ####
	for l in range(train.shape[0]):
		for c in range(train.shape[1]):
			if train[l,c] != 0.0:
				if abs(train[l,c] - pred[l,c]) > threshold:
					index_predicted_noise[l,c] = 1
				else:
					index_predicted_noise[l,c] = 0
	##########################################################################################

	#### COMPARA OS INDICES ONDE O MODELO ENCONTROU RUÍDO COM O GABARITO ####
	return confusion(index_original_noise, index_predicted_noise)


"""
Apresenta a matriz de confusão obtida na avaliação da detecção de ruído

@param original, idices que realmente foram perturbados
@param predicted, indices que o modelo encontrou ruido

"""
def confusion(original, predicted):
	true_positive = 0
	false_positive = 0
	true_negative = 0
	false_negative = 0

	for l in range(original.shape[0]):
		for c in range(original.shape[1]):
			if   (original[l,c] == 0) and (predicted[l,c] == 0):
				true_negative += 1
			elif (original[l,c] == 0) and (predicted[l,c] == 1):
				false_positive += 1
			elif (original[l,c] == 1) and (predicted[l,c] == 0):
				false_negative += 1
			elif (original[l,c] == 1) and (predicted[l,c] == 1):
				true_positive += 1

	print("Confusion Matrix:")
	print("\t\tTrue\tFalse")
	print("Noise:\t\t", true_positive, "\t", false_positive)
	print("Non Noise:\t", true_negative, "\t", false_negative)

	return (true_positive, false_positive, true_negative, false_negative)

"""
Gera ruído em uma base de treino
O critério para gerar ruído é:
Se rating < 3
rand(3, 5)
Se rating > 3
rand(1, 3)
Se rating = 3
rand(valor diferente de 3)

A matriz de indices perturbados contém a seguinte característica:
Se valor = -1, o indice é não válido
Se valor = 0, a predição não contém ruído
Se valor = 1, a predição contains ruído

@param rating_train, matriz de treino original
@param nNoise, número de ratings que serão perturbados

@return index_noise, matriz com os indices que foram perturbados
@return new_train, nova matriz gerada a partir da perturbação da batriz de treino
"""

def generate_noise(rating_train, nNoise):
	#### INICIALIZA A MATRIZ DE INDICES COM -1 ####
	index_noise = np.full((rating_train.shape), -1, dtype=int)
	#### INDICES QUE SÃO VÁLIDOS SÃO ALTERADOS PARA 0 ####
	index_noise[rating_train > 0.0] = 0

	#### CONSTROI DICIONÁRIO DE INDICES QUE PODEM SER PERTURBADOS ####
	valid_index = {}
	count = 0;
	
	for l in range(rating_train.shape[0]):
		for c in range(rating_train.shape[1]):
			if rating_train[l,c] > 0.0:
				valid_index[count] = l,c
				count += 1
	##################################################################

	if nNoise > len(valid_index):
		print("Impossivel gerar essa quantidade de ruido.")
		exit(1)

	#### NOVA MATRIZ DE TREINO QUE SERÁ PERTURBADA ####
	new_train = rating_train.copy()

	##### PARA CADA RUÍDO QUE PRECISA SE GERADO, É SORTEADO UM RUÍDO VÁLIDO.  #####
	##### SE ESTE INDICE AINDA NÃO POSSUIR RUÍDO, SEU VALOR É PERTURBADO E    #####
	##### A MATRIZ QUE GUARDA OS INDICES DOS VALORES PERTURBADOS, ATUALIZADA. #####
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
	###############################################################################

	return (index_noise, new_train)


if __name__ == '__main__':


	divisoes = 5

	output_1 = open("./tests/test_normal_rsvd.csv", "w")
	output_1.write("iteration,rmse,mae,elapsed\n")

	output_2 = open("./tests/test_mahony_rsvd.csv", "w")
	output_2.write("iteration,threshold,rmse,mae,elapsed\n")

	output_3 = open("./tests/test_mahony_detect_natural_noise_rsvd.csv", "w")
	output_3.write("iteration,threshold,true_positive,false_positive,true_negative,false_negative\n")

	output_4 = open("./tests/test_mahony_detect_generate_noise_rsvd.csv", "w")
	output_4.write("iteration,threshold,noise,true_positive,false_positive,true_negative,false_negative\n")


	for i in range(divisoes):
		print("Calculando para a parte", i)

		I, rating_train, rating_test = create_dataset(i, divisoes)

		
		############################# TESTANDO RSVD SEM TIRAR RUÍDO #############################
		print("Testando RSVD sem tirar ruido:")                                               #
		iteration, rmse, mae, elapsed = rsvd(I, rating_train, rating_test, persist=True)      #
		output_1.write("%d, %.4f, %.4f, %.4f\n" % (i, rmse, mae, elapsed))                    #
		output_1.flush()                                                                      #
		print("-------------------------------------------")                                  #
		#########################################################################################
		

		###### TESTANDO RSVD COM REMOÇÃO DE RUÍDO USANDO MAHONY E VARIAÇÃO DO THRASHOLD #######
		for threshold in range(1,5):                                                          #
			print("Testando a remoção de ruido com Mahony com threshold: ", threshold)        #
			new_rating = noise_rsvd(rating_train, threshold=threshold)                        #
			iteration, rmse, mae, elapsed = rsvd(I, new_rating, rating_test, persist=None)    #
			output_2.write("%d, %d, %.4f, %.4f, %.4f\n" % (i, threshold, rmse, mae, elapsed)) #
			output_2.flush()                                                                  #
			print("-------------------------------------------")                              #
		#########################################################################################


		################################ TESTANDO DETECÇÃO DE RUÍDO NATURAL COM MAHONY VARIANDO O THRASHOLD #################################
		print("Testando a detecção de ruído natural variando o threshold:")                                                               #
		for threshold in range(1,5):                                                                                                      #
			true_positive, false_positive, true_negative, false_negative = noise_detection(rating_train, nNoise=0, threshold=threshold)   #
			output_3.write("%d, %d, %d, %d, %d, %d\n" % (i, threshold, true_positive, false_positive, true_negative, false_negative))     #
			output_3.flush()                                                                                                              #
		#####################################################################################################################################


		###################################### TESTANDO DETECÇÃO DE RUÍDO GERADO COM MAHONY VARIANDO O THRASHOLD #######################################
		for threshold in range(1,5):                                                                                                                 #
			for noise in range(800,40001, 800):                                                                                                      #
				new_rating = noise_rsvd(rating_train, threshold=threshold)                                                                           #
				true_positive, false_positive, true_negative, false_negative = noise_detection(new_rating, nNoise=noise, threshold=threshold)        #
				output_4.write("%d, %d, %d, %d, %d, %d, %d\n" % (i, threshold, noise, true_positive, false_positive, true_negative, false_negative)) #
				output_4.flush()                                                                                                                     #
		################################################################################################################################################

	output_1.close()
	output_2.close()
	output_3.close()
	output_4.close()



	# I, rating_train, rating_test = create_dataset(0, 5)

	# index_noise, new_train = generate_noise(rating_train, 285)

	# # possible_noise, threshold = classify_possible_noise(rating_train)
	# possible_noise, threshold, count_pn = classify_possible_noise(new_train)
	# print("Possíveis ruídos:", count_pn)			#


	# ############# TESTANDO REMOÇÃO DE RUÍDO #############
	# print("Operações para remover ruido:")			#
	# # new_rating = noise_rsvd(rating_train, possible_noise=possible_noise, threshold_vec=threshold)				#
	# new_rating = noise_rsvd(new_train, possible_noise=possible_noise, threshold_vec=threshold)				#
	# print("------------------------------")			#

	