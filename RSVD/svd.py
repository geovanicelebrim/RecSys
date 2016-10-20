#-*-coding:utf-8-*-
import numpy as np
import os.path
import time
from cross_validation import treino_teste_split

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

	R = np.zeros((n_users, n_items))
	for line in train_data:
		R[line[0]-1, line[1]-1] = line[2]  

	T = np.zeros((n_users, n_items))
	for line in test_data:
		T[line[0]-1, line[1]-1] = line[2]

	# Indices validos da matriz de teste
	I2 = T.copy()
	I2[I2 > 0] = 1
	I2[I2 == 0] = 0

	return (I2, R, T)

# Preve o item baseado no produto escalar
def prediction(P,Q):
	return np.dot(P.T,Q)

def calc_erros(I,R,Q,P):
 	matriz = (I * (R - prediction(P,Q)))
 	nR = len((R[R>0]))
 	
 	#Calcula RMSE
 	r = np.sqrt(np.sum(matriz**2)/nR)
 	#Calcula MAE
 	m = np.sum(abs(matriz))/nR
 	return (r,m)

I2, R, T = crete_dataset()

lmbda = 0.1 # Taxa de regularização
k = 15  # Dimensão de características
m, n = R.shape  # Número de usuários e itens
n_epochs = 30  # Número de épocas
gamma = 0.01  # Taxa de aprendizado (max 0.5)

P = np.random.rand(k,m) # Matriz de variáveis latentes do usuário
Q = np.random.rand(k,n) # Matriz de variáveis latentes do item


users,items = R.nonzero()
output = open("./svd_erros_teste.csv", 'w')
output.write("epoch,rmse,mae\n")
start = time.time()
for epoch in range(n_epochs):
	print("\rProgresso: ", ((epoch+1)*100)//n_epochs, "%", end="")
	for u, i in zip(users,items):
		e = R[u, i] - prediction(P[:,u],Q[:,i])  # Calcula o erro do gradiente
		P[:,u] += gamma * ( e * Q[:,i] - lmbda * P[:,u]) # Atualiza a matriz de características do usuário
		Q[:,i] += gamma * ( e * P[:,u] - lmbda * Q[:,i])  # Atualiza a matriz de características do item
	test_rmse, test_mae = calc_erros(I2,T,Q,P)
	output.write("{},{},{}\n".format(epoch,test_rmse,test_mae))
output.close()
print("")
end = time.time()
elapsed = end - start
print("Tempo: ", elapsed)