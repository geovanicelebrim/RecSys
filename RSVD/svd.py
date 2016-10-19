#-*-coding:utf-8-*-
import numpy as np
from sklearn import cross_validation as cv
import matplotlib.pyplot as plt

n_users = len(open('./ml-100k/u.user', 'r').readlines())
n_items = len(open('./ml-100k/u.item', 'r').readlines())

lines_train = open('./ml-100k/treino.txt').readlines()
lines_test = open('./ml-100k/teste.txt').readlines()

train_data = [ list(map(int, l.split())) for l in lines_train ]

test_data = [ list(map(int, l.split())) for l in lines_test ]

# Cria as matrizes de treino e teste
R = np.zeros((n_users, n_items))

for line in train_data:
	R[line[0]-1, line[1]-1] = line[2]  

T = np.zeros((n_users, n_items))
for line in test_data:
	T[line[0]-1, line[1]-1] = line[2]
	
# Indices validos da matriz de treino
I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

# Indices validos da matriz de teste
I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

# Preve o item baseado no produto escalar
def prediction(P,Q):
	return np.dot(P.T,Q)

lmbda = 0.1 # Taxa de regularização
k = 20  # Dimensão de características
m, n = R.shape  # Número de usuários e itens
n_epochs = 100  # Número de épocas
gamma = 0.01  # Taxa de aprendizado (max 0.5)

P = np.random.rand(k,m) # Matriz de variáveis latentes do usuário
Q = np.random.rand(k,n) # Matriz de variáveis latentes do item

# Calcula o RMSE
def rmse(I,R,Q,P):
	return np.sqrt(np.sum((I * (R - prediction(P,Q)))**2)/len(R[R > 0]))

# Calcula o MAE
def mae(I,R,Q,P):
	return np.sum(abs(I * (R - prediction(P,Q))))/len((R[R>0]))

train_errors_rmse = []
train_errors_mae = []
test_errors_rmse = []
test_errors_mae = []

users,items = R.nonzero()
output = open("./svd_erros_teste.csv", 'w')
output.write("epoch,rmse,mae\n")
for epoch in range(n_epochs):
	print("\rProgresso: ", ((epoch+1)*100)//n_epochs, "%", end="")
	for u, i in zip(users,items):
		e = R[u, i] - prediction(P[:,u],Q[:,i])  # Calcula o erro do gradiente
		P[:,u] += gamma * ( e * Q[:,i] - lmbda * P[:,u]) # Atualiza a matriz de características do usuário
		Q[:,i] += gamma * ( e * P[:,u] - lmbda * Q[:,i])  # Atualiza a matriz de características do item
	# train_rmse = rmse(I,R,Q,P)
	# train_mae = mae(I,R,Q,P)
	test_rmse = rmse(I2,T,Q,P)
	test_mae = mae(I2,T,Q,P)
	output.write("{},{},{}\n".format(epoch,test_rmse,test_mae))
	# train_errors_rmse.append(train_rmse)
	# train_errors_mae.append(train_mae)
	test_errors_rmse.append(test_rmse)
	test_errors_mae.append(test_mae)
output.close()
print("")

# plt.plot(range(n_epochs), train_errors_rmse, marker='o', label='Training Data - RMSE');
# plt.plot(range(n_epochs), train_errors_mae, marker='o', label='Training Data - MAE');
# plt.plot(range(n_epochs), test_errors_rmse, marker='v', label='Test Data - RMSE');
# plt.plot(range(n_epochs), test_errors_mae, marker='v', label='Test Data - MAE');
# plt.title('RSVD Learning Curve')
# plt.xlabel('Number of Epochs');
# # plt.ylabel('RMSE');
# plt.ylabel('Error');
# plt.legend()
# plt.grid()
# plt.show()