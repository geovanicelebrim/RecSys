import os.path
from numpy.random import shuffle

path = "ml-100k/"

"""
Divide o conjunto de dados em dois subconjuntos: treino e teste, 
a uma porcentagem dada como parâmetro.

@param file Caminho do arquivo de entrada
@param percent Porcentagem do conjunto de treino em relação ao conjunto de dados
@return train Dados do conjunto de treino
@return test Dados do conjunto de teste

"""
def split_dataset(file='./ml-100k/u.data', percent=.9):

	data = open(file, 'r', encoding="utf-8").readlines()

	train_file = path + "treino.txt"
	test_file = path + "teste.txt"

	if not os.path.exists(test_file) or not os.path.exists(test_file):
		print("Construindo conjunto de treino e teste.")
		size = len(data)
		position = int(size*percent)
		shuffle(data)
		train, test = data[:position], data[position:]
		
		with open(train_file, 'w') as f:
			for linha in train:
				f.write(linha)

		with open(test_file, 'w') as f:
			for linha in test:
				f.write(linha)
	else:
		print("Carregando conjunto de treino e teste.")
		train = open(train_file)
		test = open(test_file)

	return train, test


if __name__ == '__main__':
	split_dataset("./ml-100k/u.data")