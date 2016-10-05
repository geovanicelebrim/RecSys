#-*- coding: utf-8 -*-

import math
import random
import time

"""
Rating
Classe que representa a nota de um usuário para um item.
"""
class Rating:
    def __init__(self, userid, movieid, rating):
        self.uid = userid-1 
        self.mid = movieid-1
        self.rat = rating

class SvdMatrix:
    """
    trainfile -> nome do arquivo para treino
    nusers -> número de usuários no conjunto de dados
    nmovies -> número de filmes no conjunto de dados
    r -> aproximação (para U e V)
    lrate -> taxa de aprendizagem
    regularizer -> Regularizador
    typefile -> 0 se, por menor conjunto de dados MovieLens
                1 se a médio ou MovieLens maiores conjunto de dados
    TALVEZ, TAAAAAALVEZ O r (FDP) seja o número de "vizinhos" do elemento.
        Slide: "Podemos pegar as k primeiras colunas e obter uma matriz aproximada."
    """
    def __init__(self, trainfile, nusers, nmovies, r=30, lrate=0.035, regularizer=0.01, typefile=0):
        self.trainrats = []
        self.testrats = []
        
        self.nusers = nusers
        self.nmovies = nmovies

        if typefile == 0:
            self.readtrainsmaller(trainfile)
        elif typefile == 1:
            self.readtrainlarger(trainfile)

        # get average rating
        avg = self.averagerating()
        # set initial values in U, V using square root
        # of average/rank
        initval = math.sqrt(avg/r)
        
        # U matrix
        self.U = [[initval]*r for i in range(nusers)]
        # V matrix -- easier to store and compute than V^T
        self.V = [[initval]*r for i in range(nmovies)]

        self.r = r
        self.lrate = lrate
        self.regularizer = regularizer
        self.minimprov = 0.001
        self.maxepochs = 30            

    """
    Retorna o produto escalar de v1 e v2
    """
    def dotproduct(self, v1, v2):
        return sum([v1[i]*v2[i] for i in range(len(v1))])

    """
    Retorna o rating estimado correspondente ao filme (movieid) visto por um usuário (userid)
    O valor está contido no intervalo [1,5]
    """
    def calcrating(self, uid, mid):
        p = self.dotproduct(self.U[uid], self.V[mid])
        if p > 5:
            p = 5
        elif p < 1:
            p = 1
        return p

    """
    Retorna a média dos ratings do dataset
    """
    def averagerating(self):
        avg = 0
        n = 0
        for i in range(len(self.trainrats)):
            avg += self.trainrats[i].rat
            n += 1
        return float(avg/n)

    """
    Prevê o rating estimado para o usuário com id i para o filme com id j
    """
    def predict(self, i, j):
        return self.calcrating(i, j)

    """
    Trains the kth column in U and the kth row in
    V^T
    See docs for more details.
    """
    def train(self, k):
        sse = 0.0
        n = 0
        for i in range(len(self.trainrats)):
            # get current rating
            crating = self.trainrats[i]
            err = crating.rat - self.predict(crating.uid, crating.mid)
            sse += err**2
            n += 1

            uTemp = self.U[crating.uid][k]
            vTemp = self.V[crating.mid][k]

            self.U[crating.uid][k] += self.lrate * (err*vTemp - self.regularizer*uTemp)
            self.V[crating.mid][k] += self.lrate * (err*uTemp - self.regularizer*vTemp)
        return math.sqrt(sse/n)

    """
    Trains the entire U matrix and the entire V (and V^T) matrix
    """
    def trainratings(self):        
        # stub -- initial train error
        oldtrainerr = 1000000.0
       
        for k in range(self.r):
            print("k=", k)
            for epoch in range(self.maxepochs):
                trainerr = self.train(k)
                
                # check if train error is still changing
                if abs(oldtrainerr-trainerr) < self.minimprov:
                    break
                oldtrainerr = trainerr
                print("epoch=", epoch, "; trainerr=", trainerr)
                
    """
    Calcula o erro MAE
    """
    def calcmae(self, arr):
        mae = 0.0
        
        for i in range(len(arr)):
            crating = arr[i]
            mae += abs(crating.rat - self.calcrating(crating.uid, crating.mid))
            
        return mae / len(arr)
    
    """
    Calculates the RMSE using between arr
    and the estimated values in (U * V^T)
    """
    def calcrmse(self, arr):
        nusers = self.nusers
        nmovies = self.nmovies
        sse = 0.0
        # total = 0
        for i in range(len(arr)):
            crating = arr[i]
            errorTemp = (crating.rat - self.calcrating(crating.uid, crating.mid))#**2
            if abs(errorTemp) < 1:
                print ("User ID: ", crating.uid, "Movie ID:", crating.mid, "Predictor", 
                    self.calcrating(crating.uid, crating.mid),"Rating: ", crating.rat)
                
            sse += errorTemp**2
            # sse += (crating.rat - self.calcrating(crating.uid, crating.mid))**2
            # total += 1
        # return math.sqrt(sse/total)
        return math.sqrt(sse/len(arr))

    """
    Read in the ratings from fname and put in arr
    Use splitter as delimiter in fname
    """
    def readinratings(self, fname, arr, splitter="\t"):
        f = open(fname)

        for line in f:
            newline = [int(each) for each in line.split(splitter)]
            userid, movieid, rating = newline[0], newline[1], newline[2]
            arr.append(Rating(userid, movieid, rating))

        arr = sorted(arr, key=lambda rating: (rating.uid, rating.mid))
        return len(arr)
        
    """
    Read in the smaller train dataset
    """
    def readtrainsmaller(self, fname):
        return self.readinratings(fname, self.trainrats, splitter="\t")
        
    """
    Read in the large train dataset
    """
    def readtrainlarger(self, fname):
        return self.readinratings(fname, self.trainrats, splitter="::")
        
    """
    Read in the smaller test dataset
    """
    def readtestsmaller(self, fname):
        return self.readinratings(fname, self.testrats, splitter="\t")
                
    """
    Read in the larger test dataset
    """
    def readtestlarger(self, fname):
        return self.readinratings(fname, self.testrats, splitter="::")


if __name__ == "__main__":
    #========= test SvdMatrix class on smallest MovieLENS dataset =========
    init = time.time()
    svd = SvdMatrix("ua.base", 943, 1682, r=50, lrate=0.001, regularizer=0.02)
    # svd = SvdMatrix("ua.base", 943, 1682)
    svd.trainratings()
    # print("rmsetrain: ", svd.calcrmse(svd.trainrats))
    print("rmaetrain: ", svd.calcmae(svd.trainrats))
    svd.readtestsmaller("ua.test")
    # print("rmsetest: ", svd.calcrmse(svd.testrats))
    print("rmaetest: ", svd.calcmae(svd.testrats))
    print("time: ", time.time()-init)