import numpy as np
import random
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=5):
        self.k = k
        self.Labels = []
        self.mu = []

    def fit_predict(self,X):
        m = X.shape[0]
        self.Labels = np.zeros([m,])

        # Initialization
        idSelected = random.sample(range(0,m),self.k)
        self.mu = X[idSelected,:]

        # Iteration
        while True:
            oldLabels = self.Labels

            Distances = np.zeros([m,self.k])
            for i in range(self.k):
                Dist = np.sum((X-np.tile(self.mu[i,:],(m,1)))*(X-np.tile(self.mu[i,:],(m,1))), axis=1)
                Distances[:,i] = np.reshape(Dist,(m,))

            self.Labels = np.argmin(Distances,axis=1)
            if(self.Labels == oldLabels).all():
                return(self.mu, self.Labels)
            else:
                for i in range(self.k):
                    kid = np.where(self.Labels == i)
                    self.mu[i,:] = np.sum(X[kid,:],axis=1) / np.shape(kid)[1]

if __name__ == '__main__':
    X = np.array( [[0,0],[1,0],[0,1],[1,1],[2,1],[1,2],[2,2],[3,2],[6,6],[7,6],
                   [8,6],[7,7],[8,7],[9,7],[7,8],[8,8],[9,8],[8,9],[9,9] ] )

    cluster = KMeans(k=2)
    mu,labels = cluster.fit_predict(X)

    print("\nCluster Labels:\n", labels)
    print("\nCluser Means:\n",mu)

    plt.scatter(X[:,0],X[:,1],c=labels,cmap='rainbow')
    plt.show()    
