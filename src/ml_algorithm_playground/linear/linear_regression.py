import numpy as np
class LinearRegression:
    def __init__(self,lr=0.01,max_iter=1000,tol=1e-6):
        self.lr = lr
        self.max_iter = max_iter
        self.W = None
        self.b = None
    
    # utility functions
    def identity(self,X):
        return X
    
    # initialization
    def __initialization(self,X,y):
        n, d = X.shape
        _, q = y.shape
        self.W = np.zeros((d,q))
        self.b = np.zeros(q)

    def fit(self,X,y):
        n, d = X.shape
        _, q = y.shape
        self.__initialization(X,y)

        for i in range(self.max_iter):
            z= X@self.W + self.b
            y_pred = self.identity(z)
            error = y_pred - y#n,q
            dw = (1/n)* X.T@error#d,q
            db = (1/n)* np.sum(error,axis=0)#q
            self.W = self.W - self.lr * dw
            self.b = self.b - self.lr * db
        return self
    
    def predict(self,X):
        z= X@self.W + self.b
        y_pred = self.identity(z)
        return y_pred