import numpy as np
class LinearRegression:
    def __init__(self,lr=0.01,max_iter=1000,tol=1e-6):
        self.lr = lr
        self.max_iter = max_iter
        self.w = None
        self.b = None
    
    # initialization
    def __initialization(self,X,y):
        n, d = X.shape
        _, q = y.shape
        self.w = np.zeros((d,q))
        self.b = np.zeros(q)

    def fit(self,X,y):
        assert X.ndim ==2
        assert y.ndim in [1,2]
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n, d = X.shape
        q = y.shape[1]
        self.__initialization(X,y)

        for i in range(self.max_iter):
            z= X@self.w + self.b
            y_pred = z
            error = y_pred - y#n,q
            dw = (1/n)* X.T@error#d,q
            db = (1/n)* np.sum(error,axis=0)#q
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
        return self
    
    def predict(self,X):
        z= X@self.w + self.b
        y_pred = z
        return y_pred