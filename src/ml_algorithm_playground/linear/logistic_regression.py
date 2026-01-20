import numpy as np
class LogisticRegression:
  def __init__(self, treshold=0.5,lr=0.01, max_iter=1000, tol=1e-4):
    self.treshold = treshold
    self.lr = lr
    self.max_iter = max_iter
    self.tol = tol
    self.weights = None
    self.bias = None


  def sigmoid(self, z):
    return 1/(1+np.exp(-z))
  
  def initialize_weights(self, n_features):
    self.weights = np.zeros(n_features)
    self.bias = 0

  def fit(self, X, y):
    y = np.asarray(y).reshape(-1,)
    assert X.ndim == 2
    assert y.ndim == 1
    n_samples, n_features = X.shape
    self.initialize_weights(n_features)
    for i in range(self.max_iter):
      z = (X@self.weights[:,None] + self.bias)#（n,d）@(d,1) = (n,1 )
      assert z.shape == (n_samples,1)
      y_hat = self.sigmoid(z)
      assert y_hat.shape == (n_samples,1)
      err = y_hat - y[:,None]
      assert err.shape == (n_samples,1)
      dw =1/n_samples*X.T@err#(d,n)@(n,1) = (d,1)
      db = 1/n_samples*np.sum(err)
      assert dw.shape == (n_features,1)
      self.weights -=self.lr*(dw.reshape(-1,))
      self.bias -=self.lr*db
      if np.linalg.norm(dw)<self.tol:
        break
    return self
  
  def predict_proba(self,X):
    z = (X@self.weights[:,None] + self.bias)
    y_hat = self.sigmoid(z)
    return y_hat
  
  def predict(self,X):
    y_hat = self.predict_proba(X)
    return (y_hat>self.treshold).astype(int)
  
