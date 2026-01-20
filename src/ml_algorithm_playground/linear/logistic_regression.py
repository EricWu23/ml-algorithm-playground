import numpy as np
class LogisticRegression:
  def __init__(self, threshold=0.5,lr=0.01, max_iter=1000, tol=1e-4):
    self.threshold = threshold
    self.lr = lr
    self.max_iter = max_iter
    self.tol = tol
    self.weights = None
    self.bias = None


  def sigmoid(self, z):
    z=np.clip(z, -500, 500)
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
      z = (X@self.weights + self.bias)#（n,d）@(d,) = (n,)
      assert z.shape == (n_samples,)
      y_hat = self.sigmoid(z)
      assert y_hat.shape == (n_samples,)
      err = y_hat - y
      assert err.shape == (n_samples,)
      dw =1/n_samples*X.T@err#(d,n)@(n,) = (d,)
      db = 1/n_samples*np.sum(err)
      assert dw.shape == (n_features,)
      self.weights -=self.lr*dw
      self.bias -=self.lr*db
      if np.linalg.norm(dw)<self.tol:
        break
    return self
  
  def predict_proba(self,X):
    z = (X@self.weights + self.bias)
    y_hat = self.sigmoid(z)
    return y_hat
  
  def predict(self,X):
    y_hat = self.predict_proba(X)
    return (y_hat>=self.threshold).astype(int)
  
