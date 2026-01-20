import numpy as np
from ml_algorithm_playground.linear.logistic_regression import LogisticRegression

def test_logistic_regression_learns_linearly_seperable_data():
  rng = np.random.default_rng(42)
  X = rng.normal(size=(100,2))
  y = ((X[:,0] - X[:,1])>0).astype(int)

  model = LogisticRegression(threshold=0.5,lr=0.01, max_iter=10000, tol=0)
  model.fit(X,y)

  pred =model.predict(X).reshape(-1,)
  acc = (pred == y).mean()

  assert acc > 0.7