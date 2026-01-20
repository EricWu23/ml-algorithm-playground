import numpy as np
from .random import get_rng

def make_binary_classification(
  n_samples = 100,
  n_features = 4,
  noise = 0.1,
  threshold =0.5,
  seed = 42
):
  rng = get_rng(seed)
  X = rng.normal(size=(n_samples,n_features))
  true_w = rng.normal(size=n_features)
  logits = X@ true_w
  probs = 1/(1+ np.exp(-logits))
  y= (probs>=threshold).astype(int)

  if noise>0:
    flip = rng.random(n_samples) < noise
    y[flip] = 1-y[flip]

  return X,y
