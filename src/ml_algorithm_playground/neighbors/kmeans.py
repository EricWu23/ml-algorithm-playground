import numpy as np

class Kmeans:
    def __init__(self, k=5, max_iters=100, tol=1e-4, rng = np.random.default_rng(seed=42)):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.rng = rng
        self._centroids = None#(k,D)
        self._old_centroids = None
        self._labels = None# (N,)
        self._iteration_counter=0
        self._shift = 0
        self._distance=None



    def __initialize__(self,X):
        """
        Initialize the K-Means model with random centroids.

        Parameters:
        X (numpy.ndarray): The input data.

        Returns:
        None

        """
        indexes = self.rng.choice(X.shape[0], self.k, replace=False)
        self._centroids = X[indexes]#(K,D)
        return self

    def __em_check__(self):
        """
        Check if stop condition is met.

        Parameters:
        None

        Returns:
        bool: True if stop condition is met, False otherwise.

        """
        if self._iteration_counter == 0:
          self._shift = self.tol+1
        else:
          self._shift = np.linalg.norm(self._centroids - self._old_centroids)
        stop =self._iteration_counter > self.max_iters or self._shift < self.tol
        return stop

    def __em_assign__(self,X):
        """
        Assign each data point to the nearest centroid.
        X = (N,D)
        centroids = (K,D)
        ||X-centroids||^2 = (N,K) =X^2 + centroids^2 - 2*X*centroids

        distance: (N,1) + (1，K) - (N,K) --> (N,K)
        self.labels: (N,K) --> (N,)

        Parameters:
        None

        Returns:
        None
        """
        c_square = (self._centroids**2).sum(axis=1)#(K,)
        c_square = c_square[None,:]#(1,K)
        self._distance = (X**2).sum(axis=1,keepdims=True)+ c_square - 2*X@self._centroids.T
        self._labels = np.argmin(self._distance,axis=1)
        return self


    def __em_update__(self,X):
        """
        Update the centroids based on the current labels through recomputing centroids of each cluster.

        Parameters:
        None

        Returns:
        None

        """
        self._old_centroids = self._centroids#(K,D)
        for k in range(self.k):
          mask = (self._labels == k)
          if mask.any():
            self._centroids[k]=X[mask].mean(axis=0)
          else:
            self._centroids[k] = self._old_centroids[k]
        return self



    def fit(self, X):
        """
        Fit the K-Means model to the input data.

        Parameters:
        X (numpy.ndarray): The input data.

        Returns:
        Kmeans: The fitted Kmeans model.

        """
        n_samples, n_features = X.shape
        self.__initialize__(X) # randomly initialize k centroids
        stop = self.__em_check__()
        while not stop:
          self.__em_assign__(X)# according to centroids, assign label
          self.__em_update__(X)# update centroids according to label

          self._iteration_counter +=1
          stop = self.__em_check__()
          print(f"Iteration: {self._iteration_counter}, Shift: {self._shift}")
        return self

    def predict(self, X):
        """
        Predict the cluster labels for each sample in X.

        Parameters:
        X (numpy.ndarray): The input data. shape = (n_samples, n_features)

        Returns:
        Y (numpy.ndarray): The predicted cluster labels. shape = (n_samples,)

        """
        self._iteration_counter = 0
        self._labels = None
        self.__em_assign__(X)
        return self._labels