import numpy as np
class YWKNNClassifier:
    '''
    written by Yujiang Wu on 01/31/2026 for educational purpose:
    config & params: 
        n_neighbors, default=5 : number of neighbors to use (k in kNN)
    initialization: 
        N/A
    fit: 
        1. Storing training data X and labels y as self.X_train and self.y_train
        2. Precompute squared sums of training data for vectorized distance computation
        3. Store unique class labels and label indexes for all training labels for later use
        4. Ensure n_neighbors does not exceed number of training samples

    predict: 
        1. Compute squared Euclidean distances between test samples and training samples in a vectorized manner
        2. For each test sample, find the indices of the k nearest neighbors from training data
        3. Perform majority voting among the k neighbors to determine predicted class label for each test sample,
           breaking ties by choosing the smallest label
        

    shapes:
        X: (n_samples, n_features), each feature is n_features-dimensional numerical vector. So any categorical features should be preprocessed into numerical format beforehand.
        y: (n_samples, C), labels are class labels or label index, C is number of classes. 
        X_test (X in predict): (m_samples, n_features), m_samples is number of test samples
        y_pred: (m_samples,), predicted class labels for test samples
    
    numpy functions used:
        np.unique, np.argpartition, np.add.at, np.argmax
    '''

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.X_train_sqsum_  = None
        self.classes_ = None
        self.y_indexes_ = None
    
    def fit(self, X, y):
        self.X_train = X  # (n,d)
        self.X_train_sqsum_  = (X**2).sum(axis=1,keepdims=True)# (n,d) -> (n,1)
        self.y_train = y#(n,)
        self.classes_,self.y_indexes_ = np.unique(self.y_train, return_inverse=True)
        self.n_neighbors = min(self.n_neighbors, self.X_train.shape[0])
        # self.classes_: (q,) unique class labels
        # self.y_indexes_: (n,) index of class for each sample
        return self
    
    def predict(self, X):
        n_test,d = X.shape
        X_sqsum_= (X**2).sum(axis=1,keepdims=True)  #(n_test,1)
        # compute distances
        distance_matrix = self.X_train_sqsum_ .T + X_sqsum_ -2*X@self.X_train.T  # (1,n) + (n_test,1) -(n_test,n)
        distance_matrix = np.maximum(distance_matrix, 0.0)# numerical stability
        # get top k indices
        topk_indices =np.argpartition(distance_matrix,
                                      kth=self.n_neighbors-1, 
                                      axis=1)[:,:self.n_neighbors]#(n_test,k)
        
        # majority vote with ties broken by smallest label
        nn_y_idx = self.y_indexes_[topk_indices] #(n_test,k)
        C = len(self.classes_)
        counts = np.zeros((n_test, C), dtype=int) # (n_test, C)
        np.add.at(counts, (np.arange(n_test)[:, None], nn_y_idx), 1)    # count by every rowm, every class #(n_test,C)
        y_pred = self.classes_[np.argmax(counts, axis=1)] # (n_test,)
        return y_pred