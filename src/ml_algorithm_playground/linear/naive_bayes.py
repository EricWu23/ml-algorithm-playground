import numpy as np
class MultinomialNaiveBayes:
    """
    written by Yujiang Wu on 01/29/2026 for educational purpose:
    config & params: alpha = 1.0 (Laplace smoothing)
    initialization: N/A
    fit: P(Y|X) = P(X|Y)P(Y) / P(X), Y_hat = argmax P(Y|X) = argmax log(P(X|Y)P(Y)) = argmax (log P(X|Y) + log P(Y)) 
    = argmax (sum(log P(x_i|Y)) + log P(Y)), i = 1,...,d
    predict: 
        Y_hat = argmax (sum(xtest_i*log P(feature_i|Y)) + log P(Y)), i = 1,...,d

    shapes:
        X: (n_samples, n_features), each feature is count of a word occurrences
        y: (n_samples, ), labels are 0/1 for not spam/spam or topic categories (q classes)
        X_test: (m_samples, n_features), m_samples is number of test samples
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None
    
    def _initialization(self, X, y):
        #assert X.dim == 2
        #if y.ndim == 1:
        #    y = y.reshape(-1, 1)
        n,d = X.shape
        self.classes_ = np.unique(y)
        self.class_log_prior_ = np.zeros(len(self.classes_))
        self.feature_log_prob_ = np.zeros((len(self.classes_), d))
        return n, d
    
    def fit(self, X, y):
        """The main goal of fitting is to compute class log prior and feature log prob"""
        n,d= self._initialization(X, y)
        self.class_log_prior_ = np.log(np.array([np.sum(y==c) for c in self.classes_])/n)
        #label to index mapping
        class_to_index = { c: i for i,c in enumerate(self.classes_)}
        y_indexes = np.array([class_to_index[label] for label in y])
        # label to one-hot encoding
        q = len(self.classes_)
        Y = np.eye(q)[y_indexes]                                # (n,q)
        # use one-hot as mask to sum feature counts per class  over samples
        feature_count = Y.T @ X                             # (q,d)
        # add smoothing (Laplace smoothing)
        smoothed_fc = feature_count + self.alpha #(q,d)
        smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1,1)  #(q,1)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)  #(q,d)
        return self
    
    def predict(self, X_test):
        """predict the class labels for samples in X_test
        X_test: (m_samples, n_features)
        """
        #assert X_test.ndim == 2
        n_test, d = X_test.shape
        y_pred_matrix = X_test@self.feature_log_prob_.T + self.class_log_prior_ # (n_test,q)
        y_pred_indexes = np.argmax(y_pred_matrix, axis=1)  # (n_test,)

        y_pred = np.array([self.classes_[i] for i in y_pred_indexes])  # (n_test,)

        return y_pred