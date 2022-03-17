import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    #Memorize training data
    def train(self, X, y):
        self.Xtr = X
        self.ytr = y
    
    
    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        # For each test image: Find closest train image, Predict label of nearest image
        for i in xrange(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]