import numpy as np

class Linear:
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr  = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, x_train,y_train):
        '''
        X_train.shape format -> (n_samples, n_features)

        '''
        n_samples, n_features = x_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):

            #calculating y_pred
            y_pred = np.dot(x_train, self.weights) + self.bias

            #calculating gradient wrt weights and bias
            dw = (1/n_samples) * np.dot(x_train.T, (y_pred - y_train))  #shape check for x_train
            db = (1/n_samples) * np.sum((y_pred - y_train))

            #weights and bias updation
            self.weights -= (self.lr*dw)
            self.bias -= (self.lr*db)

    def predict(self, x_test):

        '''
        X_test.shape format -> (n_samples, n_features)
        
        '''
        predictions = np.dot(x_test,self.weights) + self.bias

        return predictions
    

