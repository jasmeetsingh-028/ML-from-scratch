import numpy as np
from collections import Counter

def euc_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    
    def __init__(self, k = 3):
        self.k = k
    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self,x_test):
        predicted_labels = [self._predict(x) for x in x_test]  #instanciating x_text
        return np.array(predicted_labels)
    
    def _predict(self,x):
        #calculating distances
        distances = [euc_distance(x, x_dash) for x_dash in self.x_train]  #calculating distances with each datapoint in x_train
        #getting k nearest distance indexes
        nearest_labels = np.argsort(distances)[:self.k]
        #getting labels for the nearest indexes
        k_nearest_labels = [self.y_train[label] for label in nearest_labels]
        #getting most common label
        occurence_count = Counter(k_nearest_labels)
        most_common_label = occurence_count.most_common(1)[0][0]

        return most_common_label
