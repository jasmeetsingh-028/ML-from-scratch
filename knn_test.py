import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from knn import KNN

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

print(x_train.shape,y_train.shape)
print(x_train[:5])

# for x in x_train:
#     print(x.shape)
#     break

# plt.figure()
# plt.scatter(x_train[:, 0], x_train[:, 1], c = y_train, cmap = cmap, edgecolors='k', s=20)
# plt.show()

knn_model = KNN(k = 3)
knn_model.fit(x_train = x_train, y_train = y_train)
predictions = knn_model.predict(x_test)
print(predictions)