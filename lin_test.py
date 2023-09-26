import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt 
from LinearRegression import Linear

X,y = datasets.make_regression(n_samples=100, noise=20, n_features=1, random_state=4)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1234)

# plt.scatter(x_train[:,0],y_train, color = 'b', marker = 'o')
# plt.show()
print(x_train.shape, y_train.shape)

model = Linear()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
mse = np.mean((y_test - predictions)**2)
print(mse)