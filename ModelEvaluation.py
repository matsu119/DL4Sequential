from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
# from keras.optimizers import SGD
from keras.optimizers import adam
import numpy as np
import matplotlib.pyplot as plt

N = 300
X, y = datasets.make_moons(N, noise=0.3)

# グラフ表示
X0 = X[y == 0,:]
X1 = X[y == 1,:]
plt.plot(X0[:, 0], X0[:, 1], "o")
plt.plot(X1[:, 0], X1[:, 1], "o")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

model = Sequential()
model.add(Dense(3, input_dim=2))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=adam(lr=0.05), metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1000, batch_size=20)

loss_and_metrics = model.evaluate(X_test, y_test)

print(loss_and_metrics)

# グラフ表示
x0_min = X[:, 0].min() - 0.5
x0_max = X[:, 0].max() + 0.5
x1_min = X[:, 1].min() - 0.5
x1_max = X[:, 1].max() + 0.5
grid_interval = 0.01
x0, x1 = np.meshgrid(np.arange(x0_min, x0_max, grid_interval), np.arange(x1_min, x1_max, grid_interval))
grid_shape=x0.shape

x0 = x0.reshape(-1, 1)
x1 = x1.reshape(-1, 1)

grid = np.concatenate((x0, x1), axis=1)

bg_classes = model.predict(grid)

bg_classes = bg_classes.reshape(grid_shape)
plt.contourf(np.arange(x0_min, x0_max, grid_interval), np.arange(
    x1_min, x1_max, grid_interval), bg_classes, alpha=0.2)
plt.plot(X0[:, 0], X0[:, 1], "o")
plt.plot(X1[:, 0], X1[:, 1], "o")
plt.show()
