import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

M = 2 # 入力データの次元
K = 3 # クラス数
n = 100 # クラスごとのデータ数
N = n * K # 全データ数

# サンプルデータ生成
X1 = np.random.randn(n, M) + np.array([0, 10])
X2 = np.random.randn(n, M) + np.array([5, 5])
X3 = np.random.randn(n, M) + np.array([10, 0])
Y1 = np.array([[1, 0, 0] for i in range(n)])
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])

X = np.concatenate((X1, X2, X3), axis=0)
Y = np.concatenate((Y1, Y2, Y3), axis=0)

# モデル定義
model = Sequential()
model.add(Dense(input_dim=M, units=K))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1))

# 学習
minibatch_size = 50
model.fit(X, Y, epochs=20, batch_size=minibatch_size)

# 結果確認
X_, Y_=shuffle(X, Y)
classes = model.predict_classes(X_[0:10], batch_size=minibatch_size)
prob = model.predict_proba(X_[0:10], batch_size=1)

print('classified:')
print(np.argmax(model.predict(X_[0:10]), axis=1) == classes)
print()
print('output probability:')
print(prob)

# グラフ描画
print(model.get_weights())
w = model.get_weights()[0]
b = model.get_weights()[1]

def border(x, c1, c2):
    return((w[0, c1] - w[0, c2]) * x - b[c1] + b[c2]) / (w[1, c2] - w[1, c1])

plt.plot(X1[:, 0], X1[:, 1], "o")
plt.plot(X2[:, 0], X2[:, 1], "o")
plt.plot(X3[:, 0], X3[:, 1], "o")
plt.plot([-2, 5], [border(-2, 0, 1), border(5, 0, 1)])
plt.plot([-2, 12], [border(-2, 1, 2), border(12, 1, 2)])
plt.show()
