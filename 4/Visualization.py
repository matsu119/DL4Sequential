import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.layers.core import Dropout
from keras.initializers import TruncatedNormal
# from keras import backend as K
import matplotlib.pyplot as plt

# def weight_variable(shape):
#     return K.truncated_normal(shape, stddev=0.01)

mnist = datasets.fetch_mldata('MNIST original', data_home='.')

N_train = 20000
N_validation = 4000

n = len(mnist.data)
N = 30000  # MNISTの一部のデータで実験
indices = np.random.permutation(range(n))[:N]  # ランダムにN枚を選択
X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]  # 1-of-K表現に変換

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=N_train)
# 訓練データをさらに訓練データと検証データに分割
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=N_validation)

'''
モデル設定
'''
n_in = len(X[0])  # 784
n_hiddens = [200, 200, 200]
n_out = len(Y[0])  # 10
p_keep = 0.5
activation = 'relu'

model = Sequential()
for i, input_dim in enumerate(([n_in] + n_hiddens)[:-1]):
    model.add(Dense(n_hiddens[i], input_dim=input_dim, kernel_initializer=TruncatedNormal(stddev=0.01)))
    model.add(Activation(activation))
    model.add(Dropout(p_keep))

model.add(Dense(n_out, kernel_initializer=TruncatedNormal(stddev=0.01)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01), metrics=['accuracy'])

'''
モデル学習
'''
epochs = 50
batch_size = 200

hist = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_validation, Y_validation))

# print(hist.history['val_loss'])
# print(hist.history['val_acc'])

val_acc = hist.history['val_acc']

plt.rc('font', family='serif')
fig=plt.figure()
plt.plot(range(epochs), val_acc, label='acc', color='black')
plt.xlabel('epochs')
plt.show()
# plt.savefig('mnist_keras.eps')

'''
予測精度の評価
'''
loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)
