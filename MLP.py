import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

model = Sequential()

# 入力層 - 隠れ層
model.add(Dense(input_dim=2, units=2))
model.add(Activation('sigmoid'))

# 隠れ層 - 出力層
model.add(Dense(units=1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

model.fit(X, Y, epochs=4000, batch_size=4)

classes = model.predict_classes(X, batch_size=4)
prob = model.predict_proba(X, batch_size=4)

print('classified:')
print(Y == classes)
print()
print('output probability:')
print(prob)

'''
結果：
classified:
[[ True]
 [ True]
 [False]
 [False]]
output probability:
[[0.49206045]
 [0.52958643]
 [0.48314384]
 [0.5077603 ]]
'''
