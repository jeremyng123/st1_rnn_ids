from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv("train_5feats.csv", header=0)
test = pd.read_csv("test_5feats.csv", header=0)

trainlen = len(train)
testlen = len(test)

trainx = train.drop(['Malicious'], axis=1)
trainx = np.asarray(trainx)

trainy = train['Malicious']
trainy = np.asarray(trainy)

data = np.array(trainx, dtype=float)
data = np.reshape(data, (125973, 5, 1))
target = np.array(trainy, dtype=float)

x_train, x_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=4)

model = Sequential()
model.add(LSTM((1), batch_input_shape=(None, 5, 1), return_sequences=True))
model.add(LSTM((1), return_sequences=False))
model.compile(loss='mean_absolute_error',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    validation_data=(x_test, y_test))

results = model.predict(x_test)

# first 50 points
plt.scatter(range(20), results[:20], c='red', s=50)
plt.scatter(range(20), y_test[:20], c='green', s=20)
plt.show()

plt.plot(history.history['loss'])
plt.show()

testx = test.drop(['Malicious'], axis=1)
testx = np.asarray(testx)
testx = np.array(testx, dtype=float)
testx = np.reshape(testx, (22543, 5, 1))

testy = test['Malicious']
testy = np.asarray(testy)
testy = np.array(testy, dtype=float)

results = model.predict(testx)

# 20 points
plt.scatter(range(20), results[:20], c='red', s=50)
plt.scatter(range(20), testy[:20], c='green', s=20)
plt.show()

# get performance of model
tp = 0
fp = 0
tn = 0
fn = 0
pred = []
for i in results:
    for j in range(len(i)):
        if i[j] >= 0.5:
            output = 1
        else:
            output = 0
        pred.append(output)

        if output == testy[j]:
            if output == 1:
                tp += 1
            else:
                tn += 1
        else:
            if output == 1:
                fp += 1
            else:
                fn += 1


def getmetrics(tp, tn, fp, fn):
    # accuracy, recall, precision
    try:
        accuracy, recall, precision = (tp + tn) / (tp + tn + fp + fn), tp / (
            tp + fn), tp / (tp + fp)
        return accuracy, recall, precision
    except ZeroDivisionError:
        return 0, 0, 0


accuracy, recall, precision = getmetrics(tp, tn, fp, fn)

print(accuracy, precision, recall)
