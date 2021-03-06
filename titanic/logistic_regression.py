import keras as K
import numpy as np
import pandas as pd

# datasets
feature_columns = [2, 4, 5, 6, 7, 9, 11]  # "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"
target_column = [1]  # "Survived"
dataset = pd.read_csv("train.csv", quotechar='"', skipinitialspace=True, usecols=feature_columns + target_column)
if __debug__:
    print(dataset)
X = dataset.iloc[:, 1:]
Y = dataset.iloc[:, :1]
print("X: {}".format(X))
print("Y: {}".format(Y))

# TODO
x_train = None
y_train = None
x_dev = None
y_dev = None
x_test = None
y_test = None

dataset_predict = np.loadtxt("test.csv", delimiter=",", skiprows=1)
x_predict = None

# hyperparameters
input_dim = None  # TODO
epochs = 5
batch_size = 32

# initialize
model = K.Sequential()

# architecture
model.add(K.layers.Dense(units=1, activation='sigmoid', input_dim=input_dim))

# loss
model.compile(loss=K.losses.categorical_crossentropy, optimizer=K.optimizers.Adam)

# train: x_train and y_train are Numpy arrays
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)  # TODO

# evaluate: dev set
loss_and_metrics_dev = model.evaluate(x_dev, y_dev, batch_size=batch_size)

# evaluate: test set
loss_and_metrics_test = model.evaluate(x_test, y_test, batch_size=batch_size)

# predict
classes = model.predict(x_predict, batch_size=batch_size)
