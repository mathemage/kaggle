import keras as K

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
