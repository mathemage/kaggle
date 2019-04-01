import keras as K

# hyperparameters
input_dim = None  # TODO

# initialize
model = K.Sequential()

# architecture
model.add(K.layers.Dense(units=1, activation='sigmoid', input_dim=input_dim))

# loss
model.compile(loss=K.losses.categorical_crossentropy, optimizer=K.optimizers.Adam)
