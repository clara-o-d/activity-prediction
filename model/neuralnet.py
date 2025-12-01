import numpy as np
import pandas as pd
import tensorflow as tf

def assignments():
    num = random.random()
    if num < 0.6:
        return "TRAIN"
    elif num < 0.8:
        return "DEV"
    else:
        return "TEST"

"TO DO: SET ALL_SOLVENTS TO ACTUAL LOADABLE DATASET"
all_solvents = None
all_solvents["Assignment"] = [assignments() for _ in range(len(all_solvents)) ]

training_data = all_solvents[all_solvents["Assignment"] == "TRAIN"]
dev_data = all_solvents[all_solvents["Assignment"] == "DEV"]
test_data = all_solvents[all_solvents["Assignment"] == "TEST"]

"TO DO: FILL IN COLUMNS"
X_COLS = ["Organic", "Molar_Mass", ...

]
Y_COLS = ['Activity_Coefficient'

]

train_X = training_data[X_COLS].to_numpy().astype('float32')
train_Y = training_data[Y_COLS].to_numpy().astype('float32')
dev_X = dev_data[X_COLS].to_numpy().astype('float32')
dev_Y = dev_data[Y_COLS].to_numpy().astype('float32')

tf_train = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).batch(128)
tf_dev = tf.data.Dataset.from_tensor_slices((dev_X, dev_Y)).batch(128)

"TO DO: MAKE MODEL NOT TOTALLY RANDOM"

simple_model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(train_X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(Y_COLS), activation='linear')
])

simple_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae', 'mse'])

model_data = model.fit(
    tf_train,
    epochs=500,
    validation_data=tf_dev,
)

predictions = model.predict(dev_X)
print(pd.DataFrame(predictions))