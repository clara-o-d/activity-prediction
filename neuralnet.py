import numpy as np
import os
import pandas as pd
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def assignments():
    num = random.random()
    if num < 0.6:
        return "TRAIN"
    elif num < 0.8:
        return "DEV"
    else:
        return "TEST"


all_solvents = pd.read_csv(os.path.expanduser("~/Downloads/baseline_with_ion_properties.csv"))
all_solvents["Assignment"] = [assignments() for _ in range(len(all_solvents)) ]

training_data = all_solvents[all_solvents["Assignment"] == "TRAIN"]
dev_data = all_solvents[all_solvents["Assignment"] == "DEV"]
test_data = all_solvents[all_solvents["Assignment"] == "TEST"]


X_COLS = [
    "molecule_formula",
    "std_dev_simplified",
    "ref_simplified",
    "max_molality_simplified",
    "std_dev_original",
    "max_molality_original",
    "r_M_angstrom",
    "r_X_angstrom",
    "anion_1_delta_G_formation",
    "anion_1_delta_G_hydration",
    "anion_1_diffusion_coefficient",
    "anion_1_molecular_weight",
    "anion_1_n_atoms",
    "anion_1_n_elements",
    "anion_1_radius_hydrated",
    "anion_1_radius_vdw",
    "anion_1_viscosity_jones_dole",
    "cation_1_delta_G_formation",
    "cation_1_delta_G_hydration",
    "cation_1_diffusion_coefficient",
    "cation_1_molecular_weight",
    "cation_1_n_atoms",
    "cation_1_n_elements",
    "cation_1_radius_hydrated",
    "cation_1_radius_vdw",
    "cation_1_viscosity_jones_dole",
    "molecule_molecular_weight",
    "molecule_n_atoms",
    "molecule_n_elements",
    "molecule_radius_vdw",
    "cation_type_numeric",
    "anion_type_numeric",
    "electrolyte_type_numeric"
]
Y_COLS = ["B_MX_0_original", "B_MX_1_original"
]


train_X = training_data[X_COLS].to_numpy().astype('float32')
train_Y = training_data[Y_COLS].to_numpy().astype('float32')
dev_X = dev_data[X_COLS].to_numpy().astype('float32')
dev_Y = dev_data[Y_COLS].to_numpy().astype('float32')

scaler = StandardScaler()
train_X = scaler.fit_transform(training_data[X_COLS].to_numpy().astype('float32'))
dev_X = scaler.transform(dev_data[X_COLS].to_numpy().astype('float32'))

tf_train = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).batch(128)
tf_dev = tf.data.Dataset.from_tensor_slices((dev_X, dev_Y)).batch(128)


L2_DECAY = 0.0
DROPOUT_RATE = 0.2
simple_model = tf.keras.Sequential([
    tf.keras.Input(shape=(train_X.shape[1],)),

    tf.keras.layers.Dense(64, activation=None,
                          kernel_regularizer=regularizers.l2(L2_DECAY)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(DROPOUT_RATE),

    tf.keras.layers.Dense(32, activation=None,
                          kernel_regularizer=regularizers.l2(L2_DECAY)),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(DROPOUT_RATE),

    tf.keras.layers.Dense(len(Y_COLS), activation='linear')
])

simple_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
              loss='mse',
              metrics=[ 'mse',RootMeanSquaredError()])

model_data = simple_model.fit(
    tf_train,
    epochs=1200,
    validation_data=tf_dev,
)

preds = simple_model.predict(dev_X)

mse = np.sqrt(model_data.history['mse'])
val_mse = np.sqrt(model_data.history['val_mse'])




# RMSE per target
rmse_per_target = np.sqrt(np.mean((dev_Y - preds)**2, axis=0))
print("RMSE for B_MX_0:", rmse_per_target[0])
print("RMSE for B_MX_1:", rmse_per_target[1])

overall_rmse = np.sqrt(np.mean((dev_Y - preds)**2))
print("Overall RMSE:", overall_rmse)


plt.figure(figsize=(8,5))
plt.plot(mse, label='Training RMSE')
plt.plot(val_mse, label='Validation RMSE')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('RMSE (log)')
plt.title('RMSE over Epochs (log)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()


actual0 = dev_Y[:, 0]
actual1 = dev_Y[:, 1]

pred0 = preds[:, 0]
pred1 = preds[:, 1]

# Plot for B0
plt.figure(figsize=(6,6))
plt.scatter(actual0, pred0, alpha=0.5)
plt.plot([actual0.min(), actual0.max()],
         [actual0.min(), actual0.max()], 'r--')
plt.xlabel('Actual B_MX_0')
plt.ylabel('Predicted B_MX_0')
plt.title('Predicted vs Actual — B_MX_0 validation data')
plt.grid(True)
plt.show()

# Plot for B1
plt.figure(figsize=(6,6))
plt.scatter(actual1, pred1, alpha=0.5)
plt.plot([actual1.min(), actual1.max()],
         [actual1.min(), actual1.max()], 'r--')
plt.xlabel('Actual B_MX_1')
plt.ylabel('Predicted B_MX_1')
plt.title('Predicted vs Actual — B_MX_1 validation data')
plt.grid(True)
plt.show()

from sklearn.metrics import mean_squared_error


X_test = test_data[X_COLS].to_numpy().astype('float32')
Y_test = test_data[Y_COLS].to_numpy().astype('float32')

X_test_scaled = scaler.transform(X_test)


test_loss, test_mse, test_rmse = simple_model.evaluate(X_test_scaled, Y_test, verbose=2)
print("Test MSE:", test_mse)
print("Test RMSE:", test_rmse)


Y_pred = simple_model.predict(X_test_scaled)

rmse_per_target = np.sqrt(np.mean((Y_test - Y_pred)**2, axis=0))
print("RMSE per target:", rmse_per_target)

overall_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
print("Overall RMSE:", overall_rmse)

# 3. Plot actual vs predicted for each target
for i, col in enumerate(Y_COLS):
    plt.figure(figsize=(6,6))
    plt.scatter(Y_test[:, i], Y_pred[:, i], alpha=0.5)
    mn = min(Y_test[:, i].min(), Y_pred[:, i].min())
    mx = max(Y_test[:, i].max(), Y_pred[:, i].max())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel(f"Actual {col}")
    plt.ylabel(f"Predicted {col}")
    plt.title(f"Predicted vs Actual — {col} test data")
    plt.grid(True)
    plt.show()

