from time import time
import tensorflow as tf
from tensorflow import keras

from BadChess.generator import create_tfdata_set

def create_generator():
    i = keras.Input(8, 8, 12)
    l = keras.layers.Flatten()(i)
    l = keras.layers.Dense(200)(l)
    l = keras.layers.Dense(200)(l)
    o = keras.layers.Dense(1)(l)

    return keras.models.Model(inputs=i, outputs=o, name="Generator")

def create_discriminator():
    i = keras.Input(1)
    l = keras.Dense(100)(i)
    l = keras.Dense(100)(l)
    o = keras.Dense(1)(l)
    return keras.models.Model(inputs=i, outputs=o, name="Discriminator")

G_opt = tf.keras.optimizers.Adam()
D_opt = tf.keras.optimizers.Adam()

bce = keras.losses.BinaryCrossentropy()
mae = keras.losses.MeanAbsoluteError()

def loss_func_G(fake_out):
    return bce(tf.ones_like(fake_out), fake_out)

def loss_func_D(real_out, fake_out):
    real_loss = bce(tf.ones_like(real_out), real_out)
    fake_loss = bce(tf.zeros_like(fake_out), fake_out)
    return tf.add(real_loss, fake_loss)

G = create_generator()
D = create_discriminator()

G_rmse = tf.keras.metrics.RootMeanSquaredError()
D_accuracy = tf.keras.metrics.Accuracy()

def train_step(batch_G, batch_D_eval):
    # Split batch_G into feature and label
    batch_G_bitboard, batch_G_eval = batch_G

    with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
        # Propagate B_0 through G to get E_pred
        G_pred_eval = G(batch_G_bitboard, training=True)

        # Propagate E_pred through D to get T_pred_0
        truth_pred_fake = D(G_pred_eval, training=True)

        # Propagate B_1 through D to get T_pred_1
        truth_pred_real = D(batch_D_eval, training=True)

        # Use T_pred_0 and T_pred_1 to calculate (through BCE) L_D
        loss_D = loss_func_D(truth_pred_real, truth_pred_fake)

        # Use T_pred_0 and T_pred_1 to calculate (through BCE) L_G_E
        loss_G_extrinsic = loss_func_G(truth_pred_fake)

        # Calculate the MAE on E vs E_pred, L_G_I
        loss_G_intrinsic = mae(G_pred_eval, batch_G_eval)

        # Calculate L_G = L_G_I + L_G_E
        loss_G = tf.add(loss_G_intrinsic, tf.multiply(1, loss_G_extrinsic))

    G_rmse.update_state(G_pred_eval, batch_G_eval)
    D_accuracy.update_state(truth_pred_real, 1)
    D_accuracy.update_state(truth_pred_fake, 0)

    # Calculate DL_G, DL_D
    dL_G = G_tape.gradient(loss_G)
    dL_D = D_tape.gradient(loss_D)

    # Apply gradient updates to G, D
    G_opt.apply_gradients(zip(dL_G, G.trainable_weights))
    D_opt.apply_gradients(zip(dL_D, D.trainable_weights))

def train():
    N_epochs = 10
    batch_size = 10
    chunk_size = 3

    train_dataset_G = create_tfdata_set(batch_size=batch_size, chunk_size=chunk_size)
    train_dataset_D = create_tfdata_set(batch_size=batch_size, chunk_size=chunk_size)
    ds_iter = zip(train_dataset_G, train_dataset_D)

    for epoch in N_epochs:
        print(f"Epoch {epoch} - ", end="")
        start = time()

        for (batch_G, (_, batch_D)) in ds_iter:
            train_step(batch_G, batch_D)

        dt = time() - start
        print(f"Generator RMSE: {G_rmse.result()}", end=" - ")
        print(f"Discriminator Accuracy: {D_accuracy.result()}", end=" - ")
        print(f"({dt})")

        G_rmse.reset_states()
        D_accuracy.reset_states()



