from time import time
import tensorflow as tf
from tensorflow import keras

from BadChess.generator import create_tfdata_set

def create_generator() -> keras.models.Model:
    i = keras.Input(shape=(8, 8, 12), dtype=tf.float32)
    l = keras.layers.Flatten()(i)
    l = keras.layers.Dense(200)(l)
    l = keras.layers.Dense(200)(l)
    o = keras.layers.Dense(1, dtype=tf.float32)(l)

    return keras.models.Model(inputs=i, outputs=o)

def create_discriminator() -> keras.models.Model:
    i = keras.Input(shape=(1,), dtype=tf.float32)
    l = keras.layers.Dense(100)(i)
    l = keras.layers.Dense(100)(l)
    o = keras.layers.Dense(1, dtype=tf.float32)(l)
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

@tf.function
def train_step(batch_G, batch_D_eval):
    # Split batch_G into feature and label
    batch_G_bitboard, batch_G_eval = batch_G

    # Temporary squeeze
    batch_G_bitboard = tf.squeeze(batch_G_bitboard)
    batch_G_eval = tf.squeeze(batch_G_eval)
    batch_D_eval = tf.squeeze(batch_D_eval)

    with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
        # Propagate B_0 through G to get E_pred
        G_pred_eval = G(batch_G_bitboard, training=True)

        # Propagate E_pred through D to get T_pred_0
        D_pred_fake = D(G_pred_eval, training=True)

        # Propagate B_1 through D to get T_pred_1
        D_pred_real = D(batch_D_eval, training=True)

        # Use T_pred_0 and T_pred_1 to calculate (through BCE) L_D
        loss_D = loss_func_D(D_pred_real, D_pred_fake)

        # Use T_pred_0 and T_pred_1 to calculate (through BCE) L_G_E
        loss_G_extrinsic = loss_func_G(D_pred_fake)

        # Calculate the MAE on E vs E_pred, L_G_I
        loss_G_intrinsic = mae(G_pred_eval, batch_G_eval)

        # Calculate L_G = L_G_I + L_G_E
        loss_G = tf.add(loss_G_intrinsic, tf.multiply(1, loss_G_extrinsic))

    G_rmse.update_state(G_pred_eval, batch_G_eval)
    D_accuracy.update_state(D_pred_real, tf.ones_like(D_pred_real))
    D_accuracy.update_state(D_pred_fake, tf.zeros_like(D_pred_fake))

    # Calculate DL_G, DL_D
    dL_G = G_tape.gradient(loss_G, G.trainable_weights)
    dL_D = D_tape.gradient(loss_D, D.trainable_weights)

    # Apply gradient updates to G, D
    G_opt.apply_gradients(zip(dL_G, G.trainable_weights))
    D_opt.apply_gradients(zip(dL_D, D.trainable_weights))

def train():
    N_epochs = 100
    batch_size = 20
    chunk_size = 1
    n_items = 5000
    display_step = n_items // 100

    train_dataset_G = create_tfdata_set(n_items=n_items, batch_size=batch_size, chunk_size=chunk_size)
    train_dataset_D = create_tfdata_set(n_items=n_items, batch_size=batch_size, chunk_size=chunk_size)

    for epoch in range(N_epochs):
        print(f"Epoch {epoch+1}", end="")
        start = time()
        print("(", end="")
        for step, (batch_G, (_, batch_D)) in enumerate(zip(train_dataset_G, train_dataset_D)):
            if not step % display_step:
                print("=", end="")
            train_step(batch_G, batch_D)
        print(") - ", end="")

        dt = time() - start
        print(f"Generator RMSE: {G_rmse.result()}", end=" - ")
        print(f"Discriminator Accuracy: {D_accuracy.result()}", end=" - ")
        print(f"({dt:.4f}s)")

        G_rmse.reset_states()
        D_accuracy.reset_states()

train()