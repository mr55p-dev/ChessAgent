from abc import abstractmethod
from math import inf
from pathlib import Path
from readline import add_history
from typing import Iterable, List, Tuple
import tensorflow as tf
from tensorflow import keras
from tqdm import trange
from BadChess.metrics import Loss


# Function types
T_model = keras.models.Model
T_tensor = tf.Tensor
T_metrics = List[tf.keras.metrics.Metric]
T_opt = keras.optimizers.Optimizer
T_cblist = keras.callbacks.CallbackList

# Loss functions to be used at the module level
mse = tf.keras.losses.MeanSquaredError()
bce = tf.keras.losses.BinaryCrossentropy()

class BaseGAN():
    def __init__(self) -> None:
        # Models
        self.generator: T_model = self.create_generator()
        self.discriminator: T_model = self.create_discriminator()

        # Model optimizers
        self.G_opt: T_opt = tf.keras.optimizers.Adam()
        self.D_opt: T_opt = tf.keras.optimizers.Adam()

        # Model metrics
        self.M_gen_rmse = tf.keras.metrics.MeanSquaredError()
        self.M_dis_accuracy = tf.keras.metrics.Accuracy()
        self.M_gen_loss = Loss(name="gen_loss")
        self.M_dis_loss = Loss(name="dis_loss")

        # Model logs
        self.logs = {
            "gen loss" : [],
            "dis loss" : [],
            "gen rmse" : [],
            "dis acc" : [],
        }

    @staticmethod
    @abstractmethod
    def create_generator() -> T_model:
        """Function returning an uncompiled keras model for the generator"""
        ...

    @staticmethod
    @abstractmethod
    def create_discriminator() -> T_model:
        """Function returning uncompiled keras model for the discriminator"""
        ...

    @abstractmethod
    def intrinsic_generator_loss(
        self,
        prediction: T_tensor,
        truth: T_tensor) -> T_tensor:
        """Function computing the generator loss on the evaluation"""
        ...

    @abstractmethod
    def extrinsic_generator_loss(
        self,
        discriminator_output: T_tensor) -> T_tensor:
        """Function computing the generator loss based on the discriminators categotisation"""
        ...

    @abstractmethod
    def discriminator_loss(
        self,
        real_guess: T_tensor,
        fake_guess: T_tensor
        ) -> T_tensor:
        """Function computing the discriminators total loss"""
        ...

    @tf.function
    def _trainstep(self, batch_G: Tuple[T_tensor, T_tensor], batch_D: T_tensor):
        """Implements training of one batch of data"""
        # Split batch_G into feature and label
        batch_G_bitboard, batch_G_eval = batch_G

        # Temporary squeeze
        batch_G_bitboard = tf.squeeze(batch_G_bitboard)
        batch_G_eval = tf.squeeze(batch_G_eval)
        batch_D = tf.squeeze(batch_D)

        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            # Propagate B_0 through G to get E_pred
            G_pred_eval = self.generator(batch_G_bitboard, training=True)

            # Propagate E_pred through D to get T_pred_0
            D_pred_fake = self.discriminator(G_pred_eval, training=True)

            # Propagate B_1 through D to get T_pred_1
            D_pred_real = self.discriminator(batch_D, training=True)

            # Use T_pred_0 and T_pred_1 to calculate (through BCE) L_D
            loss_D = self.discriminator_loss(D_pred_real, D_pred_fake)

            # Use T_pred_0 and T_pred_1 to calculate (through BCE) L_G_E
            loss_G_extrinsic = self.extrinsic_generator_loss(D_pred_fake)

            # Calculate the MAE on E vs E_pred, L_G_I
            loss_G_intrinsic = self.intrinsic_generator_loss(G_pred_eval, batch_G_eval)

            # Calculate L_G = L_G_I + L_G_E
            loss_G = tf.add(
                loss_G_intrinsic,
                tf.multiply(
                    tf.constant(1, dtype=tf.float32),
                    loss_G_extrinsic
                )
            )

        # Update metric state
        self.M_gen_rmse.update_state(G_pred_eval, batch_G_eval)
        self.M_dis_accuracy.update_state(tf.clip_by_value(tf.round(D_pred_real), 0, 1), tf.ones_like(D_pred_real))
        self.M_dis_accuracy.update_state(D_pred_fake, tf.zeros_like(D_pred_fake))
        self.M_gen_loss.update_state(loss_G)
        self.M_dis_loss.update_state(loss_D)

        # Calculate DL_G, DL_D
        dL_G = G_tape.gradient(loss_G, self.generator.trainable_weights)
        dL_D = D_tape.gradient(loss_D, self.discriminator.trainable_weights)

        # Apply gradient updates to G, D
        self.G_opt.apply_gradients(zip(dL_G, self.generator.trainable_weights))
        self.D_opt.apply_gradients(zip(dL_D, self.discriminator.trainable_weights))

    def get_progbar_values(self):
        """Logs and stores metric state at the end of an epoch"""
        gen_loss = self.M_gen_loss.result()
        dis_loss = self.M_dis_loss.result()
        gen_rmse = self.M_gen_rmse.result()
        dis_acc = self.M_dis_accuracy.result()

        self.logs["gen loss"].append(gen_loss)
        self.logs["dis loss"].append(dis_loss)
        self.logs["gen rmse"].append(gen_rmse)
        self.logs["dis acc"].append(dis_acc)

        return [
                ("Gen loss", gen_loss),
                ("Dis loss", dis_loss),
                ("Gen RMSE", gen_rmse),
                ("Dis accuracy", dis_acc)
            ]

    def train(
        self,
        n_epochs: int,
        ds_train_generator: Iterable,
        ds_train_discriminator,
        ds_val_generator = None,
        ds_val_discriminator = None,
        ):
        """Train the model over the data for the specified number of epochs"""
        self.N_epochs = 100
        self.batch_size = 20
        self.chunk_size = 1
        self.n_items = 5000

        # Histories
        G_hist = tf.keras.callbacks.History()
        D_hist = tf.keras.callbacks.History()

        G_hist.set_model(self.generator)
        D_hist.set_model(self.discriminator)


        # Populate with typical keras callbacks
        _callbacks = [
            G_hist,
            D_hist
        ]

        target_steps = None # ds_train.cardinality
        for epoch in range(n_epochs):
            # Reset the metrics
            self.M_gen_loss.reset_states()
            self.M_gen_rmse.reset_states()

            self.M_dis_loss.reset_states()
            self.M_dis_accuracy.reset_states()

            # Display the current epoch and instantiat the progress bar
            print(f"Epoch {epoch+1}")
            progbar = tf.keras.utils.Progbar(target=target_steps)

            # Zip the training data and enumerate it (batchwise)
            for step, (batch_G, batch_D) in enumerate(zip(ds_train_generator, ds_train_discriminator)):

                # Perform the training step
                self._trainstep(batch_G, batch_D)
                progbar.update(
                    step,
                    self.get_progbar_values()
                )

            # Set the number of steps if it is unknown
            if not target_steps:
                target_steps = step
        # callbacks.on_train_end()
        return self.logs

    def save_generator(self, path: Path) -> None:
        self.generator.save(str(path))

class RNNGAN(BaseGAN):
    """
    RNN based GAN formulation, with convolutional layers to parse the Chess bitboard.
    Uses a simple RNN for the discriminator
    """
    @staticmethod
    def create_generator() -> T_model:

        wrap_td = lambda x: keras.layers.TimeDistributed(x)

        i = keras.Input(shape=(4, 8, 8, 12), dtype=tf.float32)
        l = wrap_td(keras.layers.Conv2D(34, (3, 3), padding="same"))(i)
        l = wrap_td(keras.layers.Activation('relu'))(l)
        l = wrap_td(keras.layers.BatchNormalization())(l)

        l = wrap_td(keras.layers.Conv2D(32, (3, 3), padding="same"))(l)
        l = wrap_td(keras.layers.Activation('relu'))(l)
        l = wrap_td(keras.layers.BatchNormalization())(l)

        l = wrap_td(keras.layers.Conv2D(64, (3, 3), padding="same"))(l)
        l = wrap_td(keras.layers.Activation('relu'))(l)
        l = wrap_td(keras.layers.BatchNormalization())(l)

        l = wrap_td(keras.layers.Flatten())(l)

        l = keras.layers.SimpleRNN(128, activation='relu', return_sequences=True)(l)
        l = keras.layers.SimpleRNN(128, activation='relu', return_sequences=True)(l)
        o = wrap_td(keras.layers.Dense(1, dtype=tf.float32))(l)

        return keras.models.Model(inputs=i, outputs=o)

    @staticmethod
    def create_discriminator() -> T_model:
        i = keras.Input(shape=(4, 1), dtype=tf.float32)
        l = keras.layers.SimpleRNN(128, activation='relu')(i)

        l = keras.layers.Dense(128, dtype=tf.float32)(l)
        l = keras.layers.Activation('relu')(l)

        l = keras.layers.Dense(64, dtype=tf.float32)(l)
        l = keras.layers.Activation('relu')(l)

        l = keras.layers.Dense(16, dtype=tf.float32)(l)
        l = keras.layers.Activation('relu')(l)

        o = keras.layers.Dense(1, dtype=tf.float32)(l)

        return keras.models.Model(inputs=i, outputs=o, name="Discriminator")

    def intrinsic_generator_loss(self, prediction: T_tensor, truth: T_tensor) -> T_tensor:
        return mse(prediction, truth)

    def extrinsic_generator_loss(self, discriminator_output: T_tensor) -> T_tensor:
        return bce(tf.ones_like(discriminator_output), discriminator_output)

    def discriminator_loss(self, real_guess: T_tensor, fake_guess: T_tensor) -> T_tensor:
        real_loss = bce(tf.ones_like(real_guess), real_guess)
        fake_loss = bce(tf.zeros_like(fake_guess), fake_guess)
        return tf.add(real_loss, fake_loss)

class FlatRNNGAN(BaseGAN):
    """
    RNN based GAN formulation, which parses a flattened vector representation of the bitboard.
    Uses a simple RNN for the discriminator
    """
    @staticmethod
    def create_generator() -> T_model:

        wrap_td = lambda x: keras.layers.TimeDistributed(x)

        i = keras.Input(shape=(4, 8, 8, 12), dtype=tf.float32)
        l = wrap_td(keras.layers.Flatten())(i)

        l = wrap_td(keras.layers.Dense(512, dtype=tf.float32))(l)
        l = wrap_td(keras.layers.Activation('relu'))(l)

        l = keras.layers.SimpleRNN(256, activation='relu', return_sequences=True)(l)
        l = keras.layers.SimpleRNN(128, activation='relu', return_sequences=True)(l)

        l = wrap_td(keras.layers.Dense(128, dtype=tf.float32))(l)
        l = wrap_td(keras.layers.Activation('relu'))(l)
        l = wrap_td(keras.layers.Dense(32, dtype=tf.float32))(l)
        l = wrap_td(keras.layers.Activation('relu'))(l)

        o = wrap_td(keras.layers.Dense(1, dtype=tf.float32))(l)

        return keras.models.Model(inputs=i, outputs=o)

    @staticmethod
    def create_discriminator() -> T_model:
        i = keras.Input(shape=(4, 1), dtype=tf.float32)
        l = keras.layers.SimpleRNN(128, activation='relu')(i)

        l = keras.layers.Dense(128, dtype=tf.float32)(l)
        l = keras.layers.Activation('relu')(l)

        l = keras.layers.Dense(64, dtype=tf.float32)(l)
        l = keras.layers.Activation('relu')(l)

        l = keras.layers.Dense(16, dtype=tf.float32)(l)
        l = keras.layers.Activation('relu')(l)

        o = keras.layers.Dense(1, dtype=tf.float32)(l)

        return keras.models.Model(inputs=i, outputs=o, name="Discriminator")

    def intrinsic_generator_loss(self, prediction: T_tensor, truth: T_tensor) -> T_tensor:
        return mse(prediction, truth)

    def extrinsic_generator_loss(self, discriminator_output: T_tensor) -> T_tensor:
        return bce(tf.ones_like(discriminator_output), discriminator_output)

    def discriminator_loss(self, real_guess: T_tensor, fake_guess: T_tensor) -> T_tensor:
        real_loss = bce(tf.ones_like(real_guess), real_guess)
        fake_loss = bce(tf.zeros_like(fake_guess), fake_guess)
        return tf.add(real_loss, fake_loss)
