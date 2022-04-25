from abc import abstractmethod
from typing import Iterable, List, Tuple
import tensorflow as tf
from tensorflow import keras
from tqdm import trange


T_model = keras.models.Model
T_tensor = tf.Tensor
T_metrics = List[tf.keras.metrics.Metric]
T_opt = keras.optimizers.Optimizer
T_cblist = keras.callbacks.CallbackList


class BaseGAN():
    def __init__(self) -> None:
        self.generator: T_model = self.create_generator()
        self.discriminator: T_model = self.create_discriminator()
        self.metrics: T_metrics
        self.G_opt: T_opt = tf.keras.optimizers.Adam()
        self.D_opt: T_opt = tf.keras.optimizers.Adam()
        self.cblist: T_cblist

    @abstractmethod
    def create_generator(self) -> T_model:
        ...

    @abstractmethod
    def create_discriminator(self) -> T_model:
        ...

    # @abstractmethod
    # def set_metrics(self, metrics: T_metrics) -> None:
    #     ...

    @abstractmethod
    def intrinsic_generator_loss(
        self,
        prediction: T_tensor,
        truth: T_tensor) -> T_tensor:
        ...

    @abstractmethod
    def extrinsic_generator_loss(
        self,
        discriminator_output: T_tensor) -> T_tensor:
        ...

    @abstractmethod
    def discriminator_loss(
        self,
        real_guess: T_tensor,
        fake_guess: T_tensor
        ) -> T_tensor:
        ...

    @tf.function
    def _trainstep(self, batch_G: Tuple[T_tensor, T_tensor], batch_D: T_tensor):
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

        # G_rmse.update_state(G_pred_eval, batch_G_eval)
        # D_accuracy.update_state(tf.clip_by_value(tf.round(D_pred_real), 0, 1), tf.ones_like(D_pred_real))
        # D_accuracy.update_state(D_pred_fake, tf.zeros_like(D_pred_fake))

        # Calculate DL_G, DL_D
        dL_G = G_tape.gradient(loss_G, self.generator.trainable_weights)
        dL_D = D_tape.gradient(loss_D, self.discriminator.trainable_weights)

        # Apply gradient updates to G, D
        self.G_opt.apply_gradients(zip(dL_G, self.generator.trainable_weights))
        self.D_opt.apply_gradients(zip(dL_D, self.discriminator.trainable_weights))

        return loss_G, loss_D

    def train(
        self,
        n_epochs: int,
        ds_train_generator: Iterable,
        ds_train_discriminator,
        ds_val_generator = None,
        ds_val_discriminator = None,
        ):
        # Use the progressbar callback!
        # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CallbackList
        self.N_epochs = 100
        self.batch_size = 20
        self.chunk_size = 1
        self.n_items = 5000

        # Populate with typical keras callbacks
        _callbacks = []

        # Note that we cannot use history callback without specifying the model
        # could be easy to create a quick subclass of history
        callbacks = tf.keras.callbacks.CallbackList(
            _callbacks,
            # add_history=True,
            add_progbar=True,
            # model=self.generator
        )

        logs = {}
        callbacks.on_train_begin(logs=logs)

        for epoch in range(n_epochs):
            callbacks.on_epoch_begin(epoch)
            training_batches = zip(ds_train_generator, ds_train_discriminator)

            for step, (batch_G, (_, batch_D)) in enumerate(training_batches):
                callbacks.on_batch_begin(step)
                callbacks.on_train_batch_begin(step)

                loss_G, loss_D = self._trainstep(batch_G, batch_D)

                callbacks.on_batch_end(step)
                callbacks.on_train_batch_end(step)

            callbacks.on_epoch_end(epoch)

            # Reset the metrics
            # G_rmse.reset_states()
            # D_accuracy.reset_states()
        callbacks.on_train_end()

mse = tf.keras.losses.MeanSquaredError()
bce = tf.keras.losses.BinaryCrossentropy()

class ConcreteGAN(BaseGAN):
    def create_generator(self) -> T_model:
        i = keras.Input(shape=(8, 8, 12), dtype=tf.float32)
        l = keras.layers.Flatten()(i)
        l = keras.layers.Dense(200)(l)
        l = keras.layers.Dense(200)(l)
        o = keras.layers.Dense(1, dtype=tf.float32)(l)
        return keras.models.Model(inputs=i, outputs=o)

    def create_discriminator(self) -> T_model:
        i = keras.Input(shape=(1,), dtype=tf.float32)
        l = keras.layers.Dense(100)(i)
        l = keras.layers.Dense(100)(l)
        o = keras.layers.Dense(1, dtype=tf.float32)(l)
        return keras.models.Model(inputs=i, outputs=o, name="Discriminator")

    def intrinsic_generator_loss(self, prediction: T_tensor, truth: T_tensor) -> T_tensor:
        return mse(prediction, truth)

    def extrinsic_generator_loss(self, discriminator_output: T_tensor) -> T_tensor:
        return bce(tf.zeros_like(discriminator_output), discriminator_output)

    def discriminator_loss(self, real_guess: T_tensor, fake_guess: T_tensor) -> T_tensor:
        real_loss = bce(tf.ones_like(real_guess), real_guess)
        fake_loss = bce(tf.zeros_like(fake_guess), fake_guess)
        return tf.add(real_loss, fake_loss)
