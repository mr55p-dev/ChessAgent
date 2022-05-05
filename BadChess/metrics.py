import tensorflow as tf

class Loss(tf.keras.metrics.Metric):
    def __init__(self, name="loss", **kwargs) -> None:
        """Define a customm Loss metric as tensorflow does not support one out of the box, in favour of model history.
        Implements the same methods as `tf.keras.metrics.Metric`"""
        super().__init__(name=name, **kwargs)
        self.loss = self.add_weight(name="loss_node", initializer='zeros')
        self.counter = self.add_weight(name="n_steps", initializer='zeros')

    def update_state(self, new_loss):
        self.loss.assign_add(tf.cast(new_loss, self.dtype))
        self.counter.assign_add(tf.cast(1, self.dtype))

    def result(self):
        return tf.divide(self.loss, self.counter)