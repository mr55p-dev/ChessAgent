from BadChess.model import ConcreteGAN
import tensorflow as tf


def test_generator():
    G = ConcreteGAN.create_generator()
    assert isinstance(G, tf.keras.models.Model)

def test_discriminator():
    D = ConcreteGAN.create_discriminator()
    assert isinstance(D, tf.keras.models.Model)

