from BadChess.model import create_discriminator, create_generator
import tensorflow as tf

def test_generator():
    G = create_generator()
    assert isinstance(G, tf.keras.models.Model)

def test_discriminator():
    D = create_discriminator()
    assert isinstance(D, tf.keras.models.Model)

    