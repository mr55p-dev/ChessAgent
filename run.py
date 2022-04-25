from typing import List
from BadChess.modelclass import ConcreteGAN
from BadChess.generator import create_tfdata_set


G_train = create_tfdata_set(chunk_size=1)
D_train = create_tfdata_set(chunk_size=1)

model = ConcreteGAN()
model.train(
    20,
    G_train,
    D_train
)
