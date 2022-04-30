from typing import List
from BadChess.modelclass import ConcreteGAN
from BadChess.generator import create_tfdata_set


G_train = create_tfdata_set(chunk_size=1, n_items=int(16e3), batch_size=64)
D_train = create_tfdata_set(chunk_size=1, n_items=int(16e3), batch_size=64, use_bitboard=False, skip=int(1e3))

model = ConcreteGAN()
model.train(
    20,
    G_train,
    D_train
)
