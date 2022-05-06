from typing import List
from BadChess.modelclass import ConcreteGAN, RNNGAN
from BadChess.generator import create_tfdata_set

n_items = 2_000
batch_size = 64
chunk_size = 3

gen_ds = create_tfdata_set(n_items=n_items, batch_size=batch_size, chunk_size=chunk_size, use_bitboard=True)
dis_ds = create_tfdata_set(n_items=n_items, batch_size=batch_size, chunk_size=chunk_size, use_bitboard=False)

model = RNNGAN()
model.train(
    15,
    gen_ds,
    dis_ds
)

model.save_generator("./generator_test")
