from typing import List
from BadChess.modelclass import ConcreteGAN
from BadChess.generator import collect_datasets, create_tfdata_set

n_items = 2_000_000
batch_size = 64
chunk_size = 1
# G_train, D_train = collect_datasets(chunk_size=1, batch_size=128, n_batches=50_000)
gen_ds = create_tfdata_set(n_items=n_items, batch_size=batch_size, chunk_size=chunk_size, use_bitboard=True)
dis_ds = create_tfdata_set(n_items=n_items, batch_size=batch_size, chunk_size=chunk_size, use_bitboard=False)

model = ConcreteGAN()
model.train(
    15,
    gen_ds,
    dis_ds
)
