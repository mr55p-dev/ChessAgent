from typing import List
from BadChess.model import ConcreteGAN, RNNGAN
from BadChess.generator import create_tfdata_set

n_items = 20_000
batch_size = 64
chunk_size = 3


