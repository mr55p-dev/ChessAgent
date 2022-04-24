import chunk
from BadChess.generator import bitboard_from_fen, create_tfdata_set, data_generator, move_stream
import numpy as np
import tensorflow as tf

def typecheck(move_row):
    ply, fen, eva = move_row
    assert type(ply) == int
    assert type(fen) == str
    assert type(eva) == float

def test_raw_generator():
    gen = data_generator(1)
    for _ in range(10):
        moves = next(gen)
        [typecheck(i) for i in moves]

def test_raw_fen_whitespace():
    """
    Check to make sure that pgn-extract has been compiled
    with leading/trailing line whitespace enabled, otherwise
    separating the fen strings is HARD as you run into situations
    where 'w' or 'b' are appended to the end of a string without
    the proper whitespace, making decoding the string reliably
    impossible
    """
    gen = data_generator(1)
    for _ in range(1000):
        moves = next(gen)
        for _, fen, _ in moves:
            sp = fen.split(' ')
            if sp[0][-1] in {'w'}:
                raise ValueError("Still wrong")


def test_fen_decoder():
    bitboard_tensor = bitboard_from_fen(r'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    assert bitboard_tensor.shape == (8, 8, 12)
    bitboard = bitboard_tensor.numpy()
    # K Q R B N P
    for i in range(6):
        white = bitboard[:, :, i]
        black = bitboard[:, :, i + 6]
        assert (white == np.flip(black, 0)).all()

    white_king_init = np.zeros((8, 8))
    white_king_init[0, 4] = 1
    assert (bitboard[:, :, 0] == white_king_init).all()

    white_pawns_init = np.zeros((8, 8))
    white_pawns_init[1, :] = np.ones_like(white_pawns_init[1, :])
    assert (bitboard[:, :, 5] == white_pawns_init).all()

def test_move_stream():
    n_items = 200
    gen = move_stream()
    prev_seqid = 0
    count = 0
    for _ in range(n_items):
        board, eval, seqid = next(gen)
        assert board.shape == (8, 8, 12)
        assert seqid >= prev_seqid
        assert eval.dtype == tf.float64
        prev_seqid = seqid
        count += 1
    assert count == n_items
    assert prev_seqid > 0

def test_tfdata():
    batch_size = 100
    chunk_size = 3
    board_shape = (8, 8, 12)

    dataset = create_tfdata_set(
        batch_size=batch_size,
        chunk_size=chunk_size
    )
    for item in iter(dataset.take(10)):
        board, ev = item
        assert board.shape == (batch_size, chunk_size, *board_shape)
        assert ev.shape == (batch_size, chunk_size)

