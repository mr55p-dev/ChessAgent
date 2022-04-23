from BadChess.generator import bitboard_from_fen, data_generator, data_fetcher, move_stream
import numpy as np
import tensorflow as tf

def typecheck(move_row):
    ply, fen, eva = move_row
    assert type(ply) == int
    assert type(fen) == str
    assert type(eva) == float

def test_raw_enerator():
    gen = data_generator()
    for _ in range(10):
        moves = next(gen)
        [typecheck(i) for i in moves]

def test_fetcher():
    fetch = data_fetcher(5)
    white, black = next(fetch)
    assert len(white) == 3
    assert len(black) == 2

    ply = lambda x: x[0]
    assert ply(white[1]) == ply(white[0]) + 2

    for i in white:
        typecheck(i)
    for j in black:
        typecheck(j)


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

def test_generator():
    n_items = 5
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

def test_fen_whitespace():
    """
    Check to make sure that pgn-extract has been compiled
    with leading/trailing line whitespace enabled, otherwise
    separating the fen strings is HARD as you run into situations
    where 'w' or 'b' are appended to the end of a string without
    the proper whitespace, making decoding the string reliably
    impossible
    """
    gen = data_generator()
    for _ in range(1000):
        moves = next(gen)
        for _, fen, _ in moves:
            sp = fen.split(' ')
            if sp[0][-1] in {'w'}:
                raise ValueError("Still wrong")

