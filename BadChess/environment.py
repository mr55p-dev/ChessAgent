from math import inf
from pathlib import Path
import chess
import tensorflow as tf

from BadChess.generator import bitboard_from_fen

"""
Basically the algorithm goes as follows:

For a position
- Get all the legal moves in a position
- Calculate evaluations after each move
- Follow the best choice
- Recurse to depth d
"""
class ModelMeta:
    num = 0
    interpreter = None
    inp = None
    out = None

    @classmethod
    def reset_score(cls):
        cls.num = 0

    @classmethod
    def set_interpreter(cls, interpreter):
        cls.interpreter = interpreter

    @classmethod
    def set_input(cls, input_tensor_index):
        cls.inp = input_tensor_index

    @classmethod
    def set_output(cls, output_tensor_index):
        cls.out = output_tensor_index

    @classmethod
    def infer(cls, tensor):
        cls.interpreter.set_tensor(cls.inp, tf.expand_dims(tf.cast(tensor, tf.float32), 0))
        cls.interpreter.invoke()
        return cls.interpreter.get_tensor(cls.out)

def search(board: chess.Board, depth: int, max_or_min: bool, alpha: int, beta: int):
    """Basic implementation of alpha-beta pruning to search and find the best move in a given position"""
    if depth  == 0:
        ModelMeta.num += 1
        #
        fen = board.fen()
        bitboard = bitboard_from_fen(fen)
        bitboard = tf.expand_dims(bitboard, 0)

        ev = ModelMeta.infer(bitboard)
        ev = tf.squeeze(ev)
        ev = float(ev)

        return None, ev

    bestMove = None
    if max_or_min:
        maxEval = -inf
        for move in board.legal_moves:
            board.push(move)
            _, newEval = search(board, depth - 1, not max_or_min, alpha, beta)
            board.pop()

            if newEval > maxEval:
                maxEval = newEval
                bestMove = move
            alpha = max(alpha, newEval)
            if beta <= alpha:
                break
        return bestMove, maxEval

    else:
        minEval = inf
        for move in board.legal_moves:
            board.push(move)
            _, newEval = search(board, depth - 1, not max_or_min, alpha, beta)
            board.pop()

            if newEval < minEval :
                minEval = newEval
                bestMove = move

            beta = min(beta, newEval)
            if beta <= alpha:
                break
        return bestMove, minEval
