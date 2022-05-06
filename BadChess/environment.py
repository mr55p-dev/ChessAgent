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

def load_model(path: Path):
    return tf.keras.models.load_model(path)

model = load_model(Path("./generator_test"))
class Searched:
    num = 0
    @classmethod
    def reset(cls):
        cls.num = 0

def search(board: chess.Board, depth: int, max_or_min: bool, alpha: int, beta: int):
    """Basic implementation of alpha-beta pruning to search and find the best move in a given position"""
    if depth  == 0:
        Searched.num += 1
        # 
        fen = board.fen()
        bitboard = bitboard_from_fen(fen)
        bitboard = tf.expand_dims(bitboard, 0)

        ev = model(bitboard)
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
