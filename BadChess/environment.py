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
    chunksize = None
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
        cls.interpreter.set_tensor(cls.inp, tf.expand_dims(tensor, axis=0))
        cls.interpreter.invoke()
        out = cls.interpreter.get_tensor(cls.out)
        return out

    @classmethod
    def set_chunksize(cls, size):
        cls.chunksize = size

def search(board: chess.Board, depth: int, max_or_min: bool, alpha: int, beta: int, bitboard_stack):
    """Basic implementation of alpha-beta pruning to search and find the best move in a given position"""

    # Create and cache the current bitboard state if we are at a low depth
    if depth < ModelMeta.chunksize:
        bitboard = bitboard_from_fen(board.fen())
        bitboard_stack = (*bitboard_stack, bitboard)

    if depth  == 0:
        # Add a new increment to the number of models searched
        ModelMeta.num += 1

        if len(bitboard_stack) < ModelMeta.chunksize - 1:
            # If the stack doesnt have enough cache, repeat the current element a couple times
            bitboard_stack = bitboard_stack + [bitboard_stack[-1]] * (ModelMeta.chunksize - 1)
            bitboard_stack = bitboard_stack[:ModelMeta.chunksize-1]

        # Create a "time series" tensor of inputs of the previous states
        bitboards = tf.stack(bitboard_stack, axis=0)

        # Infer the evaluation, and then take the topmost evaluation
        ev = ModelMeta.infer(bitboards)
        ev = tf.squeeze(ev)
        ev = float(ev[-1])

        return None, ev

    # Alpha-beta pruning algorithm
    bestMove = None
    if max_or_min:
        maxEval = -inf
        for move in board.legal_moves:
            board.push(move)
            _, newEval = search(board, depth - 1, not max_or_min, alpha, beta, bitboard_stack)
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
            _, newEval = search(board, depth - 1, not max_or_min, alpha, beta, bitboard_stack)
            board.pop()

            if newEval < minEval :
                minEval = newEval
                bestMove = move

            beta = min(beta, newEval)
            if beta <= alpha:
                break
        return bestMove, minEval
