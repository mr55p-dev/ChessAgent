from math import inf
from pathlib import Path
import chess
import tensorflow as tf

from BadChess.generator import bitboard_from_fen


class Config:
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
    if depth < Config.chunksize:
        bitboard = bitboard_from_fen(board.fen())
        bitboard_stack = (*bitboard_stack, bitboard)

    # Check if we are at the bottom of the stack
    if depth == 0:
        # Add a new increment to the number of models searched
        Config.num += 1

        if len(bitboard_stack) < Config.chunksize:
            # If the stack doesnt have enough cache, repeat the current element a couple times
            full_arr_of_current_board = tuple([bitboard_stack[-1]] * Config.chunksize)
            bitboard_stack = (*bitboard_stack, *full_arr_of_current_board)
            bitboard_stack = bitboard_stack[:Config.chunksize]

        # Create a "time series" tensor of inputs of the previous states
        bitboards = tf.stack(bitboard_stack, axis=0)

        # Infer the evaluation, and then take the topmost evaluation
        ev = Config.infer(bitboards)
        ev = tf.squeeze(ev)
        ev = float(ev[-1])

        return None, ev

    # Check if there are more moves to recurse down
    elif board.is_game_over():
        # Get the outcome
        winner = board.outcome().winner

        # Setup the best possible score depending on if we are minimizing or maximizing
        best = inf if max_or_min else -inf

        # Case of a draw
        if winner is None:
            return None, 0
        # Case the current player has won
        elif winner == max_or_min:
            return None, best
        #Case the current player has lost
        else:
            return None, -best

    # Alpha-beta pruning algorithm
    bestMove = None
    if max_or_min:
        maxEval = -inf
        for move in board.legal_moves:
            # Push the hypothetical move to the board, recurse on this new board and then pop it back off the stack, retaining the best evaluation
            board.push(move)
            mv, newEval = search(board, depth - 1, not max_or_min, alpha, beta, bitboard_stack)
            board.pop()

            # If our new evaluation is better then update the best move and evaluation
            if newEval > maxEval:
                maxEval = newEval
                bestMove = move

            # Update our alpha
            alpha = max(alpha, newEval)

            # If beta <= alpha we can prune the rest of the branches of the search at this level
            if beta <= alpha:
                break
        if not bestMove:
            raise ValueError("Problem...")

        return bestMove, maxEval

    else:
        minEval = inf

        # Check if there are legal moves to scan
        if not (legal_moves := list(board.legal_moves)):
            # Winner is white so term == True
            if (term := board.outcome().winner):
                minEval = -inf # Awful hell lets not do that
            elif term is None:
                minEval = 0 # A draws a draw
            else:
                minEval = inf # The best score

        for move in board.legal_moves:
            # Push the hypothetical move to the board, recurse on this new board and then pop it back off the stack, retaining the best evaluation
            board.push(move)
            _, newEval = search(board, depth - 1, not max_or_min, alpha, beta, bitboard_stack)
            board.pop()
            
            # If our new evaluation is better then update the best move and evaluation
            if newEval < minEval:
                minEval = newEval
                bestMove = move

            # Update our beta
            beta = min(beta, newEval)

            # If beta <= alpha we can prune the rest of the branches of the search at this level
            if beta <= alpha:
                break
        return bestMove, minEval
