from typing import List, Tuple
import chess
from random import randint

from BadChess.generator import bitboard_from_fen

"""
Basically the algorithm goes as follows:

For a position
- Get all the legal moves in a position
- Calculate evaluations after each move
- Follow the best choice
- Recurse to depth d
"""

model = lambda x: randint(0, 10)
class Searched:
    num = 0

def search(board: chess.Board, depth: int, max_or_min: bool, alpha: int, beta: int):
    if depth  == 0:
        Searched.num += 1

        fen = board.fen()
        bitboard = bitboard_from_fen(fen)
        return None, model(bitboard)

    bestIdx = None
    if max_or_min:
        maxEval = -100_000
        for idx, move in enumerate(board.legal_moves):
            board.push(move)
            _, newEval = search(board, depth - 1, not max_or_min, alpha, beta)
            board.pop()

            if newEval > maxEval:
                maxEval = newEval
                bestIdx = idx
            alpha = max(alpha, newEval)
            if beta <= alpha:
                break
        return bestIdx, maxEval

    else:
        minEval = 100_000
        for idx, move in enumerate(board.legal_moves):
            board.push(move)
            _, newEval = search(board, depth - 1, not max_or_min, alpha, beta)
            board.pop()

            if newEval < minEval :
                minEval = newEval
                bestIdx = idx

            beta = min(beta, newEval)
            if beta <= alpha:
                break
        return bestIdx, minEval


    # Setup the iterative recursion
    bestIdx = None
    for idx, move in enumerate(board.legal_moves):
        board.push(move)
        _, newEval = search(board, depth - 1, not max_or_min, alpha, beta)
        if op(newEval):
            maxEval = newEval
            bestIdx = idx
        alpha = alphaOp(alpha, newEval)
        if beta <= alpha:
            break

        board.pop()
    return bestIdx, maxEval




