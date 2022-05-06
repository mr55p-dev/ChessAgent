import chess
from BadChess.environment import search, Searched
from math import inf

# Setup some stuff
board = chess.Board()
while not (endstate := board.is_game_over()):
    if board.turn == chess.WHITE:
        bestIndex, withEvaluation = search(board, 4, True, -inf, inf)
        bestMove = list(board.legal_moves)[bestIndex]
        print(f"Automated move: {bestMove}")

        board.push(bestMove)
        print(board)
    else:
        move = input("Make a move: ")
        try:
            board.push_san(move)
            print(board)
        except ValueError():
            print("Bad move")
            continue

print(board.outcome())