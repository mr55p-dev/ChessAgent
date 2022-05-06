import chess
from BadChess.environment import search, Searched
from math import inf

# Setup some stuff
startfen = input("Starting position: ")
board = chess.Board(startfen if startfen else chess.STARTING_FEN)
while not (endstate := board.is_game_over()):
    print(f"Ply {board.ply()} - {'white' if board.turn else 'black'} to move")
    if board.turn == chess.WHITE:
        bestIndex, withEvaluation = search(board, 3, True, -inf, inf)
        bestMove = list(board.legal_moves)[bestIndex]

        print(f"Automated move: {bestMove} (searched {Searched.num} positions).")
        Searched.reset()

        board.push(bestMove)
        print(board)
    else:
        move = input("Make a move: ")
        try:
            board.push_san(move)
            print(board)
        except ValueError:
            print("Bad move")
            continue

print(board.outcome())