import chess
from BadChess.environment import search, Searched

# Setup some stuff
board = chess.Board()
max_or_min = True

# Do the thing!
result = search(board, 3, max_or_min, -100_000, 100_000)
print(result)
print(Searched.num)