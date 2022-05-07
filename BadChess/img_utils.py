import collections
import chess
import chess.pgn
from chess_gif.gif_maker import Gifmaker
import os

# From https://chess-gif.readthedocs.io/en/latest/chess_gif.html#submodules
def make_gif(game, outfile):
    with open('tmpfile', 'w') as f:
        f.write(str(game))
    obj = Gifmaker()
    obj.make_gif_from_pgn_file('tmpfile', gif_file_path=outfile)
    os.remove('tmpfile')

# From https://github.com/niklasf/python-chess/issues/63
def board_to_game(board):
    game = chess.pgn.Game()

    # Undo all moves.
    switchyard = collections.deque()
    while board.move_stack:
        switchyard.append(board.pop())

    game.setup(board)
    node = game

    # Replay all moves.
    while switchyard:
        move = switchyard.pop()
        node = node.add_variation(move)
        board.push(move)

    game.headers["Result"] = board.result()
    return game

def gif_from_board(board, outfile='chess.gif'):
    g = board_to_game(board)
    make_gif(g, outfile)
    print(f"Written gif to {outfile}")