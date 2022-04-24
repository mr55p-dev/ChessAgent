
import re
import subprocess
from pathlib import Path
from subprocess import DEVNULL, PIPE
from typing import Tuple

import numpy as np
import tensorflow as tf

"""This entire file should be in C++ ideally"""

"""
PLEASE DEAR GOD DONT FORGET TO OFFSET THE LABELS LOL
"""

T_match_row = Tuple[int, str, float]

def process_match(match: re.Match) -> T_match_row:
    turn, player, move, clock_raw, eval_raw, fen_raw = match.groups()
    # Calculate the ply
    turn = int(turn)
    player = 0 if player == "." else 1
    ply = player + 2 * (turn - player)

    # Parse the time
    # time = re.match(r"\d+:\d+:\d+", clock_raw)

    # Parse the evaluation
    evaluation = float(re.search(r"-?\d+\.\d+", eval_raw).group())

    # Store the fen
    fen = fen_raw.replace("{", "").replace("}", "").strip()

    # Return the all important fields
    return (ply, fen, evaluation)

def process_game(game: str) -> T_match_row:
    _, body, _ = game.split("\n\n")
    body = body.replace("\n", "")
    """
    Turn - Ply (. | ...) - Algebraic move - Clock - Evaluation - FEN string
    """
    matches = re.finditer(r"(\d+)(\.+) ?(.*?) ?({.*?}) ?({.*?}) ?({.*?}) ?", body)
    return sorted((process_match(i) for i in matches), key=lambda x: x[0])

def bitboard_from_fen(fen_string: str) -> tf.Tensor:
    """
    Function to decode a FEN string in to a (8, 8, 12) numpy bitboard array
    Bitboards are set up as following:

                axis 1
             _______________
            |               |
            |               |
    axis 0  |               |
            |               |
            |_______________|

    axis 2 contains bitboard slices for each piece type in the ordering:
    WHITE:
        - K
        - Q
        - R
        - B
        - N
        - P
    BLACK:
        - K
        - Q
        - R
        - B
        - N
        - P
    """
    fields = fen_string.split(' ')
    # Note some fen strings are supplied without castling rights
    if len(fields) != 6:
        return False

    """
    NEED to handle the case where the player to move is not separated from the fen string...
    """

    placement = fields[0]
    ranks = placement.split("/")
    # The ranks are given in reverse order, so index 0
    # of this corresponds to i=7, or the 8th rank.
    pieces = dict(zip(('K', 'Q', 'R', 'B', 'N', 'P'), range(6)))

    bitboard = np.zeros((8, 8, 12), dtype=np.short)

    for rank, squares in enumerate(ranks):
        i = 7 - rank
        # Split the rank into its characters,
        encoding = list(squares)
        j = 0
        for char in encoding:
            if char.isdigit():
                j += int(char)
                continue
            elif char in pieces: # White pieces in upper case
                bitboard[i, j, pieces[char]] = 1
            else: # Black pieces in lower case
                bitboard[i, j, 6 + pieces[char.upper()]] = 1
            if j >= 8:
                break
            j += 1

    # Decode the next turn
    # to_move = int(fields[1] == 'w')

    # Castling informataion
    # castle = fields[2]

    # Valid enpassant moves
    # en_passant = fields[3]

    # Halfmove clock 2 x moves since last pawn move or capture
    # half_moves = int(fields[4])

    # Ply
    # ply = to_move + 2 * (int(fields[5]) - to_move)
    return tf.convert_to_tensor(bitboard, dtype=tf.bool)

# Batch dimension (handled by tf.data)
# Chunk dimension (handled by me / tfdata (window size!))
# Board shape (8, 8, 12) (all me)

def data_generator(
    # n_items: int,
    # cached: bool = True,
    # buf_size: int = 1,
    # shuffled: bool = True,
    bracket: int,
    file: Path = Path("data/lichess_db_standard_rated_2018-04.pgn.bz2"),
    ):
    """
    Generator function which yields whole games from the specified bz2 archive

    args:
        int (1|2) bracket - using 1k (1) or 2k (2) elo games
        Path|str file - path to the bz2 archive to read from
    returns:
        generator[(ply: int, fen: str, evaluation: float)]
    """
    tagfile = "tags_900_1100" if bracket == 1 else "tags_1900_2100"
    # Start reading from the bz2 archive
    reader = subprocess.Popen(
        ["bzcat", str(file)], # Opens and sequentially reads the contents of a bz2 archive
        stdout=PIPE,  # Allow feeding directly into the next process
        bufsize=10240 # 10K
    )
    # Pipe the buffered output into the modified pgn-extract binary
    processer = subprocess.Popen(
        [
            "./vendor/pgn-extract/pgn-extract",
            "-D",
            "-W",
            "-7",
            "--evaluation",
            "--addfencastling",
            "--fencomments",
            f"-t{tagfile}"
        ],
        stdin=reader.stdout,
        bufsize=1024, # 1K
        universal_newlines=True,
        stdout=PIPE,
        stderr=DEVNULL
    )
    linebuffer = []
    for line in processer.stdout:
        # Check if the line to append is the first tag ([Event])
        if line[:6] == "[Event":
            # Check the linebuffer is not empty (only relevant at the start of execution)
            b = "".join(linebuffer)
            if not b: continue

            # Flush the linebuffer and process the game
            linebuffer.clear()
            yield process_game(b)

        linebuffer.append(line)

def move_stream():
    BUF_SIZE = 10
    # SHUFFLE = True

    # Get stream of games
    game_stream = data_generator(bracket=1)
    # Create a buffer of size B
    gamebuf = []
    seq_id = 0
    n = 0

    # define shuffle here
    # shuffle = shuffle() if SHUFFLE else lambda x: x

    while True:
        # Fill the buffer from the stream
        gamebuf = [next(game_stream) for _ in range(BUF_SIZE)]

        # Shuffle the buffer (?)
        # shuffle(game_buffer)

        for game in gamebuf:
            for _, fen, ev in game:
                # Decode FEN
                bitboard = bitboard_from_fen(fen)
                evaluation = tf.convert_to_tensor(ev, dtype=tf.float64)
                seq_id = tf.convert_to_tensor(seq_id, dtype=tf.uint32)
                yield bitboard, evaluation, seq_id
            # Upadte the sequence id after each game
            seq_id += 1
