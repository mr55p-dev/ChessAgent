import re
import subprocess
from pathlib import Path
from subprocess import DEVNULL, PIPE
from typing import Tuple

import numpy as np
import tensorflow as tf


# Define some common types
T_match_row = Tuple[int, str, float]
T_tfdata_output = (
    tf.TensorSpec(shape=(1,), dtype=tf.float32, name="evaluation"),
    tf.TensorSpec(shape=(), dtype=tf.uint32, name="sequence_id")
)
T_tfdata_output_bitboard = (
    tf.TensorSpec(shape=(8, 8, 12), dtype=tf.float32, name="bitboard"),
    *T_tfdata_output
)


"""Create the stream of games at the module level, to ensure that data isnt shared or that the file isn't opened lots of times"""
# Select the tags (PGN filtering rules) to use
tagfiles = ["tags_900_1100", "tags_1900_2100"]
tagfile = tagfiles[1]

# Start reading from the bz2 archive of games
reader = subprocess.Popen(
    ["bzcat", str(Path("data/lichess_db_standard_rated_2018-04.pgn.bz2"))], # Reads the contents of a .bz2 archive buffered
    stdout=PIPE,  # Pipe directly into the next proc
    bufsize=1_000_000 # 1Mb
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
    bufsize=100_000, # 100Kb
    universal_newlines=True,
    stdout=PIPE,
    stderr=DEVNULL
)


def process_match(match: re.Match) -> T_match_row:
    """Parse a matched ply into a fen string, evaluation and ply number"""
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
    """Match and process all the ply in a game into fields"""
    items = game.split("\n\n")
    if len(items) < 2:
        raise ValueError(r"\n\n delimeter not present")
    body = items[1]
    body = body.replace("\n", "")
    """
    Turn - Ply (. | ...) - Algebraic move - Clock - Evaluation - FEN string
    """
    matches = re.finditer(r"(\d+)(\.+) ?(.*?) ?({.*?}) ?({.*?}) ?({.*?}) ?", body)
    return sorted((process_match(i) for i in matches), key=lambda x: x[0])

def bitboard_from_fen(fen_string: str) -> tf.Tensor:
    """
    From https://github.com/mpags-python/coursework2021-sub3-mr55p-dev/blob/main/Chess/helpers.py
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
    WHITE : K Q R B N P
    BLACK : K Q R B N P
    """
    fields = fen_string.split(' ')
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
    return tf.convert_to_tensor(bitboard, dtype=tf.float32, name="bitboard")

def game_generator():
    """
    Generator function which yields whole games from the specified bz2 archive

    args:
        int (1|2) bracket - using 1k (1) or 2k (2) elo games
        Path|str file - path to the bz2 archive to read from
    returns:
        generator[(ply: int, fen: str, evaluation: float)]
    """
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

# I choose to implement this function 2 times, since otherwise inside the while
# loop i need to have a condition to check if we are using the bitboard, which will
# need to be evaluated on each iteration. Since i am making optimizations at the time
# of writing this change I know that the functions here dont need to change, so just
# statically implementing it twice and setting a reference to the right one further down
# feels like it should be a decent improvement in this context; forgive me DRY...
def move_stream():
    """Generates a coherent stream of positions, with associated stream id and evaluation"""
    seq_id = 0

    while True:
        # # Fill the buffer from the stream
        # gamebuf = [next(game_stream) for _ in range(BUF_SIZE)]

        # # Shuffle the buffer (?)
        # # shuffle(game_buffer)

        for game in game_generator():
            for ply, fen, ev in game:
                # Decode FEN
                bitboard = bitboard_from_fen(fen)
                evaluation = tf.reshape(
                    tf.convert_to_tensor(ev, dtype=tf.float32, name="evaluation"), (1,)
                )
                seq_id = tf.convert_to_tensor(seq_id, dtype=tf.uint32, name="sequence_id")
                yield bitboard, evaluation, seq_id
            # Upadte the sequence id after each game
            seq_id += 1

def move_stream_no_bitboard():
    seq_id = 0
    while True:
        for game in game_generator():
            for _, _, ev in game:
                # Decode FEN
                evaluation = tf.reshape(
                    tf.convert_to_tensor(ev, dtype=tf.float32, name="evaluation"), (1,)
                )
                seq_id = tf.convert_to_tensor(seq_id, dtype=tf.uint32, name="sequence_id")
                yield evaluation, seq_id
            # Upadte the sequence id after each game
            seq_id += 1


def create_tfdata_set(
    n_items: int = 512,
    shuffle_bufsize: int = 0,
    batch_size: int = 4,
    chunk_size: int = 3,
    use_bitboard: bool = True
    ):
    """
    Returns a fully formed tf.data.Dataset instance
    code based on https://stackoverflow.com/questions/55109817/batch-sequential-data-with-tf-data
    """
    if use_bitboard:
        window_to_nested_ds = lambda b, e, s: tf.data.Dataset.zip((b, e, s)).batch(chunk_size, drop_remainder=True)
        drop_game_crossover = lambda b, e, s: tf.equal(tf.size(tf.unique(s)[0]), 1)
        clip_norm_evaluation = lambda b, e, s: (b, tf.divide(tf.clip_by_value(e, -40, 40), tf.constant(40, dtype=tf.float32)), s)
        remove_seqid = lambda b, e, s: (b, e)
        ds = tf.data.Dataset.from_generator(
            move_stream,
            output_signature=T_tfdata_output_bitboard
        )
    else:
        window_to_nested_ds = lambda e, s: tf.data.Dataset.zip((e, s)).batch(chunk_size, drop_remainder=True)
        drop_game_crossover = lambda e, s: tf.equal(tf.size(tf.unique(s)[0]), 1)
        clip_norm_evaluation = lambda e, s: (tf.divide(tf.clip_by_value(e, -40, 40), tf.constant(40, dtype=tf.float32)), s)
        remove_seqid = lambda e, s: e
        ds = tf.data.Dataset.from_generator(
            move_stream_no_bitboard,
            output_signature=T_tfdata_output
        )
    ds = ds.take(n_items)
    ds = ds.map(clip_norm_evaluation)
    ds = ds.window(chunk_size, 1, stride=3, drop_remainder=True)
    ds = ds.flat_map(window_to_nested_ds)
    ds = ds.filter(drop_game_crossover)
    ds = ds.map(remove_seqid)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.cache()
    return ds.prefetch(tf.data.AUTOTUNE)