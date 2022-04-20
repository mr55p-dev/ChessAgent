
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import subprocess
from subprocess import PIPE, DEVNULL
import re

"""This entire file should be in C++ ideally"""

def process_match(match: re.Match) -> List:
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


def process_game(game: str):
    _, body, _ = game.split("\n\n")
    body = body.replace("\n", "")
    """
    Turn - Ply (. | ...) - Algebraic move - Clock - Evaluation - FEN string
    """
    matches = re.finditer(r"(\d+)(\.+) ?(.*?) ?({.*?}) ?({.*?}) ?({.*?}) ?", body)
    return sorted((process_match(i) for i in matches), key=lambda x: x[0])

def data_generator(file: Path = Path("data/lichess_db_standard_rated_2018-04.pgn.bz2")):
    # Start reading from the bz2 archive
    reader = subprocess.Popen(
        ["bzcat", str(file)], # Opens and sequentially reads the contents of a bz2 archive
        stdout=PIPE,  # Allow feeding directly into the next process
        bufsize=10240 # 10K
    )
    # Pipe the buffered output into the modified pgn-extract binary
    processer = subprocess.Popen(
        ["vendor/pgn-extract/pgn-extract", "-D", "-W", "-7", "--evaluation", "--fencomments", "-ttags"],
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

for i, _ in enumerate(data_generator()):
    print(i, end="\r")