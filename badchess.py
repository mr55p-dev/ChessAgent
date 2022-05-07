import argparse
from argparse import ArgumentParser
import itertools
from math import inf
from pathlib import Path
import pprint

import chess
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from BadChess.environment import Config, search
from BadChess.generator import create_tfdata_set
from BadChess.model import RNNGAN
from BadChess.stockfish import Stockfish


def convert_and_save_model(model, path: Path) -> None:
    """Save a model as a tflite file"""
    # Save the model
    saved_model_path = path.parent.joinpath("savedmodel")
    model.save_generator(saved_model_path)

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    tflite_model = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite_model)

def run_train(args):
    """Run model training"""
    gen_ds = create_tfdata_set(n_items=args.num_train, batch_size=args.batch, chunk_size=3, use_bitboard=True)
    dis_ds = create_tfdata_set(n_items=args.num_train, batch_size=args.batch, chunk_size=3, use_bitboard=False)

    model = RNNGAN()
    logs = model.train(
        args.epochs,
        gen_ds,
        dis_ds
    )
    if args.graph:
        # Save some graphs
        for key in logs:
            x_axis, y_axis = zip(*enumerate(logs[key]))
            plt.plot(x_axis, y_axis)
            plt.title(key)
            plt.savefig(f'modelgraphs-{key}.png')
            plt.clf()

    convert_and_save_model(model, Path("./models/generator_test_model.tflite"))

def load_model(model_path: Path):
    """Load a tflite model and do all the associated initializations"""
    # Setup the interpreter
    interpreter = tf.lite.Interpreter(str(model_path))
    interpreter.allocate_tensors()
    input_tensor_idx = interpreter.get_input_details()[0]["index"]
    output_tensor_idx = interpreter.get_output_details()[0]["index"]
    return interpreter, (input_tensor_idx, output_tensor_idx)

def run_game(args) -> None:
    """Play a game against the model specified in `args.model_path`"""
    interpreter, (inp, out) = load_model(args.model)
    Config.set_chunksize(3)
    Config.set_interpreter(interpreter)
    Config.set_input(inp)
    Config.set_output(out)
    print = lambda x: print(x) if args.verbose else lambda x: None

    # Setup some stuff, use the stockfish context manager
    board = chess.Board(args.start)
    with Stockfish(
        args.stockfish_skill,
        movetime=args.stockfish_max_time,
        max_depth=args.stockfish_max_depth
        ) as stockfish:
        while not board.is_game_over():
            print(f"Ply {board.ply()} - {'white' if board.turn else 'black'} to move")
            if board.turn == chess.WHITE:
                bestMove, withEval = search(board, args.engine_depth, True, -inf, inf, ())
                print(f"RNN move: {bestMove} (evaluated at {withEval}) (searched {Config.num} positions).")
                Config.reset_score()

                board.push(bestMove)
                print(board)
            else:
                # Set the board state for the engine and get the move
                stockfish.set_state([i.uci() for i in board.move_stack])
                move = stockfish.get_move()

                # Push the move to the board
                bestMove = chess.Move.from_uci(move)
                board.push(bestMove)
                print(f"Stockfish move: {bestMove}.")
                print(board)

    # Check the board outcome
    out = board.outcome()
    return out.winner, out.termination, board.ply()

# Argparse stuff
parser = ArgumentParser("BadChess")
subparsers = parser.add_subparsers()
training = subparsers.add_parser("train")
training.add_argument("-e", "--epochs", help="Number of epochs to train for.", type=int, default=1)
training.add_argument("-b", "--batch", help="Batch size.", type=int, default=128)
training.add_argument("-n", "--num_train", help="Number of examples to train on.", type=int, default=10_000)
training.add_argument("-o", "--output", help="Output path to write a .tflite file to.", type=str, default="generator_model.tflite")
training.add_argument("--graphs", help="Use graphs?", default=False, action="store_true", dest="graph")
training.set_defaults(func=run_train)

game = subparsers.add_parser("play")
game.add_argument("model", help="Path to the model file")
game.add_argument("-d", "--engine_depth", help="Search depth for the engine moves.", type=int, default=4)
game.add_argument("-s", "--start", help="Starting position", type=str, default=chess.STARTING_FEN)
game.add_argument("--player", help="Are you playing or is stockfish?", action="store_true", default=False)
game.add_argument("--stockfish_skill", help="Stockfish skill level prop (1-20)", type=int, default=20)
game.add_argument("--stockfish_max_depth", help="Stockfish max depth", type=int, default=10)
game.add_argument("--stockfish_max_time", help="Stockfish maximum thinking time (in ms)", type=int, default=2000)
game.add_argument("--verbose", help="How many messages to print", default=False, action="store_true")
game.set_defaults(func=run_game)

# If we are calling this file directly, parse the args called
if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)