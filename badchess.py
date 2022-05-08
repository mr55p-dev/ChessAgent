from argparse import ArgumentParser
from math import inf
from pathlib import Path

import chess
import matplotlib.pyplot as plt
import tensorflow as tf

from BadChess.environment import Config, search
from BadChess.generator import create_tfdata_set
from BadChess.model import RNNGAN
from BadChess.stockfish import Stockfish

"""
Glue module with argparse utils for:
- Training a model and saving to a .tflite file
- Loading and playing against a model
"""


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
    print(f"Written {path}")

def run_train(args):
    """Run model training"""
    gen_ds = create_tfdata_set(n_items=args.num_train, batch_size=args.batch, chunk_size=args.chunk_size, use_bitboard=True)
    dis_ds = create_tfdata_set(n_items=args.num_train, batch_size=args.batch, chunk_size=args.chunk_size, use_bitboard=False)

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
    path = args.output or "./models/generator_test_model.tflite"
    convert_and_save_model(model, Path(path))

def load_model(model_path: Path):
    """Load a tflite model and do all the associated initializations"""
    # Setup the interpreter
    interpreter = tf.lite.Interpreter(str(model_path))
    interpreter.allocate_tensors()
    input_tensor_idx = interpreter.get_input_details()[0]["index"]
    output_tensor_idx = interpreter.get_output_details()[0]["index"]
    return interpreter, (input_tensor_idx, output_tensor_idx)

def play_vs_bot(args):
    """Play at the command line against the bot"""
    interpreter, (inp, out) = load_model(args.model)
    Config.set_chunksize(args.chunk_size)
    Config.set_interpreter(interpreter)
    Config.set_input(inp)
    Config.set_output(out)

    # Setup some stuff, use the stockfish context manager
    board = chess.Board(chess.STARTING_BOARD_FEN)

    while not board.is_game_over():
        print(f"Ply {board.ply()} - {'white' if board.turn else 'black'} to move")
        if board.turn == chess.WHITE:
            bestMove, withEval = search(board, args.engine_depth, True, -inf, inf, ())
            print(f"RNN move: {bestMove} (evaluated at {withEval}) (searched {Config.num} positions).")
            Config.reset_score()

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

def run_game_vs_stockfish(args) -> None:
    """Play a game against the model specified in `args.model_path`"""
    interpreter, (inp, out) = load_model(args.model)
    Config.set_chunksize(args.chunk_size)
    Config.set_interpreter(interpreter)
    Config.set_input(inp)
    Config.set_output(out)
    printf = print if args.verbose else lambda *args, **kwargs: None

    # Setup some stuff, use the stockfish context manager
    board = chess.Board(args.start)
    with Stockfish(
        args.stockfish_skill,
        movetime=args.stockfish_max_time,
        max_depth=args.stockfish_max_depth
        ) as stockfish:
        while not board.is_game_over():
            printf(f"Ply {board.ply()} - {'white' if board.turn else 'black'} to move")
            if board.turn == chess.WHITE:
                bestMove, withEval = search(board, args.engine_depth, True, -inf, inf, ())
                printf(f"RNN move: {bestMove} (evaluated at {withEval}) (searched {Config.num} positions).")
                Config.reset_score()

                board.push(bestMove)
                printf(board)
            else:
                # Set the board state for the engine and get the move
                stockfish.set_state([i.uci() for i in board.move_stack])
                move = stockfish.get_move()

                # Push the move to the board
                bestMove = chess.Move.from_uci(move)
                board.push(bestMove)
                printf(f"Stockfish move: {bestMove}.")
                printf(board)

    # Check the board outcome
    out = board.outcome()
    return out.winner, out.termination, board.ply()

def run_game_vs_self(args) -> None:
    """Play a game against the twp models specified in `args.white_model` and `args.black_model`"""
    w_interpreter, (w_inp, w_out) = load_model(args.white_model)
    b_interpreter, (b_inp, b_out) = load_model(args.black_model)
    Config.set_chunksize(args.chunk_size)
    print = lambda x: print(x) if args.verbose else lambda x: None

    # Setup some stuff, use the stockfish context manager
    board = chess.Board(args.start)
    while not board.is_game_over():
        print(f"Ply {board.ply()} - {'white' if board.turn else 'black'} to move")
        if board.turn == chess.WHITE:
            # Need to change the model that the search engine uses for each step...
            Config.set_interpreter(w_interpreter)
            Config.set_input(w_inp)
            Config.set_output(w_out)

        else:
            Config.set_interpreter(b_interpreter)
            Config.set_input(b_inp)
            Config.set_output(b_out)

        # Evaluate and make the move
        bestMove, withEval = search(board, args.engine_depth, True, -inf, inf, ())
        print(f"RNN move: {bestMove} (evaluated at {withEval}) (searched {Config.num} positions).")
        Config.reset_score()

        board.push(bestMove)
        print(board)

    # Check the board outcome
    out = board.outcome()
    return out.winner, out.termination, board.ply()

# Argparse stuff
parser = ArgumentParser("BadChess")
subparsers = parser.add_subparsers()

# Train a model
training = subparsers.add_parser("train")
training.add_argument("-e", "--epochs", help="Number of epochs to train for.", type=int, default=1)
training.add_argument("-b", "--batch", help="Batch size.", type=int, default=128)
training.add_argument("-n", "--num_train", help="Number of examples to train on.", type=int, default=10_000)
training.add_argument("-o", "--output", help="Output path to write a .tflite file to.", type=str, default="generator_model.tflite")
training.add_argument("-c", "--chunk_size", help="Number of elements to use in prediction learning", type=int, default=3)
training.add_argument("--graphs", help="Use graphs?", default=False, action="store_true", dest="graph")
training.set_defaults(func=run_train)

# Play as yourself against a model
vs = subparsers.add_parser("vs")
vs.add_argument("model", help="Path to the model file")
vs.add_argument("-d", "--engine_depth", help="Search depth for the engine moves.", type=int, default=4)
vs.add_argument("-c", "--chunk_size", help="Number of elements to use in prediction learning", type=int, default=3)
vs.set_defaults(func=play_vs_bot)

# Put two models against one andother
game_self = subparsers.add_parser("playself")
game_self.add_argument("white_model", help="Path to the white model file")
game_self.add_argument("black_model", help="Path to the black model file")
game_self.add_argument("-d", "--engine_depth", help="Search depth for the engine moves.", type=int, default=4)
game_self.add_argument("-s", "--start", help="Starting position", type=str, default=chess.STARTING_FEN)
game_self.add_argument("--verbose", help="How many messages to print", default=False, action="store_true")
game_self.add_argument("-c", "--chunk_size", help="Number of elements to use in prediction learning", type=int, default=3)
game_self.set_defaults(func=run_game_vs_self)

# Put a model against stockfish
game = subparsers.add_parser("play")
game.add_argument("model", help="Path to the model file")
game.add_argument("-d", "--engine_depth", help="Search depth for the engine moves.", type=int, default=4)
game.add_argument("-s", "--start", help="Starting position", type=str, default=chess.STARTING_FEN)
game.add_argument("-c", "--chunk_size", help="Number of elements to use in prediction learning", type=int, default=3)
game.add_argument("--stockfish_skill", help="Stockfish skill level prop (1-20)", type=int, default=20)
game.add_argument("--stockfish_max_depth", help="Stockfish max depth", type=int, default=10)
game.add_argument("--stockfish_max_time", help="Stockfish maximum thinking time (in ms)", type=int, default=2000)
game.add_argument("--verbose", help="How many messages to print", default=False, action="store_true")
game.set_defaults(func=run_game_vs_stockfish)

# If we are calling this file directly, parse the args called
if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)