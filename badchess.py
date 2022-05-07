from argparse import ArgumentParser
import argparse
from pathlib import Path
import chess
from math import inf
import tensorflow as tf
import matplotlib.pyplot as plt

from BadChess.generator import create_tfdata_set
from BadChess.model import RNNGAN
from BadChess.environment import search, Config
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

    # Setup some stuff
    board = chess.Board(args.start)
    with Stockfish(1000) as stockfish:
        while not board.is_game_over():
            print(f"Ply {board.ply()} - {'white' if board.turn else 'black'} to move")
            if board.turn == chess.WHITE:
                bestMove, withEval = search(board, args.engine_depth, True, -inf, inf, ())
                print(bestMove)
                print(f"Automated move: {bestMove} (evaluated at {withEval}) (searched {Config.num} positions).")
                Config.reset_score()

                board.push(bestMove)
                print(board)
            elif args.player:
                move = input("Make a move: ")
                try:
                    board.push_san(move)
                    print(board)
                except ValueError:
                    print("Bad move")
                    continue
            else:
                move_list = [i.uci() for i in board.move_stack]
                stockfish.set_state(move_list)
                move = stockfish.get_move()
                board.push(chess.Move.from_uci(move))
                print(board)

    print(board.outcome())

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
game.set_defaults(func=run_game)

args = parser.parse_args()
args.func(args)