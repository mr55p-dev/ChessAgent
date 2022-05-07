from argparse import ArgumentParser
import argparse
from pathlib import Path
import chess
from math import inf
import tensorflow as tf

from BadChess.generator import create_tfdata_set
from BadChess.modelclass import RNNGAN
from BadChess.environment import search, ModelMeta


def convert_and_save_model(model, path: Path) -> None:
    # Save the model
    saved_model_path = path.parent.joinpath("savedmodel")
    model.save_generator(saved_model_path)

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    tflite_model = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite_model)

def run_train(args):
    gen_ds = create_tfdata_set(n_items=args.num_train, batch_size=args.batch, chunk_size=3, use_bitboard=True)
    dis_ds = create_tfdata_set(n_items=args.num_train, batch_size=args.batch, chunk_size=3, use_bitboard=False)

    model = RNNGAN()
    model.train(
        args.epochs,
        gen_ds,
        dis_ds
    )

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
    ModelMeta.set_chunksize(3)
    ModelMeta.set_interpreter(interpreter)
    ModelMeta.set_input(inp)
    ModelMeta.set_output(out)

    # Setup some stuff
    board = chess.Board(args.start)
    while not board.is_game_over():
        print(f"Ply {board.ply()} - {'white' if board.turn else 'black'} to move")
        if board.turn == chess.WHITE:
            bestMove, withEval = search(board, args.engine_depth, True, -inf, inf, ())
            print(bestMove)
            print(f"Automated move: {bestMove} (evaluated at {withEval}) (searched {ModelMeta.num} positions).")
            ModelMeta.reset_score()

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

# Argparse stuff
parser = ArgumentParser("BadChess")
subparsers = parser.add_subparsers()
training = subparsers.add_parser("train")
training.add_argument("-e", "--epochs", help="Number of epochs to train for.", type=int, default=1)
training.add_argument("-b", "--batch", help="Batch size.", type=int, default=128)
training.add_argument("-n", "--num_train", help="Number of examples to train on.", type=int, default=10_000)
training.add_argument("-o", "--output", help="Output path to write a .tflite file to.", type=str, default="generator_model.tflite")
training.set_defaults(func=run_train)

game = subparsers.add_parser("play")
game.add_argument("model", help="Path to the model file")
game.add_argument("-d", "--engine_depth", help="Search depth for the engine moves.", type=int, default=4)
game.add_argument("-s", "--start", help="Starting position", type=str, default=chess.STARTING_FEN)
game.set_defaults(func=run_game)

args = parser.parse_args()
args.func(args)