from argparse import ArgumentParser
import argparse
from pathlib import Path
import chess
from math import inf

from BadChess.generator import create_tfdata_set
from BadChess.modelclass import RNNGAN
from BadChess.environment import search, Meta

def run_train(args):
    gen_ds = create_tfdata_set(n_items=args.num_train, batch_size=args.batch, chunk_size=3, use_bitboard=True)
    dis_ds = create_tfdata_set(n_items=args.num_train, batch_size=args.batch, chunk_size=3, use_bitboard=False)

    model = RNNGAN()
    model.train(
        args.epochs,
        gen_ds,
        dis_ds
    )
    
    convert_and_save_model(model)


def convert_and_save_model(model, path: Path) -> None:
    model.save_generator("./generator_test")

def load_model(model_path: Path):
    ...

def run_game(args) -> None:

    model = load_model(args.model_path)
    Meta.set_model(model)

    # Setup some stuff
    startfen = input("Starting position: ")
    board = chess.Board(startfen if startfen else chess.STARTING_FEN)
    while not board.is_game_over():
        print(f"Ply {board.ply()} - {'white' if board.turn else 'black'} to move")
        if board.turn == chess.WHITE:
            bestMove, withEval = search(board, 2, True, -inf, inf)
            print(bestMove)
            print(f"Automated move: {bestMove} (evaluated at {withEval}) (searched {Meta.num} positions).")
            Meta.reset()

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

training = parser.add_subparsers("train")
training.add_argument("-e", "--epochs", help="Number of epochs to train for.", type=int, default=1)
training.add_argument("-b", "--batch", help="Batch size.", type=int, default=128)
training.add_argument("-n", "--num_train", help="Number of examples to train on.", type=int, default=10_000)
training.add_argument("-o", "--output", help="Output path to write a .tflite file to.", type=str, default="generator_model.tflite")
training.set_defaults(func=run_train)

game = parser.add_subparsers("train")
game.add_args("model", help="Path to the model file", required=True)
game.set_defaults(func=run_game)

args = parser.parse_args()