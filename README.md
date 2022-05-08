# Is it possible to augment the behaviour of a static chess agent by learning its evaluation function using a Recurrent Neural Network?

## Project layout

`BadChess` is a python module containing a few different submodules:
- `environment.py` contains the alpha-beta search algorithm and the `Config` class which provides an interface for setting a tflite intrepreter for the search algorithm to use, via classmethods
- `generator.py` contains all the functions to load a `tf.data.Dataset` instance, yielding batched sequences of bitboards and associated evaluations. Computing the bitboard is optional for speed. At the low level this module opens two subprocesses; first opening a `bzcat` subprocess which sequentially decompresses and reads an archive of PGN files. The second is the modified version of `pgn-extract` in the vendor subdirectory, which parses the stream from `bzcat` and outputs a list of FEN strings which represent a game, with each string annotated with an evaluation. The `tagfile` defined in this module specifies which constraints on Elo are applied.
- `img_utils.py` allows converting a board, with its associated stack of moves, into a GIF of a game.
- `metrics.py` has a custom implementation of a `Loss` metrc, which is used by the models
- `model.py` has a `BaseGAN` class, which abstracts the custom tensorflow training loop and the interaction of the generator and discriminator models. The abstract methods of this class are implemented in `ConcreteGAN` and `RNNGAN` which specify models for generators and discriminators in different ways.
- `stockfish.py` implements a context manager `Stockfish` for the `Engine` class. Engine inherits from `subprocess.Popen` and opens a stockfish local process, also providing some methods for communicating with the engine through UCI commands. The context manager allows using the engine with the `with` keyword.

`badchess.py` is a glue script for training, loading and simulating games against one or more models. `fight_self.py` and `fight_stockfish.py` both implement grid searches which simulate multiple games between agents and output the results. These are pretty slow and arent accessable through the command line in the same way that `badchess.py` is.

## Build the games

1. Build a local binary of `pgn-extract` using the makefile at `vendor/pgn-extract/src/Makefile`
2. Install the stockfish command line tool, and ensure it is accessable in your `$PATH` and can be called by python.
3. Ensure the `bzcat` utility is available.
4. Download the data file, available [here](https://database.lichess.org). The file used in this training is from April 2018.

## Todos

- [x] Clip and normalize the evaluations to some reasonable number
- [x] Implement RNN stuff