from subprocess import Popen, PIPE

class Engine:
    def __init__(self):
        self._engine = Popen(["stockfish"], stdin=PIPE, stdout=PIPE)

    def __enter__(self):
        self._engine.stdin.write(b"ucinewgame\n")
        self._engine.stdin.flush()
        return self._engine

    def __exit__(self, *args):
        self._engine.terminate()

# Open an engine process and write the newgame command


def get_move(enigne_ctx, fen, movetime: int = 1000) -> str:
    """Generator to write a new position to the engine. movetime is the duration to search for in ms"""
    # Write the commands to the input
    enigne_ctx.stdin.write(bytes(f"position {fen}\n", encoding="utf-8"))
    enigne_ctx.stdin.write(bytes(f"go movetime {movetime}\n", encoding="utf-8"))
    enigne_ctx.stdin.flush()

    # Go over each output line and yield the bestmove output should
    # it match the correct format
    for line in enigne_ctx.stdout:
        if len((split := str(line).strip().split(' '))) == 4:
            return str(split[1])
        else:
            continue
    raise ValueError("Stockfish failed to yield a best move.")

with Engine() as engine_ctx:
    move = get_move(engine_ctx, fen="rnb1kbnr/ppp1q1pp/8/3p3Q/3Pp3/8/PPP2PPP/RN2KBNR b KQkq - 1 5")

print(move)