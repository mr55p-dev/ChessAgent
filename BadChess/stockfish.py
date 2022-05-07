from subprocess import Popen, PIPE


# Open an engine process and write the newgame command
engine = Popen(["stockfish"], stdin=PIPE, stdout=PIPE)
engine.stdin.write(b"ucinewgame\n")
engine.stdin.flush()

def get_move(fen, movetime: int = 1000) -> str:
    """Generator to write a new position to the engine. movetime is the duration to search for in ms"""
    # Write the commands to the input
    engine.stdin.write(bytes(f"position {fen}\n", encoding="utf-8"))
    engine.stdin.write(bytes(f"go movetime {movetime}\n", encoding="utf-8"))
    engine.stdin.flush()

    # Go over each output line and yield the bestmove output should
    # it match the correct format
    for line in engine.stdout:
        if len((split := str(line).strip().split(' '))) == 4:
            yield split[1]
        else:
            continue

def compute_stockfish_move(fen: str) -> str:
    """Get a stockfish move given a position"""
    move = next(get_move(fen))
    return move


move = compute_stockfish_move(fen="rnb1kbnr/ppp1q1pp/8/3p3Q/3Pp3/8/PPP2PPP/RN2KBNR b KQkq - 1 5")

engine.terminate()

print(move)