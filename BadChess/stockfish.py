from subprocess import Popen, PIPE

class Stockfish:
    """Context wrapper for an engine (implemented on top of subprocess.Popen)"""
    def __init__(self, movetime):
        self._engine = Engine(movetime)

    def __enter__(self):
        self._engine._flush_uci()
        return self._engine

    def __exit__(self, *args):
        """Kill the process"""
        self._engine.terminate()

# Open an engine process and write the newgame command
class Engine(Popen):
    def __init__(self, movetime=1000) -> None:
        super().__init__(["stockfish"], stdin=PIPE, stdout=PIPE)
        # Set up the engine a bit
        self.stdin.write(b"uci\n")
        self.stdin.write(b"isready\n")
        self.stdin.write(b"ucinewgame\n")
        self.stdin.flush()

        # Save the movetime setting
        self._movetime = movetime

    def _flush_uci(self):

        self.stdin.flush()

    def set_state(self, move_hist):
        hist_str = f"position startpos moves {' '.join(move_hist)}\n"
        self.stdin.write(bytes(hist_str, encoding="utf-8"))
        self.stdin.flush()

    def get_move(self) -> str:
        """Generator to write a new position to the engine. movetime is the duration to search for in ms"""
        # Write the commands to the input
        self.stdin.write(bytes(f"go movetime {self._movetime}\n", encoding="utf-8"))
        self.stdin.flush()

        # Go over each output line and yield the bestmove output should
        # it match the correct format
        for line in self.stdout:
            # print(line)
            if (split := line.decode().strip().split(' '))[0] == 'bestmove':
                return str(split[1])
        raise ValueError("Stockfish failed to yield a best move.")