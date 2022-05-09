import itertools
import pprint

from tqdm import tqdm
from badchess import run_game_vs_self, parser


def tourney(match_args, n_games):
    outcomes = []
    for i in tqdm(range(n_games)):
        winner, termination, ply = run_game_vs_self(match_args)
        outcomes.append(winner)

    engine_wins = len([i for i in outcomes if i == True])
    draws = len([i for i in outcomes if i is None])
    stockfish_wins = len([i for i in outcomes if i is False])

    return engine_wins, draws, stockfish_wins

def grid_search(n_games: int = 5):
    # Engine parameters
    engine_depths = ['4']
    # These models need to be switched
    whitemodels = ["./models/generator_upper.tflite", "./models/generator_lower.tflite"]
    blackmodels = ["./models/generator_lower.tflite", "./models/generator_upper.tflite"]
    param = ["playself"]
    chunks = ['4']
    results = []

    # Get all the combinations
    searches = itertools.product(
        param,
        whitemodels,
        blackmodels,
        ["--chunk_size"], chunks,
        ["--engine_depth"], engine_depths
    )

    # Remove duplicates (if there are any)
    searches = set(searches)

    # Play a tournament for each combination
    for i in searches:
        # Construct the arguments for run game
        args = parser.parse_args(i)

        # Display the parameters as a dict
        params = vars(args)
        pprint.pprint(params)

        # Run the tournament
        engine_wins, draws, stockfish_wins = tourney(args, n_games)

        # Save the match games
        match_results = {
            "Engine wins": engine_wins,
            "Draws": draws,
            "Stockfish wins": stockfish_wins
        }

        # Save the results!
        results.append((params, match_results))
        pprint.pprint(match_results)
    return results

if __name__ == '__main__':
    gridsearchres = grid_search(5)