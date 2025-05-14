import argparse
import sys


RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
CYAN = "\033[96m"
GRAY = "\033[97m"
BLACK = "\033[98m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(description='Training script for your model.')
    parser.add_argument('mod', type=str, default="[MODEL]", help='Path to model file')
    parser.add_argument('--layers', nargs='+', type=int, default=[24, 15],
                        help='Number of nodes in each layer, e.g., --layers 24 15')
    args = parser.parse_args()
    # * load with model
    # * save model as
    # * max sessions
    # * have visuals
    # * no learning
    # * boardsize

    print(GREEN + args.mod + RESET)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)
