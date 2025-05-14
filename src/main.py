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
    parser = argparse.ArgumentParser()
    parser.add_argument('-load', nargs=1, type=str,
        help='Load Model')
    parser.add_argument('-save', nargs=1, type=str,
        help='Save Model As')
    parser.add_argument('-size', nargs=2, type=int, default=[10, 10],
        help='Set Boardsize')
    parser.add_argument('-max', nargs=1, type=int, default=100,
        help='Max Training Sessions')
    parser.add_argument('-visual', help='Display Visuals')
    parser.add_argument('-nolearning', help='Stop Learning')
    args = parser.parse_args()

    print(GREEN + str(args.size) + RESET)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)
