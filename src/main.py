from arch import Environment, Agent
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

def parse_args():
    parser = argparse.ArgumentParser(description="Learn2Slither.")

    parser.add_argument('-load', type=str, metavar='FILENAME',
        help='Path to load saved model (disables board size input).')
    parser.add_argument('-save', type=str, metavar='FILENAME',
        help='Path to save the model.')

    parser.add_argument('-size', type=int, nargs=2, metavar=('ROWS', 'COLS'),
        default=[10, 10], help='Set board size with rows and cols.')
    parser.add_argument('-max', type=int, metavar='N',
        default=100, help='Maximum number of training sessions.')

    parser.add_argument('-v', '--visual', action='store_true',
        help='Display visuals during training.')
    parser.add_argument('-n', '--nolearn', action='store_true',
        help='Disable learning (evaluation mode).')
    parser.add_argument('-s', '--stepbystep', action='store_true',
        help='Enable step-by-step mode.')

    return parser.parse_args()

def printArgs(args):
    print(GREEN + str(args.load) + RESET) 
    print(GREEN + str(args.save) + RESET)
    print(GREEN + str(args.size) + RESET)
    print(GREEN + str(args.max) + RESET) #handle
    print(GREEN + str(args.visual) + RESET) #handle
    print(GREEN + str(args.nolearn) + RESET) #handle
    print(GREEN + str(args.stepbystep) + RESET) #handle

def displayVisual():
    print()
    # TODO Slider adjusting speed
    # TODO Button to save state anytime
    # TODO lobby
    # TODO a configuration panel,
    # TODO results and statistics
    # TODO ...


def compute():
    print()
    # TODO display snake vision in terminal
    displayVisual()
    # TODO Exploration vs Exploitation: choose random instead of best


def main():
    args = parse_args()

    agent = Agent(args.load)
    b = Environment(args.size)

    if args.save is not None:
        # TODO save model
        print(GREEN + "Model Saved!", RESET)

if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except Exception as e:
    #     print(RED + "Error: " + str(e) + RESET)
    #     sys.exit(1)
