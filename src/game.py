from arch import Agent, Menu, Game
import argparse
import sys

# TODO flake8

# TODO potentially check with more states
# TODO to see if it would improve "few rounds" score
# TODO check how to handle loops with nontraining


def parse_args():
    parser = argparse.ArgumentParser(description="Learn2Slither.")

    parser.add_argument(
        '-load', type=str, metavar='FILENAME',
        help='Path to load saved model (disables board size input).'
    )
    parser.add_argument(
        '-save', type=str, metavar='FILENAME',
        help='Path to save the model.'
    )

    parser.add_argument(
        '-size', type=int, nargs=2, metavar=('ROWS', 'COLS'),
        default=[10, 10], help='Set board size with rows and cols.'
    )
    parser.add_argument(
        '-sessions', type=int, metavar='N',
        default=100, help='Number of training sessions.'
    )
    parser.add_argument(
        '-fps', type=int, metavar='N',
        default=40, help='Display speed (frames per second).'
    )

    parser.add_argument(
        '-v', '--visual', action='store_true',
        help='Display visuals during training.'
    )
    parser.add_argument(
        '-n', '--nolearn', action='store_true',
        help='Disable learning (evaluation mode).'
    )
    parser.add_argument(
        '-s', '--stepbystep', action='store_true',
        help='Enable step-by-step mode.'
    )
    parser.add_argument(
        '-m', '--manual', action='store_true',
        help='Enable manual play.'
    )

    return parser


def verify_args(args):
    assert args.size[0] > 2 and args.size[1] > 2, "improper board size"
    if args.manual:
        args.visual = True
    if args.visual:
        assert args.size[0] <= 100 and args.size[1] <= 100, "large board size"
    else:
        assert not args.stepbystep, "step-by-step reserved for GUI"
    return args


def main():
    parser = parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    args = verify_args(args)

    if args.visual:
        m = Menu(args)
        m.run()
        args = m.args

    agent = Agent(14)
    agent.load(args.load)

    g = Game(args)
    if args.manual:
        g.runManual()
    else:
        g.run(agent)

    # if args.visual:
    #     plotStats(g.lengths)

    agent.save(args.save)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\033[91m" + "Error: " + str(e) + "\033[0m")
        sys.exit(1)
