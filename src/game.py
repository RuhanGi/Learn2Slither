from arch import Game

def main():
    g = Game(10, 10)
    g.run()


if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except Exception as e:
    #     print(RED + "Error: " + str(e) + RESET)
    #     sys.exit(1)
