import numpy as np

RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
PURPLE  = "\033[35m"
CYAN    = "\033[36m"
GRAY    = "\033[90m"
BLACK   = "\033[30m"
WHITE   = "\033[37m"
RESET   = "\033[0m"

class Board:
    """
        Board
    """
    # TODO REMEMBER FLAKE8
    def __init__(self, size):
        self.rows = size[0]+2
        self.cols = size[1]+2
        self.createBoard()
        self.printBoard()
        self.printVision()

    def sendState(self):
        print()
        # dangers
        # 

    def get_random_empty_cell(self):
        zero_cells = np.argwhere(self.board == '0')
        if zero_cells.size == 0:
            print("NO EMPTY CELLS LEFT") # ! HANDLE THIS
        return tuple(zero_cells[np.random.choice(len(zero_cells))])

    def get_empty_cell(self, coord):
        i, j = coord
        neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        valid_neighbors = [(r, c) for r, c in neighbors if self.board[r, c] == '0']

        if not valid_neighbors:
            return None # ! HANDLE THIS
        return valid_neighbors[np.random.choice(len(valid_neighbors))]

    def createBoard(self):
        self.board = np.full((self.rows, self.cols), '0', dtype='<U1')
        self.board[[0, -1], :] = 'W'
        self.board[:, [0, -1]] = 'W'
        self.snake = [self.get_random_empty_cell()]
        for i in range(2):
            self.board[self.snake[-1]] = 'S'
            self.snake.append(self.get_empty_cell(self.snake[-1]))

        self.board[self.snake[-1]] = 'H'
        self.board[self.get_random_empty_cell()] = 'G'
        self.board[self.get_random_empty_cell()] = 'G'
        self.board[self.get_random_empty_cell()] = 'R'

    def printBoard(self):
        col = {'W': GRAY, '0': BLACK, 'S': CYAN, 'H': PURPLE, 'G': GREEN, 'R': RED}
        for row in self.board:
            for cell in row:
                print(col[cell] + cell, end='')
            print(RESET)

    def printVision(self):
        r, c = self.snake[-1]
        col = {'W': GRAY, '0': BLACK, 'S': CYAN, 'H': PURPLE, 'G': GREEN, 'R': RED}
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                print(self.board[i][j]if i==r or j==c else ' ', end='')
            print()

