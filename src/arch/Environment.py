from .Interpreter import Interpreter
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

import os
import time

class Environment:
    """
        Environment
    """
    # TODO REMEMBER FLAKE8
    def __init__(self, size):
        self.rows = size[0]+2
        self.cols = size[1]+2

    def getVision(self):
        r, c = self.snake[-1]
        vision = []
        vision.append(self.board[r-1::-1, c])
        vision.append(self.board[r, c+1:])
        vision.append(self.board[r+1:, c])
        vision.append(self.board[r, c-1::-1])
        return vision

    def getState(self):
        vision = self.getVision()
        # self.printVision(vision)
        return Interpreter.getState(vision)

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
        self.alive = True

        self.board[self.get_random_empty_cell()] = 'G'
        self.board[self.get_random_empty_cell()] = 'G'
        self.board[self.get_random_empty_cell()] = 'R'

    def updateSnake(self):
        self.board[(self.board == 'S') | (self.board == 'H')] = '0'
        for i in range(len(self.snake)-1):
            self.board[self.snake[i]] = 'S'
        self.board[self.snake[-1]] = 'H'

    def move(self, direction):
        moves = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        pos = tuple(np.array(self.snake[-1]) + moves[direction])

        orig = self.board[pos]
        if orig != 'G':
            self.board[self.snake.pop(0)] = '0'
        i = self.board[pos]
        self.snake.append(pos)

        if i == 'G':
            self.board[self.get_random_empty_cell()] = 'G'
        elif i == 'R':
            del self.snake[0]
            self.board[self.get_random_empty_cell()] = 'R'

        if i in ['S', 'W'] or len(self.snake) == 0:
            print(RED + "Snake DIED!" + RESET)
            self.alive = False

        self.printBoard()
        rewards = {'G': 10, 'R': -10, '0': -1, 'S': -100, 'W': -100}
        return rewards[orig]

    def start(self, agent, max):
        print("SIM STARTED!")
        for iteration in range(max):
            self.createBoard()
            self.printBoard()
            state = self.getState()
            while self.alive:
                action = agent.act(state)
                reward = self.move(action)
                next_state = self.getState()
                agent.train_step(state, action, reward, next_state, not self.alive)
                state = next_state
                time.sleep(0.2)

    def printBoard(self):
        os.system("clear")
        if (len(self.snake) > 0):
            self.updateSnake()
        col = {'W': GRAY, 'S': PURPLE, 'H': YELLOW, 'G': GREEN, 'R': RED}
        for row in self.board:
            for cell in row:
                if cell != '0':
                    print(col[cell] + cell, end='')
                else:
                    print(' ', end='')
            print(RESET)

    def printVision2(self, vision):
        padd = len(vision[-1])
        for c in reversed(vision[0]):
            print(padd * ' ' + c)
        for c in reversed(vision[-1]):
            print(c, end='')
        print('S', end='')
        for c in vision[1]:
            print(c, end='')
        print()
        for c in vision[2]:
            print(padd * ' ' + c)

    def printVision(self):
        r, c = self.snake[-1]
        col = {'W': GRAY, '0': BLACK, 'S': CYAN, 'H': BLUE, 'G': GREEN, 'R': RED}
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                print(self.board[i][j]if i==r or j==c else ' ', end='')
            print()
