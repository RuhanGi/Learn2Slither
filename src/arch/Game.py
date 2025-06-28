import numpy as np
import pygame
import signal
from .Interpreter import Interpreter
from .Slider import Slider


class Game:
    """
        Game Mechanics
    """
    # TODO REMEMBER FLAKE8

    def __init__(self, args):
        self.rows = args.size[0] + 2
        self.cols = args.size[1] + 2
        self.scale = (min(800/self.rows, 800/self.cols)).__ceil__()
        self.WIDTH = self.scale * self.cols
        self.HEIGHT = self.scale * self.rows

        self.lengths = []
        self.durations = []

        if args.visual:
            pygame.init()
            size = (self.WIDTH, self.HEIGHT + 100)
            self.screen = pygame.display.set_mode(size)
            self.slider = Slider(
                (self.WIDTH/8, self.HEIGHT + 20),
                (self.WIDTH * 3/4, 60),
                args.fps, 1, 60
            )
            pygame.display.set_caption('Learn2Slither')

        self.running = True
        self.createBoard()

    def updateSnake(self):
        self.board[(self.board == 'S') | (self.board == 'H')] = '0'
        for i in range(len(self.snake)-1):
            self.board[self.snake[i]] = 'S'
        self.board[self.snake[-1]] = 'H'

    def get_random_empty_cell(self):
        zero_cells = np.argwhere(self.board == '0')
        if zero_cells.size == 0:
            return None
        return tuple(zero_cells[np.random.choice(len(zero_cells))])

    def get_empty_cell(self, coord):
        i, j = coord
        near = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        valid = [(r, c) for r, c in near if self.board[r, c] == '0']

        if not valid:
            return None
        return valid[np.random.choice(len(valid))]

    def createBoard(self):
        self.board = np.full((self.rows, self.cols), '0', dtype='<U1')
        self.board[[0, -1], :] = 'W'
        self.board[:, [0, -1]] = 'W'
        self.snake = [self.get_random_empty_cell()]
        for i in range(2):
            self.board[self.snake[-1]] = 'S'
            self.snake.append(self.get_empty_cell(self.snake[-1]))
        self.board[self.snake[-1]] = 'H'
        mapps = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
        change = tuple(np.array(self.snake[-1])-np.array(self.snake[-2]))
        self.direction = mapps[change]
        self.alive = True
        self.moves = 0

        self.board[self.get_random_empty_cell()] = 'G'
        self.board[self.get_random_empty_cell()] = 'G'
        self.board[self.get_random_empty_cell()] = 'R'

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

    def draw(self, path, x, y, rotate=0):
        surf = pygame.image.load(path)
        surf = pygame.transform.smoothscale(surf, (self.scale, self.scale))
        if rotate != 0:
            surf = pygame.transform.rotate(surf, rotate)
        surf_rect = surf.get_rect()
        surf_rect.topleft = (self.scale * x, self.scale * y)
        self.screen.blit(surf, surf_rect)

    def renderBoard(self):
        imgmap = {
            'W': './assets/Wall.png',
            'H': './assets/Head.png',
            'G': './assets/GreenApple.png',
            'R': './assets/RedApple.png'
        }

        self.screen.fill('#A8E61D')
        for i in range(self.rows):
            for j in range(self.cols):
                if (i + j) % 2 == 0:
                    self.draw('./assets/Grass.png', j, i)

        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] not in '0SH':
                    self.draw(imgmap[self.board[i][j]], j, i)
                elif self.board[i][j] == 'H':
                    self.draw(
                        imgmap[self.board[i][j]], j, i,
                        rotate=-90 * self.direction
                    )

        if len(self.snake) > 1:
            mapps = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
            temp = np.array(self.snake[0])
            direc1 = mapps[tuple(temp - np.array(self.snake[1]))]
            self.draw('./assets/Tail.png', temp[1], temp[0], rotate=direc1*-90)
            for i in range(1, len(self.snake) - 1):
                temp = np.array(self.snake[i])
                direc1 = mapps[tuple(temp - np.array(self.snake[i-1]))]
                direc2 = mapps[tuple(temp - np.array(self.snake[i+1]))]
                if abs(direc1 - direc2) == 2:
                    self.draw(
                        './assets/Snake.png', temp[1], temp[0],
                        rotate=-90 * direc1
                    )
                elif (direc1 - direc2) % 4 == 1:
                    self.draw(
                        './assets/CornerSnake.png', temp[1], temp[0],
                        rotate=-90 * (direc2+2)
                    )
                else:
                    self.draw(
                        './assets/CornerSnake.png', temp[1], temp[0],
                        rotate=-90 * (direc1+2)
                    )

        self.slider.render(self.screen)

    def move(self, direction):
        self.direction = direction
        self.moves += 1
        moves = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        pos = tuple(np.array(self.snake[-1]) + moves[direction])

        orig = self.board[pos]
        if orig != 'G':
            self.board[self.snake.pop(0)] = '0'
        i = self.board[pos]
        self.snake.append(pos)

        if i == 'G' or i == 'R':
            if i == 'R':
                del self.snake[0]
            cell = self.get_random_empty_cell()
            if cell:
                self.board[cell] = i

        if i in ['S', 'W'] or len(self.snake) == 0:
            self.alive = False
        elif len(self.snake) > 0:
            self.updateSnake()

        done = not self.alive
        if done:
            self.lengths.append(len(self.snake))
            self.durations.append(self.moves)
            print(
                f"\rLength = {len(self.snake)}, " +
                f"Duration = {self.moves}    \t",
                end=""
            )
            self.sesscount += 1
            self.createBoard()

        rewards = {'G': 10, 'R': -10, '0': -1, 'S': -100, 'W': -100}
        return rewards[orig], done

    def event_handler(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif self.args.stepbystep:
                    self.greenlight = True
            elif event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.slider.container_rect.collidepoint(event.pos):
                    self.slider.hovered = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.slider.hovered = False

    def run(self, agent, args):
        clock = pygame.time.Clock()
        self.args = args
        self.sesscount = 0

        def handle_sigint(signal_number, frame):
            self.running = False
        signal.signal(signal.SIGINT, handle_sigint)

        if args.stepbystep:
            self.greenlight = False

        state = self.getState()
        while self.running and self.sesscount < args.sessions:

            if not args.stepbystep or self.greenlight:
                action = agent.act(state, args)
                reward, done = self.move(action)
                next_state = self.getState()
                if not args.nolearn:
                    agent.train_step(state, action, reward, next_state, done)
                state = next_state
                if args.stepbystep:
                    self.greenlight = False

            if args.visual:
                self.event_handler()
                self.renderBoard()
                args.fps = self.slider.get_value()
                pygame.display.update()
            clock.tick(args.fps)
        pygame.quit()

        if len(self.lengths) == 0:
            print(
                f"\rLength = {len(self.snake)},",
                f"Duration = {self.moves}"
            )
        else:
            print(
                f"\rMax Length = {np.max(self.lengths)},",
                f"max duration = {np.max(self.durations)}"
            )

    def printVision(self, vision):
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
