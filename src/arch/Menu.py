from .Slider import Slider
import pygame
import sys


def exits():
    pygame.quit()
    sys.exit(0)


# ? SESSIONS, ROWS, COLS, STEPBYSTEP, NOLEARN

class Configuration:
    def __init__(self, screen, args):
        self.screen = screen
        self.sliders = [
            Slider((460, 130), (230, 70), args.sessions, 1, 2000),
            Slider((460, 230), (230, 70), args.size[0], 5, 20),
            Slider((460, 330), (230, 70), args.size[1], 5, 20),
            Slider((610, 435), (60, 60), args.stepbystep, 0, 1, widthB=40),
            Slider((610, 535), (60, 60), args.nolearn, 0, 1, widthB=40)
        ]
        self.exitButton = pygame.Rect((250, 630), (300, 100))

        pygame.display.set_caption('Configuration Panel')
        self.running = True
        clock = pygame.time.Clock()
        while self.running:
            self.renderConf()
            self.event_handler()
            clock.tick(60)
        self.setArgs(args)

    def setArgs(self, args):
        args.sessions = self.sliders[0].value
        args.size[0] = self.sliders[1].value
        args.size[1] = self.sliders[2].value
        args.stepbystep = self.sliders[3].value
        args.nolearn = self.sliders[4].value
        self.args = args

    def event_handler(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    exits()
            elif event.type == pygame.QUIT:
                exits()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for s in self.sliders:
                    if s.container_rect.collidepoint(event.pos):
                        s.hovered = True
                if self.exitButton.collidepoint(event.pos):
                    self.running = False
            elif event.type == pygame.MOUSEBUTTONUP:
                for s in self.sliders:
                    s.hovered = False

    def text(self, text, font, text_col, x, y):
        img = font.render(text, True, text_col)
        self.screen.blit(img, (x, y))

    def renderConf(self):
        image = pygame.image.load("assets/Conf.png").convert()
        image = pygame.transform.scale(image, (800, 800))
        self.screen.blit(image, (0, 0))
        for s in self.sliders:
            s.render(self.screen)
        font = pygame.font.SysFont("Arial", 60)
        self.text(str(self.sliders[1].value), font, (240, 204, 114), 350, 230)
        self.text(str(self.sliders[2].value), font, (240, 204, 114), 350, 330)
        pygame.display.update()


class Menu:
    def __init__(self, args):
        self.args = args
        self.WIDTH = 800
        self.HEIGHT = 800

    def renderMenu(self):
        image = pygame.image.load("assets/Menu.png").convert()
        image = pygame.transform.scale(image, (self.WIDTH, self.HEIGHT))
        self.screen.blit(image, (0, 0))

        self.playButton = pygame.Rect((250, 520), (300, 100))
        self.confButton = pygame.Rect((250, 650), (130, 120))
        self.exitButton = pygame.Rect((420, 650), (130, 120))

        pygame.display.update()

    def menu_handler(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    exits()
            elif event.type == pygame.QUIT:
                exits()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.playButton.collidepoint(event.pos):
                    self.running = False
                elif self.confButton.collidepoint(event.pos):
                    c = Configuration(self.screen, self.args)
                    self.args = c.args
                    pygame.display.set_caption('Menu')
                    self.renderMenu()
                elif self.exitButton.collidepoint(event.pos):
                    exits()
        pygame.display.update()

    def run(self):
        pygame.init()
        pygame.display.set_caption('Menu')

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.renderMenu()
        self.running = True

        clock = pygame.time.Clock()
        while self.running:
            self.menu_handler()
            clock.tick(30)
        pygame.quit()
