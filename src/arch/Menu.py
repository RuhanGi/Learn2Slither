from .Slider import Slider
import pygame
import sys

def exits():
    pygame.quit()
    sys.exit(0)

class Configuration:
    def __init__(self, screen, args):
        self.screen = screen
        self.sliders = [
            Slider((100, 100), (600, 100), 5, 1, 10), # ? LOAD [textbox]
            Slider((100, 270), (200, 60), 0, 0, 1), # ? [switch] SAVE same as LOAD
            Slider((400, 250), (300, 100), 5, 1, 10), # ? SAVE [textbox]
            Slider((200, 450), (200, 100), 10, 5, 20), # ? ROWS
            Slider((100, 550), (100, 200), 10, 5, 20), # ? COLS
            # Slider((100, 100), (600, 60), 100, 1, 2000), # ? [textbox] SESSIONS
            Slider((500, 470), (200, 60), 0, 0, 1), # ? [switch] stepbystep
            Slider((500, 620), (200, 60), 0, 0, 1) # ? [switch] nolearn
        ]
        self.running = True
        clock = pygame.time.Clock()
        while self.running:
            self.renderConf()
            self.event_handler()
            clock.tick(60)

    def event_handler(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    exits()
            elif event.type == pygame.QUIT:
                exits()
            # elif event.type == pygame.MOUSEBUTTONDOWN:

    def renderConf(self):
        self.screen.fill('#A8E61D')
        for s in self.sliders:
            s.render(self.screen)
        pygame.display.update()

        # TODO conf 
        # ? LOAD [textbox]
        # ? [button] to save/update model
        # ? SAVE [textbox]
        # ?    SIZE      SESSION
        # ?  /\ <=10=>   [textbox]
        # ?  10         stepbystep [switch]
        # ?  \/         nolearn [switch]


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
                    # print("Conf")
                    Configuration(self.screen, self.args)
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
            clock.tick(60)
        pygame.quit()

    def getArgs(self):
        return self.args