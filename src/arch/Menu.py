from .Button import Button
import pygame
import sys

class Configuration:
    def __init__(self):
        print("CONFY")
        # TODO load and save texts
        # save under same name conf
        # size input, button to increase/decrease
        # session input
        # stepbystep and nolearn switches
 

class Menu:
    def __init__(self, args):
        self.args = args
        self.WIDTH = 800
        self.HEIGHT = 800

    def renderMenu(self):
        image = pygame.image.load("assets/Menu.png").convert()
        image = pygame.transform.scale(image, (self.WIDTH, self.HEIGHT))
        self.screen.blit(image, (0, 0))
        self.playButton = Button((250, 520), (300, 100))
        self.confButton = Button((250, 650), (130, 120))
        self.exitButton = Button((420, 650), (130, 120))
        pygame.display.update()

    def exit(self):
        pygame.quit()
        sys.exit(0)

    def menu_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.exit()
            elif event.type == pygame.QUIT:
                    self.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # print(event.pos)
                if self.playButton.rect.collidepoint(event.pos):
                    self.running = False
                elif self.confButton.rect.collidepoint(event.pos):
                    print("Conf")
                elif self.exitButton.rect.collidepoint(event.pos):
                    self.exit()

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