import pygame

class Button():
    def __init__(self, topleft, size, text=None):
        self.topleft = topleft
        self.size = size
        self.rect = pygame.Rect(topleft, size)
        self.text = text
    
    # def draw(self, screen):
        # color = (200, 100, 100)
        # hovcolor = (100, 200, 100)
        # mouse = pygame.mouse.get_pos()
        # if self.topleft[0] <= mouse[0] <= self.topleft[0] + self.size[0] and self.topleft[1] <= mouse[1] <= self.topleft[1] + self.size[1]:
        #     pygame.draw.rect(self.screen, hovcolor, [600, 200, 70, 50])
        # else:
        #     pygame.draw.rect(self.screen, color, [600, 200, 70, 50])
