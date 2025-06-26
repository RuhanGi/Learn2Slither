import pygame

SELECTED = {
    True:"white",
    False:"brown"
}

class Slider:
    def __init__(self, topleft: tuple, size: tuple, initial: int, min: int, max: int, widthB=20) -> None:
        self.topleft = topleft
        self.size = size
        self.hovered = False
        self.grabbed = False

        self.min = min
        self.max = max

        if initial < min:
            inital = min
        elif initial > max:
            inital = max

        self.value = initial
        x = (initial-min) / (max - min) * size[0]

        self.container_rect = pygame.Rect(self.topleft[0], self.topleft[1], self.size[0], self.size[1])
        self.button_rect = pygame.Rect(self.topleft[0] + x - widthB/2, self.topleft[1] - 5, widthB, self.size[1] + 10)
        
    def move_slider(self):
        pos = pygame.mouse.get_pos()[0]
        if pos < self.topleft[0]:
            pos = self.topleft[0]
        if pos > self.topleft[0] + self.size[0]:
            pos = self.topleft[0] + self.size[0]
        
        button_val = pos - self.topleft[0]
        self.value = round((button_val/self.size[0]) * (self.max-self.min) + self.min)
        self.button_rect.centerx = self.topleft[0] + (self.value-self.min) / (self.max-self.min) * self.size[0]

    def render(self, screen):
        if self.hovered:
            self.move_slider()
        pygame.draw.rect(screen, "darkgreen", self.container_rect)
        pygame.draw.rect(screen, SELECTED[self.hovered], self.button_rect)

    def get_value(self):
        return self.value
        val_range = self.size[0]
        button_val = self.button_rect.centerx - self.topleft[0]
        return round((button_val/val_range) * (self.max-self.min) + self.min)
