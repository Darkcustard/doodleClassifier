from inspect import walktree
import pygame
from lib import pymenu

PALETTE = pygame.display.set_mode((1920,1080))













running = True

while running:

    PALETTE.fill((150,150,150))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    pygame.display.update()