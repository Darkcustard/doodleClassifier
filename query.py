import pymenu
import pygame
import numpy

from random import choice
from math import sqrt
from keras import models

import tensorflow as tf


limitVram = True

if limitVram:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)



win = pygame.display.set_mode((500,500))
pixels = []
classifier = models.load_model('classifier.ai')

prompts = ["Airplane","Bee","Banana","Eiffel Tower","Bicycle","Bulldozer"]


#region general ui
background = pymenu.Panel(win,
    {
        "pos": (0,0),
        "size": (500,500),
        "color": (130,168,200),
    })



foreground = pymenu.Panel(win,
    {
        "pos": (15,15),
        "size": (470,470),
        "color": (160,198,230),
        "parent": background,
        "outline": True,
        "outline_size": 1,
    })


prompt = pymenu.Panel(win,
    {
        "pos" : (70,110),
        "color" : (255,255,255),
        "size" : (250,30),
        "text" : choice(prompts),
        "text_pos" : (10,5),
        "text_size" : 30,

        "parent" : foreground,
    })


guess = pymenu.Panel(win,
    {
        "pos" : (70,450),
        "color" : (255,255,255),
        "size" : (250,30),
        "text" : "",
        "text_pos" : (10,5),
        "text_size" : 30,

        "parent" : foreground,


    })

def clearScreen():
    
    for row in pixels:
        for pixel in row:
            pixel.color = (0,0,0)


    prompt.text = choice(prompts)
    prompt.compile()

    guess.text = ''
    guess.compile()

button_clear = pymenu.Button(win,
    {
        "pos": (405,25),
        "size": (60,30),
        "color": (220,220,230),
        "parent": foreground,
        "color_hover": (210,210,220),
        "color_clicked": (200,200,210),
        "function_up": clearScreen,

        "text" : "Clear",
        "text_pos" : (10,8),
        "text_size" : 20,
    })



def queryAi():




    #format pixels (panels) into 'mnist-like' image
    image_raw = []
    for y in pixels:
        row = []
        for x in y:
            row.append((sum(x.color)/3))
        image_raw.append(row)
    
    image = numpy.array(image_raw)
    image = numpy.reshape(image,(1,28,28,1))

    
    output = list(classifier.predict(image)[0])
    
    description = ''

    if output.index(max(output)) == 0:
        description = 'airplane'

    if output.index(max(output)) == 1:
        description = 'bee'

    if output.index(max(output)) == 2:
        description = 'banana'

    if output.index(max(output)) == 3:
        description = 'eiffel tower'

    if output.index(max(output)) == 4:
        description = 'bicycle'

    if output.index(max(output)) == 5:
        description = 'bulldozer'

    guess.text = 'I think its a '+description
    guess.compile()

    print(description)
    
    


button_query = pymenu.Button(win,
    {
        "pos": (405,65),
        "size": (60,30),
        "color": (220,220,230),
        "parent": foreground,
        "color_hover": (210,210,220),
        "color_clicked": (200,200,210),
        "function_up": queryAi,
        
        "text" : "Query",
        "text_pos" : (10,8),
        "text_size" : 20,


    })


#region drawspace
drawspace_panel = pymenu.Panel(win,
    {
        "pos":(70,150),
        "parent": background,
        "size":(280,280),
        "color": (0,0,0),
    })


#generate drawing panels
for y in range(28):

    row = []

    for x in range(28):
        
        row.append(pymenu.Panel(win,
            {
                "pos":(70+x*10,150+y*10),
                "color":(0,0,0),
                "size":(10,10),
                "parent": drawspace_panel,
                "round_radius" : 0,
            }))

    pixels.append(row)





#endregion


def handleDrawing(pixels):

    radius = 6
    x, y = pygame.mouse.get_pos()

    for row in pixels:
        for pixel in row:

            center = (pixel.pos[0]+pixel.size[0]/2,pixel.pos[1]+pixel.size[1]/2)
            distance = sqrt((center[0]-x)**2+(center[1]-y)**2)

            if distance < radius:
                pixel.color = (255,255,255)





#endregion


run = True
while run:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    #checking if mouse clicked
    left, _, _ = pygame.mouse.get_pressed()
    
        
    

    background.draw()
    if left:
        handleDrawing(pixels)
    pygame.display.update()