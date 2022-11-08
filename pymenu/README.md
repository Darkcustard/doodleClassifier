
# pymenu
Pygame based library for ui simplification
Elements can be individually handled or parented to another element to be handled automatically under the parents.draw() method
Each element has a lot of customization but default values permit you to use the library with very little code.

# Installation

1. Download or clone the pymenu repository into your project folder

2. Install dependancies: "pip install -r requirements.txt"

# Usage

1. Import the pymenu library: "import pymenu"

2. Import the pygame library: "import pygame"

3. Create a pygame window: "WIN = pygame.display.set_mode((500,500))"

4. Define your menu elements "button = pymenu.Button(WIN,{})"

5. Run the element's .draw() method to draw it and its child elements.

6. Run "pygame.display.update()" to refresh the window.



