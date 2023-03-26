import pygame
from PIL.Image import Image

pygame.init()
width, height = 300, 300
screen = pygame.display.set_mode((width, height))

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
line_width = 5
last_pos = None


def save_image(surface):
    filename = 'drawing.png'
    pygame.image.save(surface, filename)
    print(f"Image saved as '{filename}'")


def draw_smth(event):
    global last_pos
    if event.type == pygame.MOUSEBUTTONDOWN:
        last_pos = event.pos
    elif event.type == pygame.MOUSEMOTION:
        if last_pos is not None:
            pygame.draw.line(screen, BLACK, last_pos, event.pos, line_width)
            last_pos = event.pos
    elif event.type == pygame.MOUSEBUTTONUP:
        last_pos = None

    # Check for save button press
    if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
        save_image(screen)

# Set up screen and caption
pygame.display.set_caption("Drawing App")

# Fill background with white
screen.fill(WHITE)

# Run the game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        draw_smth(event)

    pygame.display.update()

# Quit the game
pygame.quit()
