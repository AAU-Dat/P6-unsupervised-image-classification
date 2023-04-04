import pygame
import torchvision

# Initialize the game engine
pygame.init()
# Set up the drawing window
width, height = 300, 300
# Create the screen
screen = pygame.display.set_mode((width, height))

# Background color
WHITE = (255, 255, 255)
# Line color
BLACK = (0, 0, 0)
# Line width
line_width = 20
# Last position
last_pos = None


def save_image(surface):
    pygame.image.save(surface, 'temp.png')
    save_img = torchvision.io.read_image('./temp.png', torchvision.io.ImageReadMode.GRAY)

    print(f"Shape of save_img: {save_img.shape}")

    # Transform the image to 28x28 pixels
    save_img = torchvision.transforms.functional.resize(save_img, (28, 28))
    
    # Save the image
    torchvision.io.write_png(save_img, './output.png')
    
# Clear the screen and start over
def clear_screen():
    # Fill background with white
    screen.fill(WHITE)

# Draw on the screen
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

    # Check for clear button press
    if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
        clear_screen()

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
