import pygame
import predictModel

# Constants
BLACK, WHITE, RED, GREEN = [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0]
INITIAL_COLOR, FONT_SIZE = (255, 128, 0), 500
WIDTH, HEIGHT, RADIUS = 640, 640, 7

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH * 2, HEIGHT))
screen.fill(WHITE)
pygame.font.init()

def draw_partition_line():
    pygame.draw.line(screen, BLACK, [WIDTH, 0], [WIDTH, HEIGHT], 8)

def roundline(srf, color, start, end, radius=1):
    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.draw.circle(srf, color, (x, y), radius)

def main_loop():
    draw_on, last_pos = False, (0, 0)
    try:
        while True:
            draw_partition_line()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    raise StopIteration
                if e.type == pygame.MOUSEBUTTONDOWN:
                    if e.button == 3:
                        screen.fill(WHITE)
                    else:
                        pygame.draw.circle(screen, BLACK, e.pos, RADIUS)
                        draw_on, last_pos = True, e.pos
                if e.type == pygame.MOUSEBUTTONUP and e.button != 3:
                    draw_on = False
                    img = crope(screen)
                    pygame.image.save(img, "out.png")
                    output_img = predictModel.predictedImage("out.png")
                    show_output_image(output_img)
                if e.type == pygame.MOUSEMOTION and draw_on:
                    pygame.draw.circle(screen, BLACK, e.pos, RADIUS)
                    roundline(screen, BLACK, e.pos, last_pos, RADIUS)
                    last_pos = e.pos
            pygame.display.flip()
    except StopIteration:
        pygame.quit()

def crope(original):
    cropped = pygame.Surface((WIDTH - 5, HEIGHT - 5))
    cropped.blit(original, (0, 0), (0, 0, WIDTH - 5, HEIGHT - 5))
    return cropped

def show_output_image(img):
    surf = pygame.pixelcopy.make_surface(img)
    surf = pygame.transform.rotate(surf, -270)
    surf = pygame.transform.flip(surf, 0, 1)
    screen.blit(surf, (WIDTH + 2, 0))

if __name__ == "__main__":
    main_loop()
