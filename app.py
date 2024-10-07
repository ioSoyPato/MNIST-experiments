import pygame
import predictModel
import numpy as np
from PIL import Image

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
                    # Procesar imagen directamente sin guardar
                    cropped_img = crope(screen)
                    img_array = pygame.surfarray.array3d(cropped_img)
                    img_array = preprocess_image(img_array)
                    predicted_digit = predictModel.predictedImageArray(img_array)
                    print(f"Predicted Digit: {predicted_digit}")
                    # Mostrar el resultado en pantalla
                    screen.fill(WHITE, (WIDTH + 20, HEIGHT // 2 - 100, WIDTH - 40, 200))
                    font = pygame.font.SysFont("comicsansms", 72)
                    text_surface = font.render(str(predicted_digit), True, RED)
                    screen.blit(text_surface, (WIDTH + 20, HEIGHT // 2))
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

def preprocess_image(img_array):
    # Convertir imagen a escala de grises, cambiar tamaño a 28x28 y normalizar
    img = Image.fromarray(np.uint8(img_array)).convert('L').resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

if __name__ == "__main__":
    main_loop()
