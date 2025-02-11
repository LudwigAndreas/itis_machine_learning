import pygame
import time
import math
import random


pygame.init()


screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Dot Painter")


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DOT_COLOR = (255, 0, 0)


DOT_RADIUS = 5
DOT_INTERVAL = 0.1
NEARBY_DOT_COUNT = 3
NEARBY_DOT_DISTANCE = 20


def draw_dot(surface, position):
    pygame.draw.circle(surface, DOT_COLOR, position, DOT_RADIUS)


def is_new_dot_valid(new_dot, existing_dots):
    for dot in existing_dots:
        distance = math.sqrt((new_dot[0] - dot[0])
                             ** 2 + (new_dot[1] - dot[1]) ** 2)
        if distance < DOT_RADIUS * 2:
            return False
    return True


def generate_nearby_dots(main_dot, count, max_distance, existing_dots):
    nearby_dots = []
    while len(nearby_dots) < count:

        offset_x = random.randint(-max_distance, max_distance)
        offset_y = random.randint(-max_distance, max_distance)
        new_dot = (main_dot[0] + offset_x, main_dot[1] + offset_y)

        if 0 <= new_dot[0] < screen_width and 0 <= new_dot[1] < screen_height:
            if is_new_dot_valid(new_dot, existing_dots):
                nearby_dots.append(new_dot)

    return nearby_dots


def main():
    dots = []
    running = True
    screen.fill(WHITE)
    last_dot_time = 0
    mouse_pressed = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pressed = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_pressed = False

        if mouse_pressed:
            current_time = time.time()
            if current_time - last_dot_time >= DOT_INTERVAL:
                mouse_pos = pygame.mouse.get_pos()
                if is_new_dot_valid(mouse_pos, dots):
                    draw_dot(screen, mouse_pos)
                    dots.append(mouse_pos)
                    last_dot_time = current_time

                    nearby_dots = generate_nearby_dots(
                        mouse_pos, NEARBY_DOT_COUNT, NEARBY_DOT_DISTANCE, dots)
                    for dot in nearby_dots:
                        draw_dot(screen, dot)
                        dots.append(dot)

        pygame.display.flip()

    pygame.quit()
    print(dots)


if __name__ == "__main__":
    main()
