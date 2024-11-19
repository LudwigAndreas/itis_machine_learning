import pygame
import time
import math
import random
import numpy as np
from collections import deque
from sklearn.cluster import DBSCAN as SklearnDBSCAN
import matplotlib.pyplot as plt


pygame.init()


screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Dot Painter")


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DOT_COLOR = (255, 0, 0)
RED = (255, 0, 0)
BACKGROUND_COLOR = (255, 255, 255)


DOT_RADIUS = 5
DOT_INTERVAL = 0.1
NEARBY_DOT_COUNT = 3
NEARBY_DOT_DISTANCE = 20
EPSILON = 30
MIN_SAMPLES = 5


def draw_dot(surface, position, color):
    pygame.draw.circle(surface, color, position, DOT_RADIUS)


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


def dbscan(dots, epsilon, min_samples):
    labels = [-1] * len(dots)
    visited = [False] * len(dots)
    cluster_id = 0

    def region_query(idx):
        neighbors = []
        for i, dot in enumerate(dots):
            if math.sqrt((dots[idx][0] - dot[0]) ** 2 + (dots[idx][1] - dot[1]) ** 2) < epsilon:
                neighbors.append(i)
        return neighbors

    def expand_cluster(idx, neighbors, cluster_id):
        labels[idx] = cluster_id
        queue = deque(neighbors)
        while queue:
            current_idx = queue.popleft()
            if not visited[current_idx]:
                visited[current_idx] = True
                new_neighbors = region_query(current_idx)
                if len(new_neighbors) >= min_samples:
                    queue.extend(new_neighbors)
            if labels[current_idx] == -1:
                labels[current_idx] = cluster_id

    for i in range(len(dots)):
        if not visited[i]:
            visited[i] = True
            neighbors = region_query(i)
            if len(neighbors) < min_samples:
                labels[i] = -1
            else:
                expand_cluster(i, neighbors, cluster_id)
                cluster_id += 1

    return labels


def sklearn_dbscan(dots, epsilon, min_samples):

    dots_array = np.array(dots)
    db = SklearnDBSCAN(eps=epsilon, min_samples=min_samples)
    labels = db.fit_predict(dots_array)
    return labels


def redraw_clusters(screen, labels, dots):
    cluster_colors = {}
    for i in range(len(labels)):
        cluster = labels[i]
        if cluster != -1:
            if cluster not in cluster_colors:
                cluster_colors[cluster] = (random.randint(0, 255),
                                           random.randint(0, 255),
                                           random.randint(0, 255))
            color = cluster_colors[cluster]
        else:
            color = RED
        draw_dot(screen, dots[i], color)


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
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:

                    if len(dots) >= MIN_SAMPLES:
                        labels = dbscan(dots, EPSILON, MIN_SAMPLES)
                        redraw_clusters(screen, labels, dots)

                elif event.key == pygame.K_e:

                    if len(dots) >= MIN_SAMPLES:
                        labels = sklearn_dbscan(dots, EPSILON, MIN_SAMPLES)
                        redraw_clusters(screen, labels, dots)

        if mouse_pressed:
            current_time = time.time()
            if current_time - last_dot_time >= DOT_INTERVAL:
                mouse_pos = pygame.mouse.get_pos()
                if is_new_dot_valid(mouse_pos, dots):
                    draw_dot(screen, mouse_pos, DOT_COLOR)
                    dots.append(mouse_pos)
                    last_dot_time = current_time

                    nearby_dots = generate_nearby_dots(
                        mouse_pos, NEARBY_DOT_COUNT, NEARBY_DOT_DISTANCE, dots)
                    for dot in nearby_dots:
                        draw_dot(screen, dot, DOT_COLOR)
                        dots.append(dot)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
