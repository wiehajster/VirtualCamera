import pygame
import numpy as np
from datetime import datetime

from sorting import sort_polygons
from utils import load_coordinates, translation, rotationOX, rotationOY, rotationOZ, transform, project, \
    draw


if __name__ == '__main__':
    d = 2000
    size = 500
    t_step = 100
    r_step = np.pi * 1 / 180
    d_step = 10
    directory = 'E:/Semestr 6/Grafika komputerowa/obrazki/'
    RED = pygame.Color(255, 0, 0)

    polygons = load_coordinates('example_coordinates1_output.txt')
    # polygons = load_coordinates('overlap_test_2.txt')
    # plot_polygons(polygons)

    t = np.array([0.0, 0.0, 4005.0])
    t_matrix = translation(t[0], t[1], t[2])
    matrix = np.identity(4)
    matrix = np.dot(matrix, t_matrix)
    for polygon in polygons:
        transform(polygon.lines, matrix)

    pygame.init()
    screen = pygame.display.set_mode([size, size])
    clock = pygame.time.Clock()

    running = True

    while running:
        matrix = np.identity(4)
        t = np.array([0.0, 0.0, 0.0])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    name = 'screenshot' + '_' + date + '.png'
                    pygame.image.save(screen, directory + name)

        if pygame.key.get_pressed()[pygame.K_UP]:
            t[1] -= t_step
        if pygame.key.get_pressed()[pygame.K_DOWN]:
            t[1] += t_step
        if pygame.key.get_pressed()[pygame.K_a]:
            t[0] += t_step
        if pygame.key.get_pressed()[pygame.K_d]:
            t[0] -= t_step
        if pygame.key.get_pressed()[pygame.K_w]:
            t[2] -= t_step
        if pygame.key.get_pressed()[pygame.K_s]:
            t[2] += t_step
        if pygame.key.get_pressed()[pygame.K_COMMA]:
            d += d_step
        if pygame.key.get_pressed()[pygame.K_PERIOD]:
            if (d - d_step) > 0:
                d -= d_step
        if pygame.key.get_pressed()[pygame.K_z]:
            r_matrix = rotationOX(-r_step)
            matrix = np.dot(r_matrix, matrix)
        if pygame.key.get_pressed()[pygame.K_x]:
            r_matrix = rotationOX(r_step)
            matrix = np.dot(r_matrix, matrix)
        if pygame.key.get_pressed()[pygame.K_c]:
            r_matrix = rotationOY(r_step)
            matrix = np.dot(r_matrix, matrix)
        if pygame.key.get_pressed()[pygame.K_v]:
            r_matrix = rotationOY(-r_step)
            matrix = np.dot(r_matrix, matrix)
        if pygame.key.get_pressed()[pygame.K_b]:
            r_matrix = rotationOZ(-r_step)
            matrix = np.dot(r_matrix, matrix)
        if pygame.key.get_pressed()[pygame.K_n]:
            r_matrix = rotationOZ(r_step)
            matrix = np.dot(r_matrix, matrix)

        t_matrix = translation(t[0], t[1], t[2])
        matrix = np.dot(matrix, t_matrix)
        sorted_polygons = sort_polygons(polygons)
        projected_polygons = []
        colors = []

        for polygon in sorted_polygons:
            transform(polygon.lines, matrix)
            projected_lines = project(polygon.lines, d, size)
            projected_points = [line.points[0].coordinates for line in projected_lines]
            if len(projected_points) > 2:
                projected_polygons.append(projected_points)
                colors.append(polygon.color)

        screen.fill((255, 255, 255))
        draw(projected_polygons, colors, screen)
        pygame.display.flip()
        clock.tick(5)

    pygame.quit()
