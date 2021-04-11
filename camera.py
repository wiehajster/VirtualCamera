import pygame
import numpy as np
from datetime import datetime
pygame.init()

class Point:
    def __init__(self, coordinates):
        self.coordinates = np.array(coordinates + [1.0])
      
    def __str__(self):
        return f'x = {self.coordinates[0]}, y = {self.coordinates[1]}, z = {self.coordinates[2]}, {self.coordinates[3]}'

    def __repr__(self):
        return f'x = {self.coordinates[0]}, y = {self.coordinates[1]}, z = {self.coordinates[2]}, {self.coordinates[3]}'
    
    def multiply(self, matrix):
        self.coordinates = np.dot(matrix, self.coordinates)

    def normalize(self):
        k = 1/self.coordinates[3]
        self.coordinates *= k
        self.coordinates[3] = 1.0

class Point2D:
    def __init__(self, coordinates):
        self.coordinates = np.array(coordinates)

    def __str__(self):
        return f'x = {self.coordinates[0]}, y = {self.coordinates[1]}'

    def __repr__(self):
        return f'x = {self.coordinates[0]}, y = {self.coordinates[1]}'

class Line:
    def __init__(self, points):
        self.points = points

    def __str__(self):
        return f'p1 = ({self.points[0]}), p2 = ({self.points[1]})\n'

    def __repr__(self):
        return f'p1 = ({self.points[0]}), p2 = ({self.points[1]})\n'


def load_coordinates(filename):
    with open(filename, 'r') as f:
        lines_txt = f.read().split('\n\n')
        lines = []
        for line_txt in lines_txt:
            points_txt = line_txt.split('\n')
            points = []
            for point_txt in points_txt:
                coordinates = point_txt.split(' ')
                coordinates = [float(c) for c in coordinates]
                p = Point(coordinates)
                points.append(p)
            
            line = Line(points)
            lines.append(line)
    return lines

def translation(x, y, z):
    matrix = np.array([[1, 0, 0, x],
                       [0, 1, 0, y],
                       [0, 0, 1, z],
                       [0, 0, 0, 1]])
    return matrix

def zoom(sx, sy, sz):
    matrix = np.array([[sx, 0, 0, 0],
                       [0, sy, 0, 0],
                       [0, 0, sz, 0],
                       [0, 0, 0, 1]])
    return matrix

def rotationOX(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)

    matrix = np.array([[1, 0, 0, 0],
                       [0, cos, -sin, 0],
                       [0, sin, cos, 0],
                       [0, 0, 0, 1]])
    return matrix

def rotationOY(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)

    matrix = np.array([[cos, 0, sin, 0],
                       [0, 1, 0, 0],
                       [-sin, 0, cos, 0],
                       [0, 0, 0, 1]])
    return matrix

def rotationOZ(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)

    matrix = np.array([[cos, -sin, 0, 0],
                       [sin, cos, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
    return matrix

def transform(lines, matrix):
    for i in range(len(lines)):
        for j in range(len(lines[i].points)):
            lines[i].points[j].multiply(matrix)
            #lines[i].points[j].normalize()

def is_visible(point):
    if point.coordinates[2] >= d:
        return True
    return False

def project_point(point):
    k = d/point.coordinates[2]
    x = k * point.coordinates[0] + size/2
    y = size/2 - k * point.coordinates[1]

    return Point2D([x, y])

def project_normally(p1, p2):
    pro_p1 = project_point(p1)
    pro_p2 = project_point(p2)

    return Line([pro_p1, pro_p2])

def project_cut(p1, p2):
    if p1.coordinates[2] == d:
        x = p1.coordinates[0] + size/2
        y = size/2 - p1.coordinates[1]

        pro_p1 = Point2D([x, y])
        pro_p2 = Point2D([x, y])

        return Line([pro_p1, pro_p2])

    else:
        Z = p1.coordinates[2] - p2.coordinates[2]
        z = p1.coordinates[2] - d
        k = z/Z
        x = p1.coordinates[0] + ((p2.coordinates[0] - p1.coordinates[0]) * k)
        y = p1.coordinates[1] + ((p2.coordinates[1] - p1.coordinates[1]) * k)
        x += size/2
        y = size/2 - y

        pro_p1 = project_point(p1)
        pro_p2 = Point2D([x, y])

        return Line([pro_p1, pro_p2])


def project(lines):
    projections = []

    for line in lines:
        p1 = line.points[0]
        p2 = line.points[1]

        if is_visible(p1) and is_visible(p2):
            line = project_normally(p1, p2)
            projections.append(line)

        elif is_visible(p1) and not is_visible(p2):
            line = project_cut(p1, p2)
            projections.append(line)

        elif not is_visible(p1) and is_visible(p2):
            line = project_cut(p2, p1)
            projections.append(line)
        
    return projections 

def draw(projections):
    for i,line in enumerate(projections):
        p1 = line.points[0]
        p2 = line.points[1]
        pygame.draw.line(screen, RED, p1.coordinates, p2.coordinates)

d = 150
size = 500
t_step = 10
r_step = np.pi*5/180
d_step = 10
directory = 'E:/Semestr 6/Grafika komputerowa/obrazki/'
RED = pygame.Color(255, 0, 0)
'''
colors = np.random.choice(range(256), size=(4,3))
colors = [tuple(color) for color in colors]
GREEN = pygame.Color(0, 255, 0)
BLUE = pygame.Color(0, 0, 255)
BLACK = pygame.Color(0, 0, 0)
colors = [RED, GREEN, BLUE, BLACK]
'''
lines = load_coordinates('example_coordinates1_output.txt')
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
                pygame.image.save(screen, directory+name)

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
    objects = transform(lines, matrix)

    screen.fill((255, 255, 255))

    projections = project(lines)
    draw(projections)

    pygame.display.flip()

    clock.tick(10)

pygame.quit()