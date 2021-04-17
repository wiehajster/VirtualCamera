import pygame
import numpy as np
from datetime import datetime
from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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
        k = 1 / self.coordinates[3]
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


class Polygon:
    def __init__(self, points, lines):
        self.points = points
        self.lines = lines

    def get_points_coords(self):
        points = [p.coordinates[:3] for p in self.points]
        return points

    def __str__(self):
        points_str = ''
        for i, point in enumerate(self.points):
            points_str += f'p{i} = ({point})\n'
        return points_str

    def __repr__(self):
        points_str = ''
        for i, point in enumerate(self.points):
            points_str += f'p{i} = ({point})\n'
        return points_str


def load_coordinates(filename):
    with open(filename, 'r') as f:
        polygons_txt = f.read().split('\n\n')
        polygons = []
        for polygon_txt in polygons_txt:
            points_txt = polygon_txt.split('\n')
            points = []
            for point_txt in points_txt:
                coordinates = point_txt.split(' ')
                coordinates = [float(c) for c in coordinates]
                p = Point(coordinates)
                points.append(p)

            lines = []
            for i in range(-1, len(points) - 1, 1):
                line_points = [points[i], points[i + 1]]
                line = Line(line_points)
                lines.append(line)
            polygon = Polygon(points, lines)
            polygons.append(polygon)
    return polygons


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
            # lines[i].points[j].normalize()


def is_visible(point):
    if point.coordinates[2] >= d:
        return True
    return False


def project_point(point):
    k = d / point.coordinates[2]
    x = k * point.coordinates[0] + size / 2
    y = size / 2 - k * point.coordinates[1]

    return Point2D([x, y])


def project_normally(p1, p2):
    pro_p1 = project_point(p1)
    pro_p2 = project_point(p2)

    return Line([pro_p1, pro_p2])


def project_cut(p1, p2):
    if p1.coordinates[2] == d:
        x = p1.coordinates[0] + size / 2
        y = size / 2 - p1.coordinates[1]

        pro_p1 = Point2D([x, y])
        pro_p2 = Point2D([x, y])

        return Line([pro_p1, pro_p2])

    else:
        Z = p1.coordinates[2] - p2.coordinates[2]
        z = p1.coordinates[2] - d
        k = z / Z
        x = p1.coordinates[0] + ((p2.coordinates[0] - p1.coordinates[0]) * k)
        y = p1.coordinates[1] + ((p2.coordinates[1] - p1.coordinates[1]) * k)
        x += size / 2
        y = size / 2 - y

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


'''
def draw(projections):
    for i,line in enumerate(projections):
        p1 = line.points[0]
        p2 = line.points[1]
        pygame.draw.line(screen, RED, p1.coordinates, p2.coordinates)
'''


def draw(polygons):
    for i, polygon in enumerate(polygons):
        pygame.draw.polygon(screen, RED, polygon)


def example_plane():
    fig = plt.figure()
    ax = Axes3D(fig)
    x = [0, 1, 1]
    y = [0, 0, 1]
    z = [0, 1, 0]
    verts = [list(zip(x, y, z))]
    ax.add_collection3d(Poly3DCollection(verts), zs=z)
    plt.show()


def get_plane_equation(polygon):
    points = polygon.get_points_coords()
    p1, p2, p3 = points[:3]
    p1p2 = p2 - p1
    p1p3 = p3 - p1
    cross = np.cross(p1p2, p1p3)
    D = np.dot(p1, cross)
    coeffs = np.array([cross[0], cross[1], cross[2], -D])
    return coeffs


def plot_plane(coeffs, polygon):
    a, b, c, d = coeffs

    x = np.linspace(-10, 10, 10)
    y = np.linspace(-10, 10, 10)

    X, Y = np.meshgrid(x, y)
    Z = (d - a * X - b * Y) / c

    fig = plt.figure()
    ax = Axes3D(fig)
    print(polygon)
    surf = ax.plot_surface(X, Y, Z, color='r')
    ax.add_collection3d(Poly3DCollection(polygon))
    plt.show()


# def test_01(polygons):
#     sorted_polygons = sorted(polygons, key=lambda x: max(
#         [x.get_points_coords()[0][2], x.get_points_coords()[1][2], x.get_points_coords()[2][2],
#          x.get_points_coords()[3][2]]), reverse=True)
#
#     reorder = False
#
#     s_zmax = max([sorted_polygons[0].get_points_coords()[0][2], sorted_polygons[0].get_points_coords()[1][2],
#                   sorted_polygons[0].get_points_coords()[2][2],
#                   sorted_polygons[0].get_points_coords()[3][2]])
#     s_zmin = min([sorted_polygons[0].get_points_coords()[0][2], sorted_polygons[0].get_points_coords()[1][2],
#                   sorted_polygons[0].get_points_coords()[2][2],
#                   sorted_polygons[0].get_points_coords()[3][2]])
#
#     s_xr = max([sorted_polygons[0].get_points_coords()[0][0], sorted_polygons[0].get_points_coords()[1][0],
#                 sorted_polygons[0].get_points_coords()[2][0],
#                 sorted_polygons[0].get_points_coords()[3][0]])
#     s_xl = min([sorted_polygons[0].get_points_coords()[0][0], sorted_polygons[0].get_points_coords()[1][0],
#                 sorted_polygons[0].get_points_coords()[2][0],
#                 sorted_polygons[0].get_points_coords()[3][0]])
#
#     s_yr = max([sorted_polygons[0].get_points_coords()[0][1], sorted_polygons[0].get_points_coords()[1][1],
#                 sorted_polygons[0].get_points_coords()[2][1],
#                 sorted_polygons[0].get_points_coords()[3][1]])
#     s_yl = min([sorted_polygons[0].get_points_coords()[0][1], sorted_polygons[0].get_points_coords()[1][1],
#                 sorted_polygons[0].get_points_coords()[2][1],
#                 sorted_polygons[0].get_points_coords()[3][1]])
#
#     for polygon in sorted_polygons[1:]:
#         p_zmax = max([polygon.get_points_coords()[0][2], polygon.get_points_coords()[1][2],
#                       polygon.get_points_coords()[2][2],
#                       polygon.get_points_coords()[3][2]])
#         p_zmin = min([polygon.get_points_coords()[0][2], polygon.get_points_coords()[1][2],
#                       polygon.get_points_coords()[2][2],
#                       polygon.get_points_coords()[3][2]])
#
#         p_xr = max([polygon.get_points_coords()[0][0], polygon.get_points_coords()[1][0],
#                     polygon.get_points_coords()[2][0],
#                     polygon.get_points_coords()[3][0]])
#         p_xl = min([polygon.get_points_coords()[0][0], polygon.get_points_coords()[1][0],
#                     polygon.get_points_coords()[2][0],
#                     polygon.get_points_coords()[3][0]])
#
#         p_yr = max([polygon.get_points_coords()[0][1], polygon.get_points_coords()[1][1],
#                     polygon.get_points_coords()[2][1],
#                     polygon.get_points_coords()[3][1]])
#         p_yl = min([polygon.get_points_coords()[0][1], polygon.get_points_coords()[1][1],
#                     polygon.get_points_coords()[2][1],
#                     polygon.get_points_coords()[3][1]])
#
#         if s_zmin <= p_zmax:  # idziemy do Testu 1
#             if (s_xr <= s_xl) or \
#                     ((p_zmax <= s_zmax and p_zmin >= s_zmin) and (p_xr <= s_xr and p_xl >= s_xl)):
#                 continue
#             elif (s_xl <= p_xl and s_xr >= p_xl and s_xr <= p_xr) or \
#                     (s_yl <= p_yl and s_yr >= p_yl and p_yr <= p_yr):  # idziemy do Testu 2
#                 pass
#         else:
#             continue


def test_23(polygon1, polygon2, outside):
    # wyznaczenie płaszczyzny P
    coeffs = get_plane_equation(polygon2)
    # sprawdzenie, czy S jest całkowicie z jednej lub drugiej strony płaszczyzny P
    polygon1_points = polygon1.get_points_coords()

    # narysowanie sytuacji w celu testowania
    # plot_plane(coeffs, polygon1_points)

    polygon1_points = np.array([np.append(p, 1.0) for p in polygon1_points])
    result = np.dot(polygon1_points, coeffs)
    result = np.where(result > 0) if outside == True else np.where(result < 0)

    if len(result[0]) == len(polygon1_points):
        return True
    return False


if __name__ == '__main__':
    d = 2500
    size = 500
    t_step = 20
    r_step = np.pi * 1 / 180
    d_step = 10
    directory = 'E:/Semestr 6/Grafika komputerowa/obrazki/'
    RED = pygame.Color(255, 0, 0)

    colors = np.random.choice(range(256), size=(24, 3))
    colors = [tuple(color) for color in colors]

    polygons = load_coordinates('example_coordinates1_output.txt')
    '''
    S = [[4, 8, 5], [-8, -7, 5], [16, 12, 5]]
    P = [[1, 3, 2], [2, 2, 1], [4, 2, 3]]
    S = [Point(p) for p in S]
    S = Polygon(S, [])
    P = [Point(p) for p in P]
    P = Polygon(P, [])
    print(S)
    print(P)
    print(test_23(S, P, True)) 
    '''

    pygame.init()

    t = np.array([0.0, 0.0, 6000.0])
    t_matrix = translation(t[0], t[1], t[2])
    matrix = np.identity(4)
    matrix = np.dot(matrix, t_matrix)
    for polygon in polygons:
        transform(polygon.lines, matrix)

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
        projected_polygons = []
        for polygon in polygons:
            transform(polygon.lines, matrix)
            projected_lines = project(polygon.lines)
            projected_points = []
            projected_points = [line.points[0].coordinates for line in projected_lines]
            if len(projected_points) > 2:
                projected_polygons.append(projected_points)

        screen.fill((255, 255, 255))
        draw(projected_polygons)

        pygame.display.flip()

        clock.tick(10)

    pygame.quit()
