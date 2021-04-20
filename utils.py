import numpy as np
import pygame
from shapely.geometry import Polygon as SPolygon, MultiPolygon

# from collections import OrderedDict
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import scipy as sp
# import matplotlib.colors as colors

from structs import Point, Point2D, Line, Polygon


def convert2PolygonList(polygon):
    if isinstance(polygon, MultiPolygon):
        return list(polygon)
    return [polygon]


def compute_z(point, coeffs):
    A, B, C, D = coeffs
    z = (np.dot(point, [A, B]) + D) / -C
    return z


def cut_polygon(polygon1, polygon2):
    pol1_points = polygon1.get_points_coords()
    pol2_points = polygon2.get_points_coords()

    pol1_xy = list(map(lambda p: (p[0], p[1]), pol1_points))
    pol2_xy = list(map(lambda p: (p[0], p[1]), pol2_points))

    pol1 = SPolygon(pol1_xy)
    pol2 = SPolygon(pol2_xy)

    intersection = pol1.intersection(pol2)
    nonoverlap = pol1.difference(intersection)

    polygons = list(map(convert2PolygonList, [intersection, nonoverlap]))

    coeffs = get_plane_equation(polygon1)

    new_polygons = []
    for polygon in polygons:
        for pol in polygon:
            print(pol)
            pol = [Point([c1, c2, compute_z([c1, c2], coeffs)]) for c1, c2 in
                   zip(pol.exterior.xy[0], pol.exterior.xy[1])][:-1]

            lines = []
            for i in range(-1, len(pol) - 1, 1):
                line_points = [pol[i], pol[i + 1]]
                line = Line(line_points)
                lines.append(line)

            pol = Polygon(pol, lines, polygon1.color)
            new_polygons.append(pol)

    return new_polygons


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
            color = np.random.randint(low=0, high=256, size=(3))
            color = tuple(color)
            polygon = Polygon(points, lines, color=color)
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


def is_visible(point, d):
    if point.coordinates[2] >= d:
        return True
    return False


def project_point(point, d, size):
    k = d / point.coordinates[2]
    x = k * point.coordinates[0] + size / 2
    y = size / 2 - k * point.coordinates[1]

    return Point2D([x, y])


def project_normally(p1, p2, d, size):
    pro_p1 = project_point(p1, d, size)
    pro_p2 = project_point(p2, d, size)

    return Line([pro_p1, pro_p2])


def project_cut(p1, p2, d, size):
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

        pro_p1 = project_point(p1, d, size)
        pro_p2 = Point2D([x, y])

        return Line([pro_p1, pro_p2])


def project(lines, d, size):
    projections = []

    for line in lines:
        p1 = line.points[0]
        p2 = line.points[1]

        if is_visible(p1, d) and is_visible(p2, d):
            line = project_normally(p1, p2, d, size)
            projections.append(line)

        elif is_visible(p1, d) and not is_visible(p2, d):
            line = project_cut(p1, p2, d, size)
            projections.append(line)

        elif not is_visible(p1, d) and is_visible(p2, d):
            line = project_cut(p2, p1, d, size)
            projections.append(line)

    return projections


def draw(polygons, colors, screen):
    for polygon, color in zip(polygons, colors):
        pygame.draw.polygon(screen, color, polygon)


# def example_plane():
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     x = [0, 1, 1]
#     y = [0, 0, 1]
#     z = [0, 1, 0]
#     verts = [list(zip(x, y, z))]
#     ax.add_collection3d(Poly3DCollection(verts), zs=z)
#     plt.show()


def get_plane_equation(polygon):
    points = polygon.get_points_coords()
    p1, p2, p3 = points[:3]
    p1p2 = p2 - p1
    p1p3 = p3 - p1
    cross = np.cross(p1p2, p1p3)
    D = np.dot(p1, cross)
    coeffs = np.array([cross[0], cross[1], cross[2], -D])
    return coeffs


# def plot_plane(coeffs, polygon):
#     a, b, c, d = coeffs
#
#     x = np.linspace(-10, 10, 10)
#     y = np.linspace(-10, 10, 10)
#
#     X, Y = np.meshgrid(x, y)
#     Z = (d - a * X - b * Y) / c
#
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     print(polygon)
#     surf = ax.plot_surface(X, Y, Z, color='r')
#     ax.add_collection3d(Poly3DCollection(polygon))
#     plt.show()


# def plot_polygons(polygons):
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     for polygon in polygons:
#         polygon = polygon.get_points_coords()
#         polygon = [list(polygon)]
#         tri = Poly3DCollection(polygon)
#         # tri.set_color(colors.rgb2hex(sp.rand(3)))
#         tri.set_edgecolor('k')
#         ax.add_collection3d(tri)
#     plt.show()
