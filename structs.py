import numpy as np


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
    def __init__(self, points, lines, color):
        self.points = points
        self.lines = lines
        self.color = color

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
