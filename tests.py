from shapely.geometry import Polygon as ShapelyPolygon
import numpy as np

from camera import get_plane_equation


def test_0(polygon_s, polygon_p):
    s_z_list = []

    p_z_list = []

    for point in polygon_s.get_points_coords():
        s_z_list.append(point[2])

    for point in polygon_p.get_points_coords():
        p_z_list.append(point[2])

    s_zmin = min(s_z_list)
    p_zmax = max(p_z_list)

    if s_zmin > p_zmax:
        return False  # passed Test 0 - nie wykonujemy kolejnych testów
    else:
        return True  # failed Test 0 - wykonujemy kolejne testy


def test_1(polygon_s, polygon_p):
    s_x_list = []
    s_y_list = []
    s_z_list = []

    p_x_list = []
    p_y_list = []
    p_z_list = []

    for point in polygon_s.get_points_coords():
        s_x_list.append(point[0])
        s_y_list.append(point[1])
        s_z_list.append(point[2])

    for point in polygon_p.get_points_coords():
        p_x_list.append(point[0])
        p_y_list.append(point[1])
        p_z_list.append(point[2])

    s_xmax = max(s_x_list)
    s_xmin = min(s_x_list)

    s_ymax = max(s_y_list)
    s_ymin = min(s_y_list)

    s_zmax = max(s_z_list)
    s_zmin = min(s_z_list)

    p_xmax = max(p_x_list)
    p_xmin = min(p_x_list)

    p_ymax = max(p_y_list)
    p_ymin = min(p_y_list)

    p_zmax = max(p_z_list)
    p_zmin = min(p_z_list)

    if ((s_xmax <= p_xmin or p_xmax <= s_xmin or s_zmax <= p_zmin or p_zmax <= s_zmin) and (
            s_xmax <= p_xmin or p_xmax <= s_xmin or s_ymax <= p_ymin or p_ymax <= s_ymin)) or \
            (p_xmax <= s_xmin or s_xmax <= p_xmin or p_zmax <= s_zmin or s_zmax <= p_zmin) and (
            p_xmax <= s_xmin or s_xmax <= p_xmin or p_ymax <= s_ymin or s_ymax <= p_ymin):
        return 1  # passed Test 1_1 - nie wykonujemy kolejnych testów

    if ((s_xmin < p_xmin and p_xmax < s_xmax and s_zmin < p_zmin and p_zmax < s_zmax) or
        (s_xmin < p_xmin and p_xmax < s_xmax and s_ymin < p_ymin and p_ymax < s_ymax)) or \
            ((p_xmin < s_xmin and s_xmax < p_xmax and p_zmin < s_zmin and s_zmax < p_zmax) or
             (p_xmin < s_xmin and s_xmax < p_xmax and p_ymin < s_ymin and s_ymax < p_ymax)):
        return 2  # passed Test 1_2 - nie wykonujemy kolejnych testów

    if ((s_xmin < p_xmin < s_xmax < p_xmax and s_zmin < p_zmin and p_zmax < s_zmax) or
        (s_xmin < p_xmin < s_xmax < p_xmax and s_ymin < p_ymin and p_ymax < s_ymax)) or \
            ((p_xmin < s_xmin < p_xmax < s_xmax and p_zmin < s_zmin and s_zmax < p_zmax) or
             (p_xmin < s_xmin < p_xmax < s_xmax and p_ymin < s_ymin and s_ymax < p_ymax)):
        return 3  # wykonujemy kolejne testy

    return -1


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


def test_4(polygon_s, polygon_p):
    """Funkcja zwraca True jeżeli ściany nie przecinają się"""
    polygon_s_points = polygon_s.get_points_coords()
    polygon_p_points = polygon_p.get_points_coords()
    polygon_s_xy = list(map(lambda p: (p[0], p[1]), polygon_s_points))
    polygon_p_xy = list(map(lambda p: (p[0], p[1]), polygon_p_points))
    shapely_polygon_s = ShapelyPolygon(polygon_s_xy)
    shapely_polygon_p = ShapelyPolygon(polygon_p_xy)

    intersection = shapely_polygon_s.intersection(shapely_polygon_p)

    if intersection.is_empty:
        return True  # ściany nie przecinają się

    return False
