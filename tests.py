def test_0(polygon_s, polygon_p):
    """Jak funkcja zwróci True to jest depth overlap"""
    # s_zmin = min([polygon_s.get_points_coords()[0][2], polygon_s.get_points_coords()[1][2],
    #               polygon_s.get_points_coords()[2][2],
    #               polygon_s.get_points_coords()[3][2]])
    # p_zmax = max([polygon_p.get_points_coords()[0][2], polygon_p.get_points_coords()[1][2],
    #               polygon_p.get_points_coords()[2][2],
    #               polygon_p.get_points_coords()[3][2]])

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
    # s_xmax = max([polygon_s.get_points_coords()[0][0], polygon_s.get_points_coords()[1][0],
    #               polygon_s.get_points_coords()[2][0],
    #               polygon_s.get_points_coords()[3][0]])
    # s_xmin = min([polygon_s.get_points_coords()[0][0], polygon_s.get_points_coords()[1][0],
    #               polygon_s.get_points_coords()[2][0],
    #               polygon_s.get_points_coords()[3][0]])
    #
    # s_ymax = max([polygon_s.get_points_coords()[0][1], polygon_s.get_points_coords()[1][1],
    #               polygon_s.get_points_coords()[2][1],
    #               polygon_s.get_points_coords()[3][1]])
    # s_ymin = min([polygon_s.get_points_coords()[0][1], polygon_s.get_points_coords()[1][1],
    #               polygon_s.get_points_coords()[2][1],
    #               polygon_s.get_points_coords()[3][1]])
    #
    # s_zmax = max([polygon_s.get_points_coords()[0][2], polygon_s.get_points_coords()[1][2],
    #               polygon_s.get_points_coords()[2][2],
    #               polygon_s.get_points_coords()[3][2]])
    # s_zmin = min([polygon_s.get_points_coords()[0][2], polygon_s.get_points_coords()[1][2],
    #               polygon_s.get_points_coords()[2][2],
    #               polygon_s.get_points_coords()[3][2]])
    #
    # p_xmax = max([polygon_p.get_points_coords()[0][0], polygon_p.get_points_coords()[1][0],
    #               polygon_p.get_points_coords()[2][0],
    #               polygon_p.get_points_coords()[3][0]])
    # p_xmin = min([polygon_p.get_points_coords()[0][0], polygon_p.get_points_coords()[1][0],
    #               polygon_p.get_points_coords()[2][0],
    #               polygon_p.get_points_coords()[3][0]])
    #
    # p_ymax = max([polygon_p.get_points_coords()[0][1], polygon_p.get_points_coords()[1][1],
    #               polygon_p.get_points_coords()[2][1],
    #               polygon_p.get_points_coords()[3][1]])
    # p_ymin = min([polygon_p.get_points_coords()[0][1], polygon_p.get_points_coords()[1][1],
    #               polygon_p.get_points_coords()[2][1],
    #               polygon_p.get_points_coords()[3][1]])
    #
    # p_zmax = max([polygon_p.get_points_coords()[0][2], polygon_p.get_points_coords()[1][2],
    #               polygon_p.get_points_coords()[2][2],
    #               polygon_p.get_points_coords()[3][2]])
    # p_zmin = min([polygon_p.get_points_coords()[0][2], polygon_p.get_points_coords()[1][2],
    #               polygon_p.get_points_coords()[2][2],
    #               polygon_p.get_points_coords()[3][2]])

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

    if (s_xmax <= p_xmin or p_xmax <= s_xmin or s_zmax <= p_zmin or p_zmax <= s_zmin) and (
            s_xmax <= p_xmin or p_xmax <= s_xmin or s_ymax <= p_ymin or p_ymax <= s_ymin):
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
        return 3

    return -1

# def test_1_1(polygon_s, polygon_p):
#     """Jak funkcja zwróci True to bounding boxes nie nachodzą na siebie"""
#     s_xmax = max([polygon_s.get_points_coords()[0][0], polygon_s.get_points_coords()[1][0],
#                   polygon_s.get_points_coords()[2][0],
#                   polygon_s.get_points_coords()[3][0]])
#     s_xmin = min([polygon_s.get_points_coords()[0][0], polygon_s.get_points_coords()[1][0],
#                   polygon_s.get_points_coords()[2][0],
#                   polygon_s.get_points_coords()[3][0]])
#
#     s_ymax = max([polygon_s.get_points_coords()[0][1], polygon_s.get_points_coords()[1][1],
#                   polygon_s.get_points_coords()[2][1],
#                   polygon_s.get_points_coords()[3][1]])
#     s_ymin = min([polygon_s.get_points_coords()[0][1], polygon_s.get_points_coords()[1][1],
#                   polygon_s.get_points_coords()[2][1],
#                   polygon_s.get_points_coords()[3][1]])
#
#     s_zmax = max([polygon_s.get_points_coords()[0][2], polygon_s.get_points_coords()[1][2],
#                   polygon_s.get_points_coords()[2][2],
#                   polygon_s.get_points_coords()[3][2]])
#     s_zmin = min([polygon_s.get_points_coords()[0][2], polygon_s.get_points_coords()[1][2],
#                   polygon_s.get_points_coords()[2][2],
#                   polygon_s.get_points_coords()[3][2]])
#
#     p_xmax = max([polygon_p.get_points_coords()[0][0], polygon_p.get_points_coords()[1][0],
#                   polygon_p.get_points_coords()[2][0],
#                   polygon_p.get_points_coords()[3][0]])
#     p_xmin = min([polygon_p.get_points_coords()[0][0], polygon_p.get_points_coords()[1][0],
#                   polygon_p.get_points_coords()[2][0],
#                   polygon_p.get_points_coords()[3][0]])
#
#     p_ymax = max([polygon_p.get_points_coords()[0][1], polygon_p.get_points_coords()[1][1],
#                   polygon_p.get_points_coords()[2][1],
#                   polygon_p.get_points_coords()[3][1]])
#     p_ymin = min([polygon_p.get_points_coords()[0][1], polygon_p.get_points_coords()[1][1],
#                   polygon_p.get_points_coords()[2][1],
#                   polygon_p.get_points_coords()[3][1]])
#
#     p_zmax = max([polygon_p.get_points_coords()[0][2], polygon_p.get_points_coords()[1][2],
#                   polygon_p.get_points_coords()[2][2],
#                   polygon_p.get_points_coords()[3][2]])
#     p_zmin = min([polygon_p.get_points_coords()[0][2], polygon_p.get_points_coords()[1][2],
#                   polygon_p.get_points_coords()[2][2],
#                   polygon_p.get_points_coords()[3][2]])
#
#     if (s_xmax <= p_xmin or p_xmax <= s_xmin or s_zmax <= p_zmin or p_zmax <= s_zmin) and (
#             s_xmax <= p_xmin or p_xmax <= s_xmin or s_ymax <= p_ymin or p_ymax <= s_ymin):
#         return True  # passed Test 1_1 - nie wykonujemy kolejnych testów
#
#     return False  # failed Test 1_1 - wykonujemy kolejne testy
#
#
# def test_1_2(polygon_s, polygon_p):
#     """Jak funkcja zwróci True to bounding boxes są jedno w drugim"""
#     s_xmax = max([polygon_s.get_points_coords()[0][0], polygon_s.get_points_coords()[1][0],
#                   polygon_s.get_points_coords()[2][0],
#                   polygon_s.get_points_coords()[3][0]])
#     s_xmin = min([polygon_s.get_points_coords()[0][0], polygon_s.get_points_coords()[1][0],
#                   polygon_s.get_points_coords()[2][0],
#                   polygon_s.get_points_coords()[3][0]])
#
#     s_ymax = max([polygon_s.get_points_coords()[0][1], polygon_s.get_points_coords()[1][1],
#                   polygon_s.get_points_coords()[2][1],
#                   polygon_s.get_points_coords()[3][1]])
#     s_ymin = min([polygon_s.get_points_coords()[0][1], polygon_s.get_points_coords()[1][1],
#                   polygon_s.get_points_coords()[2][1],
#                   polygon_s.get_points_coords()[3][1]])
#
#     s_zmax = max([polygon_s.get_points_coords()[0][2], polygon_s.get_points_coords()[1][2],
#                   polygon_s.get_points_coords()[2][2],
#                   polygon_s.get_points_coords()[3][2]])
#     s_zmin = min([polygon_s.get_points_coords()[0][2], polygon_s.get_points_coords()[1][2],
#                   polygon_s.get_points_coords()[2][2],
#                   polygon_s.get_points_coords()[3][2]])
#
#     p_xmax = max([polygon_p.get_points_coords()[0][0], polygon_p.get_points_coords()[1][0],
#                   polygon_p.get_points_coords()[2][0],
#                   polygon_p.get_points_coords()[3][0]])
#     p_xmin = min([polygon_p.get_points_coords()[0][0], polygon_p.get_points_coords()[1][0],
#                   polygon_p.get_points_coords()[2][0],
#                   polygon_p.get_points_coords()[3][0]])
#
#     p_ymax = max([polygon_p.get_points_coords()[0][1], polygon_p.get_points_coords()[1][1],
#                   polygon_p.get_points_coords()[2][1],
#                   polygon_p.get_points_coords()[3][1]])
#     p_ymin = min([polygon_p.get_points_coords()[0][1], polygon_p.get_points_coords()[1][1],
#                   polygon_p.get_points_coords()[2][1],
#                   polygon_p.get_points_coords()[3][1]])
#
#     p_zmax = max([polygon_p.get_points_coords()[0][2], polygon_p.get_points_coords()[1][2],
#                   polygon_p.get_points_coords()[2][2],
#                   polygon_p.get_points_coords()[3][2]])
#     p_zmin = min([polygon_p.get_points_coords()[0][2], polygon_p.get_points_coords()[1][2],
#                   polygon_p.get_points_coords()[2][2],
#                   polygon_p.get_points_coords()[3][2]])
#
#     if ((s_xmin < p_xmin and p_xmax < s_xmax and s_zmin < p_zmin and p_zmax < s_zmax) or
#         (s_xmin < p_xmin and p_xmax < s_xmax and s_ymin < p_ymin and p_ymax < s_ymax)) or \
#             ((p_xmin < s_xmin and s_xmax < p_xmax and p_zmin < s_zmin and s_zmax < p_zmax) or
#              (p_xmin < s_xmin and s_xmax < p_xmax and p_ymin < s_ymin and s_ymax < p_ymax)):
#         return True  # passed Test 1_2 - nie wykonujemy kolejnych testów
#
#     return False  # failed Test 1_2 - wykonujemy kolejne testy
#
#
# def test_1_3(polygon_s, polygon_p):
#     s_xmax = max([polygon_s.get_points_coords()[0][0], polygon_s.get_points_coords()[1][0],
#                   polygon_s.get_points_coords()[2][0],
#                   polygon_s.get_points_coords()[3][0]])
#     s_xmin = min([polygon_s.get_points_coords()[0][0], polygon_s.get_points_coords()[1][0],
#                   polygon_s.get_points_coords()[2][0],
#                   polygon_s.get_points_coords()[3][0]])
#
#     s_ymax = max([polygon_s.get_points_coords()[0][1], polygon_s.get_points_coords()[1][1],
#                   polygon_s.get_points_coords()[2][1],
#                   polygon_s.get_points_coords()[3][1]])
#     s_ymin = min([polygon_s.get_points_coords()[0][1], polygon_s.get_points_coords()[1][1],
#                   polygon_s.get_points_coords()[2][1],
#                   polygon_s.get_points_coords()[3][1]])
#
#     s_zmax = max([polygon_s.get_points_coords()[0][2], polygon_s.get_points_coords()[1][2],
#                   polygon_s.get_points_coords()[2][2],
#                   polygon_s.get_points_coords()[3][2]])
#     s_zmin = min([polygon_s.get_points_coords()[0][2], polygon_s.get_points_coords()[1][2],
#                   polygon_s.get_points_coords()[2][2],
#                   polygon_s.get_points_coords()[3][2]])
#
#     p_xmax = max([polygon_p.get_points_coords()[0][0], polygon_p.get_points_coords()[1][0],
#                   polygon_p.get_points_coords()[2][0],
#                   polygon_p.get_points_coords()[3][0]])
#     p_xmin = min([polygon_p.get_points_coords()[0][0], polygon_p.get_points_coords()[1][0],
#                   polygon_p.get_points_coords()[2][0],
#                   polygon_p.get_points_coords()[3][0]])
#
#     p_ymax = max([polygon_p.get_points_coords()[0][1], polygon_p.get_points_coords()[1][1],
#                   polygon_p.get_points_coords()[2][1],
#                   polygon_p.get_points_coords()[3][1]])
#     p_ymin = min([polygon_p.get_points_coords()[0][1], polygon_p.get_points_coords()[1][1],
#                   polygon_p.get_points_coords()[2][1],
#                   polygon_p.get_points_coords()[3][1]])
#
#     p_zmax = max([polygon_p.get_points_coords()[0][2], polygon_p.get_points_coords()[1][2],
#                   polygon_p.get_points_coords()[2][2],
#                   polygon_p.get_points_coords()[3][2]])
#     p_zmin = min([polygon_p.get_points_coords()[0][2], polygon_p.get_points_coords()[1][2],
#                   polygon_p.get_points_coords()[2][2],
#                   polygon_p.get_points_coords()[3][2]])
#
#     if ((s_xmin < p_xmin < s_xmax < p_xmax and s_zmin < p_zmin and p_zmax < s_zmax) or
#         (s_xmin < p_xmin < s_xmax < p_xmax and s_ymin < p_ymin and p_ymax < s_ymax)) or \
#             ((p_xmin < s_xmin < p_xmax < s_xmax and p_zmin < s_zmin and s_zmax < p_zmax) or
#              (p_xmin < s_xmin < p_xmax < s_xmax and p_ymin < s_ymin and s_ymax < p_ymax)):
#         return True
#
#     return False
