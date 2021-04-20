from tests import test_0, test_1, test_23, test_4
from utils import cut_polygon


def sort_polygons(polygons):
    sorted_polygons = sorted(polygons, key=lambda polygon: max([point[2] for point in polygon.get_points_coords()]),
                             reverse=True)
    i = 0

    while i < len(sorted_polygons):
        j = i + 1
        while j < len(sorted_polygons):
            # print('\ni', i, 'j', j)
            S = sorted_polygons[i]
            P = sorted_polygons[j]
            # print('Polygon S', S)
            # print('Polygon P', P)

            # Test 0
            # False -> passed Test 0 - nie wykonujemy kolejnych testów
            # True  -> failed Test 0 - wykonujemy kolejne testy

            # print('Test 0', test_0(S, P))
            if test_0(S, P):
                j += 1
                continue

            # Test 1
            # 1 -> passed Test 1_1 - nie wykonujemy kolejnych testów
            # 2 -> passed Test 1_2 - nie wykonujemy kolejnych testów
            # 3 -> wykonujemy kolejne testy

            if not test_1(S, P):
                j += 1
                continue

            # Test 2
            # True -> jest polygon1 jest outside lub inside polygon2
            # False -> polygon1 nie jest outside lub inside polygon2
            # print('Test 2', test_23(S, P, outside=True))

            if test_23(S, P, outside=True):
                j += 1
                continue
            if test_23(P, S, outside=False):
                j += 1
                continue

            if test_23(P, S, outside=True):
                j += 1
                continue
            if test_23(S, P, outside=False):
                j += 1
                continue

            # Test 4
            # True -> ściany nie przecinają się
            # False -> ściany przecinają się

            if test_4(S, P):
                j += 1
                continue

            new_polygons = cut_polygon(S, P)
            new_polygons = sorted(new_polygons,
                                  key=lambda polygon: max([point[2] for point in polygon.get_points_coords()]),
                                  reverse=True)
            del sorted_polygons[i]
            sorted_polygons[i:i] = new_polygons

            j = j + len(new_polygons) - 1

            # Zamiana S i P miejscami i powtórka testów 2 i 3
            # print('Powtorzenie testów')
            # print('Test 2', test_23(P, S, outside=True))
            # print('Test 3', test_23(S, P, outside=False))
            # if test_23(P, S, outside=True) or test_23(S, P, outside=False):
            #     temp = S
            #     sorted_polygons[i] = P
            #     sorted_polygons[j] = S
            #     i -= 1
            #     break
            # else:
            #     # print('Cut polygon')
            #     new_polygons = cut_polygon(S, P)
            #     new_polygons = sorted(new_polygons, key=lambda polygon: max([point[2] for point in polygon.get_points_coords()]), reverse=True)
            #     # print(len(new_polygons), 'new polygons')
            #     # print(new_polygons)
            #     del sorted_polygons[i]
            #     sorted_polygons[i:i] = new_polygons
            #     j = j + len(new_polygons) - 1

        i += 1

    return sorted_polygons
