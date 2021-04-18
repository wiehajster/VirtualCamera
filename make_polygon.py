def preprocess(filename):
    objects =  open(filename, 'r').read().split('\n\n')
    f = open(filename[:-4]+'_output.txt', 'w')
    text = ''
    for obj in objects:
        lines = obj.split('\n')
        for polygon in polygons:
            for point in polygon:
                text += lines[point] + '\n'
            text += '\n'    
    f.write(text[:-2])
    f.close()

polygons = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (3, 2, 6, 7), (3, 0, 4, 7)]

preprocess('example_coordinates1.txt')