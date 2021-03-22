def preprocess(filename):
    objects =  open(filename, 'r').read().split('\n\n')
    f = open(filename[:-4]+'_output.txt', 'w')
    text = ''
    for obj in objects:
        lines = obj.split('\n')
        for edge in edges:
            e0 = edge[0]
            e1 = edge[1]

            text += lines[e0] + '\n' + lines[e1] + '\n\n'
    f.write(text[:-2])
    f.close()



edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

preprocess('example_coordinates1.txt')