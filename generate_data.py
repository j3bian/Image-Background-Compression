
from subprocess import call
from sys import argv, path

if __name__ == '__main__':
    if len(argv) == 1:
        call(['py',  '-2',  './data_generation/gen_shapes.py', '1000000', 'dataset'])
    elif len(argv) > 3:
        call(['py',  '-2',  './data_generation/gen_shapes.py', argv[1], argv[2], argv[3]])
    else:
        call(['py',  '-2',  './data_generation/gen_shapes.py', argv[1], argv[2]])
