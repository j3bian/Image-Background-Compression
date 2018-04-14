import sys
import os
import pickle

import numpy as np

sys.path.append('data_generation/shapeset/shapeset2_1cspo_2_3.5000.test_code')

from image_config import *


if __name__ == '__main__':
    n = int(sys.argv[1])
    file_name = sys.argv[2]

    print_status = sys.argv[3] == 'True' if len(sys.argv) > 3 else False

    
    d = dataset(shapeset=2,seed=2,tag='test',free='color,orientation', fixed='size,position')
    d.data_directory = "."
    
    shapes = []
    
    last_percent = 0.0
    counter = 0.0

    while len(shapes) < n:
        scene = d.problem.generate()
        if counter/n - last_percent >= 0.1 and print_status:
            print('{:.2%}'.format(counter/n))
            last_percent = counter/n

        img = np.zeros((16, 16), dtype='uint8')
        img[np.where(scene.matrix(16, 16) != scene.color)] = 1
        shapes.append(img)
        counter += 1.0
        
    with open(file_name+str(n)+'.p', 'wb') as f:
        pickle.dump(shapes, f)

