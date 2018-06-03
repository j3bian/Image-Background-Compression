'''
Generate Triangles
'''

import numpy as np
from .shape_generator import Shape_Generator

class Triangle_Generator(Shape_Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _generate(self, i:int):
        '''
        Generate a triangluar shape satisfying self.min_size constraint
        '''
        shape = self._blank_canvas()

        while np.sum(shape) < self.min_size:
            shape = self._blank_canvas()
            # generate random triplet
            # Use Delaunay Triangulation 
# i love susan so much omg omgo omg omgo mgom