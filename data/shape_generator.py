'''
Generate shapes
'''
import numpy as np
from .noise_generator import NoiseGenerator

class Shape_Generator():
    def __init__(self, dim=16, noise_generator:NoiseGenerator=None, noise_properties:dict = {}, blank_ratio:float = 0.5, min_size:float = 0.25):
        self.dim = dim
        if noise_generator is None:
            self.noise_generator = NoiseGenerator(dim=dim, **noise_properties)
        else:
            self.noise_generator = noise_generator
        assert blank_ratio >= 0 and blank_ratio <= 1
        self.blank_ratio = blank_ratio
        assert min_size >= 0 and min_size < 1
        self.min_size = dim**2 * min_size
    
    def _blank_canvas(self):
        '''
        Generate a blank canvas to be filled
        '''
        return np.zeros((self.dim, self.dim))

    def generate(self, N:int):
        '''
        Generate list of data pairs (input, shape, label)

        input : np.array((dim, dim))
            Input for training
        
        shape : np.array((dim, dim))
            Target for training
        
        label : 1 or 0
            1 = shape present
            0 = no shape present
        
        N : int
            number of data
        '''
        data = []
        n_data = (1 - self.blank_ratio)*N
        n_random = self.blank_ratio*N

        for i, shape in enumerate([self._generate(i) for i in range(n_data)]):
            if i % 100 == 0:
                print('Adding noise of {}th shape'.format(i))
            data.append(
                self.noise_generator.randomize_image(shape).reshape((self.dim, self.dim, 1)),
                shape.reshape((self.dim, self.dim)),
                [1]
            )
        
        for i, shape in enumerate([self._blank_canvas() for i in range(n_random)]):
            if i % 100 == 0:
                print('Adding noise of {}th empty canvas'.format(i))
            data.append(
                self.noise_generator.randomize_image(shape).reshape((self.dim, self.dim, 1)),
                shape.reshape((self.dim, self.dim)),
                [0]
            )
        
        data = np.random.permutation(data)

        return zip(*data)


        
    def _generate(self, i:int = 1):
        '''
        Create a shape

        i : int
            used for debugging/monitoring status
        '''
        pass

