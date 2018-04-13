'''
Final Project - Winter 2018
Written by: Mike (JingHongYu) Bian

Load data for training
'''

import pickle
from random import randint
from .noise_generator import NoiseGenerator
import numpy as np

def get_data(source:str, dim:int = 16, noise_properties:dict = {}):
    '''
    gets a dictionary of data using ShapeLoader from a scource

    Inputs:
        source = path to pickled file shapes
        dim = size of the images
        noise_properties = properties for the NoiseGenerator 
    Returns:
        dictionary of training and test data
    '''

    SL = ShapeLoader(dim, None, noise_properties)
    x, y, l = SL.load(source)
    n_samples = len(x)

    x_test = x[:int(n_samples/10)]
    y_test = y[:int(n_samples/10)]
    l_test = l[:int(n_samples/10)]
    x_train = x[int(n_samples/10):]
    y_train = y[int(n_samples/10):]
    l_train = l[int(n_samples/10):]

    return {
        'x_train' : np.array(x_train),
        'y_train' : np.array(y_train),
        'l_train' : np.array(l_train),
        'x_test' : np.array(x_test),
        'y_test' : np.array(y_test),
        'l_test' : np.array(l_test)
    }

class ShapeLoader:
    '''
    Loads shapes from file and adds noise to the samples
    '''

    def __init__(self, dim=16, noise_generator:NoiseGenerator=None, noise_properties:dict = {}):
        self.dim = dim
        if noise_generator is None:
            self.noise_generator = NoiseGenerator(dim=dim, **noise_properties)
        else:
            self.noise_generator = noise_generator

    def load(self, path:str):
        '''
        Load all shapes from path. The file at path must be a pickled file containing an iterable of
        2D arrays (images) of size (dim * dim) containing 1 shape (object of interest)

        Generates an even number of filled and empty images

        Inputs:
            path = path to pickled file shapes
        Returns:
            iterable of 3 Tuples:
                image
                target
                label (contains shape or no shape)
        '''
        with open(path, mode='rb') as f:
            shapes = pickle.load(f)
        data = [self.generate_data(shape, i) for i, shape in enumerate(shapes)] + [self.empty_shape(i) for i in range(len(shapes))]
        data = np.random.permutation(data)
        return zip(*data)

    def generate_data(self, shape, i:int):
        '''
        Generates a tuple for a given shape (image)

        Inputs:
            shape = 2D array of an image with 1 object within it
            i = i'th image tuple generated (for monitoring)
        Returns:
            3-Tuple of :
                Image with added noise
                Target (shape)
                Label 1 (contains object)
        '''
        if i % 100 == 0:
            print('Loading {}th image'.format(i))

        label = shape.copy()
        canvas = shape * randint(1, 255)
        # label = canvas.clip(0, 1)
        
        canvas = self.noise_generator.randomize_image(canvas)

        return canvas.reshape((self.dim, self.dim, 1)), label.reshape((self.dim, self.dim, 1)), [1]

    def empty_shape(self, i:int):
        '''
        Generate a tuple for a empty image

        Inputs:
            i = i'th random tuple generated (for monitoring)
        Returns:
            3-Tuple of :
                Image with added noise
                Empty target (zeros)
                Label 0 (no object)
        '''
        if i % 100 == 0:
            print('Generate {}th random'.format(i))
        
        canvas = self.noise_generator.randomize_image(np.zeros((self.dim, self.dim)))

        return canvas.reshape((self.dim, self.dim, 1)), np.zeros((self.dim, self.dim, 1)), [0]
