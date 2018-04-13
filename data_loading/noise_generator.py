'''
Final Project - Winter 2018
Written by: Mike (JingHongYu) Bian

Generates noise to the given image
'''

from random import randint, getrandbits, uniform
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def apply_layer(canvas, layer, mask=None, alpha=0.5):
    '''
    Applies layer onto canvas, only where mask allows
    '''
    assert canvas.shape == layer.shape
    if mask is None:
        mask = np.ones(canvas.shape)
    assert mask.shape == canvas.shape
    transparent = np.where(mask == 0)

    hybrid_layer = layer.copy()
    hybrid_layer[transparent] = canvas[transparent]

    return (1-alpha) * canvas + alpha * hybrid_layer

class NoiseGenerator:
    '''
    NoiseGenerator adds different types of noise to given canvas (2D array of images).
    All images must be greyscale (0..255).

    Types of noise:
        Gradients (shading)
        Blurs (Gaussian Blurs)
        Random noise
        Blocking (foreground covering)
        Background (random or plain background)
        Image inversion (color inversion)
    '''

    def __init__(self, dim=16,
                 grad1_alpha=0.5, grad2_alpha=0.5,
                 blur_radius=0,
                 noise_alpha=0,
                 foreground_objects=10, object_size=12,
                 random_background=0.1,
                 invert=True):
        '''
        Initialize with the noise properties.

        Inputs:
            dim = dimension of the images (images must be square)
            grad1_alpha = max gradient visibility on the original object(s) in on given image
            grad2_alpha = max gradient visibility on the whole image (after noise added)
            blur_radius = the blur radius for gaussian blurring
            noise_alpha = the max visibility of random noise
            foreground_objects = maximum foreground blocking canvas
            object_size = maximum size of blocking objects
            random_background = probability the background will be random (static noise)
            invert = randomly invert the image
        '''
        self.dim = dim

        self.grad1_alpha = grad1_alpha
        self.grad2_alpha = grad2_alpha

        self.blur_radius = blur_radius
        self.noise_alpha = noise_alpha
        self.object_size = object_size

        self.random_background = random_background
        self.invert = invert

        self.horizontal_grad = np.tile(np.arange(self.dim), (self.dim, 1)) / self.dim * 255
        self.vertical_grad = np.rot90(self.horizontal_grad)

        self.foreground_objects = foreground_objects


    def _generate_gradient(self, angle:float):
        '''
        Generates a gradient at an angle

        Inputs:
            angle = direction of gradient (radians)
        Returns:
            canvas with gradient at a given angle
        '''
        rotation = int(angle / np.pi * 2)
        small_angle = angle - rotation*np.pi/2
        return np.rot90(np.sin(small_angle)*self.horizontal_grad + np.cos(small_angle)*self.vertical_grad, rotation)

    def _generate_background(self, canvas):
        '''
        Generates a background where "0" is treated as transparent

        Input:
            canvas = 2D array of image
        Returns:
            canvas with a background behind the foreground objects
        '''
        if uniform(0, 1) < self.random_background:
            # Add noise to background
            bg = np.random.randint(0, 255, canvas.shape)
        else:
            bg = np.ones(canvas.shape) * randint(0, 255)

        canvas = apply_layer(bg, canvas, canvas, 1)
        return canvas

    def _get_region(self, size):
        '''
        Create an random region by expanding by tracing the path of a random search pattern

        Inputs:
            size = max size of region
        Returns:
            canvas with a single object
        '''
        a = np.zeros((size, size))
        start = (randint(0, size), randint(0, size))

        def new_pairs(location):
            return [(location, (i, j)) for i in range(-1, 1) for j in range(-1, 1)]

        queue = new_pairs(start)
        
        def inbounds(location):
            return min(location) >= 0 and max(location) < size

        object_size = randint(0, self.object_size)
        for _ in range(object_size):
            location, direction = queue.pop(randint(0, len(queue) - 1))
            new_loc = (location[0] + direction[0], location[1] + direction[1])
            while not inbounds(new_loc):
                location, direction = queue.pop(randint(0, len(queue) - 1))
                new_loc = (location[0] + direction[0], location[1] + direction[1])
            a[new_loc] = 1
            queue.extend(new_pairs(new_loc))
        return a

    def _generate_objects(self, canvas):
        '''
        Generate <= self.foreground_objects number of objects (regions) on the canvas

        Inputs:
            canvas = 2D array to to add objects on
        Outputs:
            canvas with objects overlaid
        '''
        num_objects = randint(0, self.foreground_objects)
        for _ in range(num_objects):
            region = self._get_region(canvas.shape[0])
            canvas[np.where(region == 1)] = randint(1, 255)
        return canvas


    def randomize_image(self, canvas):
        '''
        Add all initialized randomizations onto canvas

        Inputs:
            canvas = 2D array of unaltered image
        Returns:
            canvas of randomized image
        '''
        # Prepare Gradients
        grad1 = self._generate_gradient(uniform(0, 2*np.pi))
        grad2 = self._generate_gradient(uniform(0, 2*np.pi))

       
        grad1_alpha = uniform(0, self.grad1_alpha)
        grad2_alpha = uniform(0, self.grad2_alpha)

        # Apply gradient to objects
        canvas = apply_layer(canvas, grad1, canvas, grad1_alpha)
        
        # Add foreground objects
        canvas = self._generate_objects(canvas)
        canvas = self._generate_background(canvas)

        # Apply gradient over all canvas
        canvas = apply_layer(canvas, grad2, alpha=grad2_alpha)

        # Add random noise over canvas
        alpha = uniform(0, self.noise_alpha)
        canvas = (1-alpha) * canvas +  alpha * np.random.randint(0, 255, canvas.shape)

        canvas = canvas.clip(0, 254) # PIL gaussian blur bug

        # Blur
        img = Image.fromarray(canvas.astype('uint8'), mode='L')
        blurred = img.filter(ImageFilter.GaussianBlur(self.blur_radius*self.dim))
        canvas = np.array(blurred)

        # Randomly invert images
        if getrandbits(1) and self.invert:
            canvas = 255 - canvas

        return canvas / 255