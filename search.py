'''
Final Project - Winter 2018
Written by: Mike (JingHongYu) Bian

Search algorithm for foreground
'''
import numpy as np
from PIL import Image, ImageFilter
from keras import Model
from matplotlib import pyplot as plt
from sys import getsizeof
import pickle
import bz2
import random

def open_image(path:str):
    '''
    Loads an image from path as a greyscale image

    Inputs:
        path = source of image
    Returns:
        grey = greyscale image
    '''
    img = Image.open(path)
    img.load()
    grey = img.convert(mode='L')
    return grey

class ForegroundSearch:
    '''
    Uses a set of encoder, decoder and discriminator to search for the foreground of images
    '''

    def __init__(self, encoder:Model, decoder:Model, discriminator:Model, step=4, noise_threshold = 0.5, mask_threshold = 0.2, displays=None):
        '''
        Initialize the searcher with the encoder, decoder and discriminator

        Inputs:
            encoder = shape encoder
            decoder = shape decoder
            discriminator = model for classifying shapes from non-shapes
            step = window stride step
            noise_threshold = discriminator threshold for classifying shapes
            mask_threshold = hard cut-off for decoder mask
            displays = list of debugging displays of algorithm during runtime
                Options:
                    original = original image
                    masked = red mask on foreground (intensity shows certainty of the algorithm)
                    foreground = background replaced with blue
                    mask = plot with only mask (with a legend from 0 to 1)
                    SNR = histogram signal to noise ratio determined by the discriminator (for determining noise_threshold)
                    mask_histogram = histogram of mask intensities (for determining mask_threshold)
        '''
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.input_size = encoder.input_shape[1]
        self.step = step
        self.noise_threshold = noise_threshold
        self.mask_threshold = mask_threshold
        if displays is None:
            self.displays = []
        elif displays == 'all':
            self.displays = ['original', 'masked', 'foreground', 'mask', 'SNR', 'mask_histogram', 'recreate']
        else:
            self.displays = displays
        self.SNRs = []
        self.mask_values = []
    
    def _create_tiles(self, img:Image.Image, window_size:int):
        '''
        Create (overlapping) tiles for input into the encoder. Each tile will be the of
        (size * size) but reshaped to be (input_size * input_size)

        Inputs:
            img = Image to slice into tiles
            size = size of each tile
        Returns:
            2-Tuple:
                tiles = an array of tiles
                tile_area = tuple of (x, y) coordinates for the specific tile
        '''
        tiles = []
        tile_area = []
        desired_size = (
            int(img.size[0] / window_size * self.input_size),
            int(img.size[1] / window_size * self.input_size)
            )
        # print("desired size {}".format(desired_size))
        img = img.copy().resize(desired_size)
        for i in range(0, img.size[0] - self.input_size + 1, self.step):
            for j in range(0, img.size[1] - self.input_size + 1, self.step):
                tile = img.crop((i, j, i + self.input_size, j + self.input_size))
                tiles.append(np.array(tile).reshape((self.input_size, self.input_size, 1))/255)
                tile_area.append((i, j))
        return np.array(tiles), tile_area
    
    def _run(self, img:Image.Image, size:int):
        '''
        Run the search algorithm on a specific sized window across the img

        Inputs:
            img = image to search
            size = window size
        Returns:
            mask = 2D array where pixel is 1 for foreground and 0 for background
        '''
        window_size = min(img.size) * size
        print('Running with window size ' + str(window_size))
        img = img.copy()
        img.load()
        encoding = []
        tiles, tile_area = self._create_tiles(img, window_size)
        encoded = self.encoder.predict(tiles, verbose=1)
        snrs = self.discriminator.predict(encoded, verbose=1)
        decoded = self.decoder.predict(encoded)
        rows = int((img.size[0] / window_size * self.input_size))
        cols = int((img.size[1] / window_size * self.input_size))
        mask = np.zeros((rows, cols))
        img_array = np.array(img.resize(reversed(mask.shape)))
        for t, (i, j), shape, snr, code in zip(tiles, tile_area, decoded, snrs, encoded):
            snr = snr.squeeze()
            self.SNRs.append(snr)
            if snr > self.noise_threshold:
                mask[i:i+self.input_size, j:j+self.input_size] += shape.squeeze()
                self.mask_values.extend(shape.flatten())
                encoding.append(((i, j), code, np.median(img_array[i:i+self.input_size, j:j+self.input_size])))
                # plt.figure()
                # plt.imshow(t.squeeze(), cmap='gray')
                # plt.show()
        mask = mask/np.max(mask)
        mask = np.array(Image.fromarray(np.swapaxes(mask, 0, 1)).resize(img.size))
        self._display_figures(img, mask, window_size, ((rows, cols), encoding))
        thresholded_mask = np.zeros(mask.shape)
        thresholded_mask[np.where(mask > self.mask_threshold)] = 1
        thresholded_mask[np.where(mask <= self.mask_threshold)] = 0
        return thresholded_mask, (self.input_size, (rows, cols), encoding)
        
    def _display_figures(self, img:Image.Image, mask, window_size:int, encoding_bundle:tuple):
        '''
        Display figures if necessary

        Inputs:
            mask = 2D array of the foreground mask
        '''
        if 'original' in self.displays:
            plt.figure()
            plt.imshow(np.asarray(img), cmap='gray')
            plt.title('Original Image')
        
        if 'masked' in self.displays:
            colored = img.convert(mode='RGB')
            masked = np.array(colored).copy() / 2
            masked[:, :, 0] += (200/2) * mask
            
            plt.figure()
            plt.imshow(masked.astype('uint8'))
            plt.title('Masked image at size {}. Red = foreground'.format(window_size))
        
        if 'foreground' in self.displays:
            masked = np.array(img.convert(mode='RGB')).copy()
            masked[np.where(mask < self.mask_threshold)] = [0, 0, 255]

            plt.figure()
            plt.imshow(masked.astype('uint8'))
            plt.title('Foreground of image at size {}. Blue = Background'.format(window_size))

        if 'mask' in self.displays:
            width = mask.shape[1]
            border = [np.linspace(0, 1, width)] * int(mask.shape[0]/10)

            plt.figure()
            plt.imshow(np.vstack((border, mask)), cmap='gray')
            plt.title('Mask at size {}'.format(window_size))
        
        if 'SNR' in self.displays:
            plt.figure()
            plt.hist(self.SNRs)
            plt.title('SNR Levels at size {}'.format(window_size))
        
        if 'mask_histogram' in self.displays:
            plt.figure()
            plt.hist(self.mask_values)
            plt.title('Mask values at size {}'.format(window_size))
        
        if 'recreate' in self.displays:
            shape = encoding_bundle[0]
            encoding = encoding_bundle[1]
            recreated = np.zeros(shape) - 1

            positions, codes, fills = zip(*encoding)
            shapes = self.decoder.predict(np.array(codes))
            for (i, j), shape, fill in zip(positions, shapes, fills):
                recreated[i:i+self.input_size, j:j+self.input_size] += shape.squeeze() * fill
            plt.figure()
            plt.imshow(np.swapaxes(recreated, 0, 1), cmap='gray')
            plt.title('Recreated at window size {}, {:.2f}kb'.format(window_size, getsizeof(bz2.compress(pickle.dumps(encoding, 4)))/1000))

        
        # if show:
        #     plt.show()
        return

    def __call__(self, img:Image.Image, sizes=None, combine_mask='OR'):
        '''
        Run the search algorithm on image

        Inputs:
            img = image
            sizes = list of window sizes (as fractions of the smallest side of the image)
                defaults = [1/2, 1/4, 1/8]
            combine_mask = method to combine the masks
        Return:
            Image of the only foreground
        '''
        if sizes is None:
            sizes = [1/2, 1/4, 1/8, 1/16, 1/32]
        
        masks = [self._run(img, size)[0] for size in sizes]

        # combine mask
        mask = masks[0]
        for m in masks[1:]:
            if combine_mask == 'OR':
                mask += m
            elif combine_mask == 'AND':
                mask *= m
        mask.clip(0, 1)

        foreground = np.array(img.convert(mode='RGB')).copy()
        foreground[np.where(mask == 0)] = [0, 0, 255]
        return foreground, mask

            
