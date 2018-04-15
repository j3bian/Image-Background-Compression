'''
Final Project - Winter 2018
Written by: Mike (JingHongYu) Bian

Test foreground extraction algorithm
'''
from keras.models import load_model
from PIL import ImageFilter, Image

from search import ForegroundSearch, open_image
from matplotlib import pyplot as plt
import numpy as np

encoder = load_model('default_encoder.nn')
decoder = load_model('default_decoder.nn')
discriminator = load_model('default_discriminator.nn')

IMG_PATH = 'sample'

image = open_image(IMG_PATH + '.jpg')
blurred = image.copy().filter(ImageFilter.GaussianBlur(radius=2))

search = ForegroundSearch(encoder, decoder, discriminator)

foreground, mask = search(image)

plt.figure()
plt.imshow(np.asarray(image), cmap='gray')
plt.title('Original')

plt.figure()
plt.imshow(foreground)
plt.title('Foreground')

compressed = np.array(image).copy()
compressed[np.where(mask == 0)] = np.asarray(blurred)[np.where(mask == 0)]

plt.figure()
plt.imshow(compressed, cmap='gray')
plt.title('Compressed')

compressed = Image.fromarray(compressed)
compressed.save(IMG_PATH + '_compressed.jpg')

plt.show()