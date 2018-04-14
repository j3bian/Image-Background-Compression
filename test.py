'''
Final Project - Winter 2018
Written by: Mike (JingHongYu) Bian

Test foreground extraction algorithm
'''
from keras.models import load_model

from search import ForegroundSearch, open_image
from matplotlib import pyplot as plt
import numpy as np

encoder = load_model('default_encoder.nn')
decoder = load_model('default_decoder.nn')
discriminator = load_model('default_discriminator.nn')

image = open_image('sample.jpg')

search = ForegroundSearch(encoder, decoder, discriminator)

foreground = search(image)

plt.figure()
plt.imshow(np.asarray(image), cmap='gray')
plt.title('Original')

plt.figure()
plt.imshow(foreground)
plt.title('Foreground')

plt.show()