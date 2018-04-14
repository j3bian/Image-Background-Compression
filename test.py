
from keras.models import load_model

from search import ForegroundSearch, open_image
from matplotlib import pyplot as plt


encoder = load_model('encoder.nn')
decoder = load_model('decoder.nn')
discriminator = load_model('discriminator.nn')

image = open_image('')

# search = ForegroundSearch(encoder, decoder, discriminator, displays=['masked', 'foreground'])
search = ForegroundSearch(encoder, decoder, discriminator)

foreground = search(image)

plt.figure()
plt.imshow(image)
plt.title('Original')
plt.show()

plt.figure()
plt.imshow(foreground)
plt.title('Foreground')

plt.show()