'''
Final Project - Winter 2018
Written by: Mike (JingHongYu) Bian

Train VAE encoder, decoder and disciminator
'''
import time

from keras.callbacks import TensorBoard

from models import create_VAE, create_VAE_discriminator, unbundle_discriminator
from data_loading import get_data


DATAPATH = ''
ID = str(int(time.time))

data = get_data(DATAPATH)
x_train = data['x_train']
y_train = data['y_train']
l_train = data['l_train']
x_test = data['x_test']
y_test = data['y_test']
l_test = data['l_test']

vae, encoder, decoder = create_VAE()

epochs = 50
vae.fit(x_train, y_train,
        epochs=epochs,
        validation_data=(x_test, y_test),
        batch_size=32,
        callbacks=[TensorBoard(log_dir='./tmp/'+ID+'/vae')])
vae.save('vae.nn')
encoder.save('encoder.nn')
decoder.save('decoder.nn')

discriminator = create_VAE_discriminator(encoder)
discriminator.fit

epochs = 10
discriminator.fit(x_train, l_train,
                  epochs=epochs,
                  validation_data=(x_test, l_test),
                  batch_size=32,
                  callbacks=[TensorBoard(log_dir='./tmp/'+ID+'/discriminator')])

discriminator = unbundle_discriminator(encoder, discriminator)
discriminator.save('discriminator.nn')
