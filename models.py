'''
Final Project - Winter 2018
Written by: Mike (JingHongYu) Bian

Helpers for creating models
'''
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D, Dropout, Flatten, Lambda, Input, UpSampling2D, Reshape
from keras.callbacks import Callback, TensorBoard
from keras import metrics, optimizers, Model
from keras import backend as K
import numpy as np


def create_VAE(input_size=16):
    '''
    Returns a VAE model that takes in a image of size (input_size, input_size)

    Inputs:
        inputs_size = the size of input images to the encoder
    Returns:
        3-Tuple:
            VAE model,
            VAE encoder,
            VAE decoder
    '''
    encoding_size = 3
    epsilon_std = 1.0
    conv_filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)
    conv_levels = 3
    intermediate_size = 128

    x = Input(shape=(input_size, input_size, 1))
    h = Conv2D(conv_filters, kernel_size, activation='relu', padding='same')(x)
    for _ in range(conv_levels - 1):
        h = MaxPooling2D(pool_size, padding='same')(h)
        h = Conv2D(conv_filters, kernel_size, activation='relu', padding='same')(h)
    # h = MaxPooling2D(pool_size, padding='same')(h)
    # h = Conv2D(cov_filters, kernel_size, activation='relu', padding='same')(h)
    original_shape = h.get_shape().as_list()[1:]
    h = Flatten()(h)
    h = Dense(intermediate_size)(h)

    z_mean = Dense(encoding_size)(h)
    z_log_var = Dense(encoding_size)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], encoding_size), mean=0.,
                                stddev=epsilon_std)
        return z_mean + K.exp(z_log_sigma / 2) * epsilon
    
    z = Lambda(sampling)([z_mean, z_log_var])
    generator_input = Input(shape=z.get_shape().as_list()[1:])

    h = Dense(intermediate_size)
    decode_h = h(z)
    generator_h = h(generator_input)
    
    h = Dense(np.cumprod(original_shape)[-1])
    decode_h = h(decode_h)
    generator_h = h(generator_h)

    h = Reshape(original_shape)
    decode_h = h(decode_h)
    generator_h = h(generator_h)

    h = Conv2D(conv_filters, kernel_size, activation='relu', padding='same')
    decode_h = h(decode_h)
    generator_h = h(generator_h)

    for _ in range(conv_levels - 1):
        h = UpSampling2D(pool_size)
        decode_h = h(decode_h)
        generator_h = h(generator_h)
        h = Conv2D(conv_filters, kernel_size, activation='relu', padding='same')
        decode_h = h(decode_h)
        generator_h = h(generator_h)

    h = Dense(1, activation='sigmoid')
    decode_h = h(decode_h)
    generator_h = h(generator_h)


    encoder = Model(x, z)
    generator = Model(generator_input, generator_h)
    vae = Model(x, decode_h)

    def vae_loss(x, x_decoded_mean):
        xent_loss = K.sum(K.sum(metrics.binary_crossentropy(x, x_decoded_mean), axis=-1), axis=-1)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss
    
    opt = optimizers.Adam()

    vae.compile(optimizer=opt, loss=vae_loss)
    generator.compile(optimizer=opt, loss='binary_crossentropy')
    encoder.compile(optimizer=opt, loss='binary_crossentropy')

    return vae, encoder, generator


def create_VAE_discriminator(encoder:Model):
    '''
    Creates a discriminator to classify images into containing and not containing objects based off of a trained VAE encoder.
    Locks the weights of the encoder and only trains end discriminator level.
    Inputs:
        encoder = encoder model from the VAE that encodes an image
    Returns:
        A discrminator model built off of the encoder 
    '''
    for layer in encoder.layers:
        layer.trainable = False
    output = Dense(1, activation='sigmoid')(encoder.layers[-1].output)
    discriminator = Model(encoder.layers[0].input, output)
    discriminator.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return discriminator


def unbundle_discriminator(encoder:Model, discriminator:Model):
    '''
    Unbundles the encoder within the discriminator by extracting the last
    layer of the discriminator

    Inputs:
        encoder = bundled encoder within discriminator
        discriminator = discriminator with bundled encoder
    Returns:
        model of discriminator without bundled encoder
    '''
    input_layer = Input(encoder.output_shape[1:])
    output_layer = discriminator.layers[-1](input_layer)
    discriminator = Model(input_layer, output_layer)
    discriminator.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return discriminator
