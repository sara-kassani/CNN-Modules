from tensorflow.keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, UpSampling2D, Dropout, Lambda, Activation, Add, LeakyReLU, ZeroPadding2D


def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x