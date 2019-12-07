from tensorflow.keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, UpSampling2D, Dropout, Lambda, Activation, Add, LeakyReLU, ZeroPadding2D


def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x