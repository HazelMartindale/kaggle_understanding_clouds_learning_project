import tensorflow as tf


# Encoder Utilities
def conv2d_block(input_tensor, n_filters, kernel_size=3):
    """
    Adds 2 convolutional layers with the parameters passed to it

    Args:
      input_tensor (tensor) -- the input tensor
      n_filters (int) -- number of filters
      kernel_size (int) -- kernel size for the convolution

    Returns:
      tensor of output features
    """
    # first layer
    x = input_tensor
    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                                   kernel_initializer='he_normal', padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)

    return x


def encoder_block(inputs, n_filters=64, pool_size=(2, 2), dropout=0.3):
    """
    Adds two convolutional blocks and then perform down sampling on output of convolutions.

    Args:
        inputs (tensor) -- the input tensor
        n_filters (int) -- number of filters
        pool_size (int) -- size for pooling layer
        dropout (float) -- the dropout rate
    Returns:
        f - the output features of the convolution block
        p - the maxpooled features with dropout
    """

    f = conv2d_block(inputs, n_filters=n_filters)
    p = tf.keras.layers.MaxPooling2D(pool_size=pool_size, padding='same')(f)
    p = tf.keras.layers.Dropout(dropout)(p)

    return f, p


def encoder(inputs, dropout=0.3):
    """
    This function defines the encoder or downsampling path.

    Args:
        inputs (tensor) -- batch of input images
        dropout (float) -- the dropout rate
    Returns:
        p4 - the output maxpooled features of the last encoder block
        (f1, f2, f3, f4) - the output features of all the encoder blocks
    """
    f1, p1 = encoder_block(inputs, n_filters=8, pool_size=(2, 2), dropout=dropout)
    f2, p2 = encoder_block(p1, n_filters=16, pool_size=(2, 2), dropout=dropout)
    f3, p3 = encoder_block(p2, n_filters=32, pool_size=(2, 2), dropout=dropout)
    f4, p4 = encoder_block(p3, n_filters=64, pool_size=(2, 2), dropout=dropout)

    return p4, (f1, f2, f3, f4)


def bottleneck(inputs):
    """
    This function defines the bottleneck convolutions to extract more features before the upsampling layers.

    Args:
        inputs (tensor) -- batch of input features

    Retutns:
        bottleneck (tensor) -- output features of the decoder block
    """

    bottle_neck = conv2d_block(inputs, n_filters=128)

    return bottle_neck


# Decoder Utilities

def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=2, dropout=0.3,
                  padding='same'):
    """
    defines the one decoder block of the UNet

    Args:
        inputs (tensor) -- batch of input features
        conv_output (tensor) -- features from an encoder block
        n_filters (int) -- number of filters
        kernel_size (int) -- kernel size
        strides (int) -- strides for the deconvolution/upsampling
        dropout (float) -- the dropoutrate
        padding (string) -- "same" or "valid", tells if shape will be preserved by zero padding

    Returns:
        c (tensor) -- output features of the decoder block
    """

    u = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides=strides, padding=padding)(inputs)

    c = tf.keras.layers.concatenate([u, conv_output])
    c = tf.keras.layers.Dropout(dropout)(c)
    c = conv2d_block(c, n_filters, kernel_size=3)

    return c


def decoder(inputs, convs, output_channels, dropout=0.3):
    """
    Defines the decoder of the UNet chaining together 4 decoder blocks.

    Args:
        inputs (tensor) -- batch of input features
        convs (tuple) -- features from the encoder blocks
        output_channels (int) -- number of classes in the label map
        dropout (float) -- the dropout rate
    Returns:
        outputs (tensor) -- the pixel wise label map of the image
    """

    f1, f2, f3, f4 = convs

    c6 = decoder_block(inputs, f4, n_filters=64, kernel_size=(3, 3), strides=(2, 2), dropout=dropout)
    c7 = decoder_block(c6, f3, n_filters=32, kernel_size=(3, 3), strides=(2, 2), dropout=dropout)
    c8 = decoder_block(c7, f2, n_filters=16, kernel_size=(3, 3), strides=(2, 2), dropout=dropout)
    c9 = decoder_block(c8, f1, n_filters=8, kernel_size=(3, 3), strides=(2, 2), dropout=dropout)

    outputs = tf.keras.layers.Conv2D(output_channels, (1, 1), activation='sigmoid')(c9)

    return outputs


def unet(input_shape=(320, 480, 3,), dropout=0.3, output_channels=4):
    """
    Defines the UNet by connecting the encoder, bottleneck and decoder.

    Args:
        input_shape (tuple(int)) -- shape of inout layer
        dropout (float) -- the dropout rate
        output_channels (int) -- the number of output classes
    """

    # specify the input shape
    inputs = tf.keras.layers.Input(shape=input_shape)

    # feed the inputs to the encoder
    encoder_output, convs = encoder(inputs, dropout=dropout)

    # feed the encoder output to the bottleneck
    bottle_neck = bottleneck(encoder_output)

    # feed the bottleneck and encoder block outputs to the decoder
    # specify the number of classes via the `output_channels` argument
    outputs = decoder(bottle_neck, convs, output_channels=output_channels, dropout=dropout)

    # create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model
