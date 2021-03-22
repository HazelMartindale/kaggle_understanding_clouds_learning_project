import tensorflow as tf
import urllib


def get_vgg16_weights():
    """

    Returns: string

    """
    urllib.urlretrieve("https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
                       filename="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")    # assign to a variable
    vgg_weights_path = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    return vgg_weights_path


def block(f, n_convs, filters, kernel_size, activation, pool_size,
          pool_stride, block_name):
    """
     Defines a block in the VGG network.
    Args:
        f:
        n_convs: (tensor) -- input image
        filters: (int) -- number of convolution layers to append
        kernel_size: (int) -- number of filters for the convolution layers
        activation: (string or object) -- activation to use in the convolution
        pool_size: (int) -- size of the pooling layer
        pool_stride: (int) -- stride of the pooling layer
        block_name: (string) -- name of the block

    Returns:
       tensor containing the max-pooled output of the convolutions
    """

    for i in range(n_convs):
        f = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same',
                                   name="{}_conv{}".format(block_name, i + 1))(f)

    x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_stride,
                                     name="{}_pool{}".format(block_name, n_convs))(f)

    return f, x


def VGG_16(image_input, vgg_weights_path):
    """
    This function defines the VGG encoder.

    Args:
        image_input (tensor) - batch of images
        vgg_weights_path (string) -- the path to the vgg16 weights file
    Returns:
        tuple of tensors - output of all encoder blocks plus the final convolution layer
    """

    # create 5 blocks with increasing filters at each stage.
    # you will save the output of each block (i.e. p1, p2, p3, p4, p5). "p" stands for the pooling layer.
    f1, x = block(image_input, n_convs=2, filters=64, kernel_size=(3, 3),
                  activation='relu', pool_size=(2, 2), pool_stride=(2, 2),
                  block_name='block1')

    f2, x = block(x, n_convs=2, filters=128, kernel_size=(3, 3), activation='relu',
                  pool_size=(2, 2), pool_stride=(2, 2), block_name='block2')

    f3, x = block(x, n_convs=3, filters=256, kernel_size=(3, 3), activation='relu',
                  pool_size=(2, 2), pool_stride=(2, 2), block_name='block3')

    f4, x = block(x, n_convs=3, filters=512, kernel_size=(3, 3), activation='relu',
                  pool_size=(2, 2), pool_stride=(2, 2), block_name='block4')

    f5, x = block(x, n_convs=3, filters=512, kernel_size=(3, 3), activation='relu',
                  pool_size=(2, 2), pool_stride=(2, 2), block_name='block5')
    p5 = x

    # create the vgg model
    vgg = tf.keras.Model(image_input, p5)

    # load the pretrained weights you downloaded earlier
    vgg.load_weights(vgg_weights_path)

    return f1, f2, f3, f4, f5, p5


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


def bottleneck(inputs):
    """
    This function defines the bottleneck convolutions to extract more features before the upsampling layers.
    """

    bottle_neck = conv2d_block(inputs, n_filters=128)

    return bottle_neck


# Decoder Utilities

def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=2, dropout=0.3, padding='same'):
    """
    defines the one decoder block of the UNet

    Args:
        inputs (tensor) -- batch of input features
        conv_output (tensor) -- features from an encoder block
        n_filters (int) -- number of filters
        kernel_size (int) -- kernel size
        strides (int) -- strides for the deconvolution/upsampling
        padding (string) -- "same" or "valid", tells if shape will be preserved by zero padding
        dropout (float) -- the dropout rate

    Returns:
        c (tensor) -- output features of the decoder block
    """

    u = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides=strides, padding=padding)(inputs)

    c = tf.keras.layers.concatenate([u, conv_output])
    c = tf.keras.layers.Dropout(dropout)(c)
    c = conv2d_block(c, n_filters, kernel_size=3)

    return c


def decoder(convs, inputs, output_channels, dropout=0.3):
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
    f1, f2, f3, f4, f5 = convs
    print(f1, f2, f3, f4, f5)
    c6 = decoder_block(inputs, f5, n_filters=64, kernel_size=(3, 3), strides=(2, 2), dropout=dropout)
    c7 = decoder_block(c6, f4, n_filters=32, kernel_size=(3, 3), strides=(2, 2), dropout=dropout)
    c8 = decoder_block(c7, f3, n_filters=16, kernel_size=(3, 3), strides=(2, 2), dropout=dropout)
    c9 = decoder_block(c8, f2, n_filters=8, kernel_size=(3, 3), strides=(2, 2), dropout=dropout)
    c10 = decoder_block(c9, f1, n_filters=4, kernel_size=(3, 3), strides=(2, 2), dropout=dropout)
    outputs = tf.keras.layers.Conv2D(output_channels, (1, 1), activation='sigmoid')(c10)

    return outputs


def segmentation_model(model_input_shape, vgg_weights_path):
    """
    Defines the final segmentation model by chaining together the encoder and decoder.
    Args:
        model_input_shape (tuple(ints)) -- the shape of the input tensor
        vgg_weights_path (string) -- the file path to the weight file
    Returns:
      keras Model that connects the encoder and decoder networks of the segmentation model
    """

    inputs = tf.keras.layers.Input(shape=model_input_shape)  # (224,224,3,))
    convs = VGG_16(image_input=inputs, vgg_weights_path=vgg_weights_path)
    outputs = decoder(convs[:-1], convs[-1], 4)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model
