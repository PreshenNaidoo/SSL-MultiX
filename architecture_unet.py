import keras
import tensorflow as tf


def downsample_block(x, num_filters, dropout_prob=0, max_pooling=True):
    """
    Builds a downsampling block for a U-Net model with optional dropout and max pooling.

    Parameters:
        x (tensor): Input tensor.
        num_filters (int): Number of filters for the convolutional layers.
        dropout_prob (float): Dropout probability (default is 0).
        max_pooling (bool): Whether to apply max pooling (default is True).

    Returns:
        tuple: Output tensor after downsampling and skip connection tensor.
    """

    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    skip = x

    if max_pooling:
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    if dropout_prob > 0:
        x = tf.keras.layers.Dropout(dropout_prob)(x)

    return x, skip


# decoder
def upsample_block(x, skip_layer, num_filters):
    """
    Builds an upsampling block for a U-Net model with skip connections and transposed convolutions.

    Parameters:
        x (tensor): Input tensor from the previous layer.
        skip_layer (tensor): Tensor from the corresponding downsampling layer for skip connection.
        num_filters (int): Number of filters for the convolutional layers.

    Returns:
        tensor: Output tensor after upsampling and merging with the skip connection.
    """

    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.concatenate([x, skip_layer], axis=-1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x

def UNET(input_shape, num_output_channels, dropout = 0, is_segmentation = False, mixed_precision = False):
    """
    Creates a U-Net model for segmentation or general image-to-image tasks.

    Parameters:
        input_shape (tuple): Shape of the input images.
        num_output_channels (int): Number of output channels.
        dropout (float): Dropout probability (default is 0).
        is_segmentation (bool): Whether the model is for segmentation (default is False).
        mixed_precision (bool): Whether to use mixed precision for the final layer (default is False).

    Returns:
        keras.Model: Compiled U-Net model.
    """

    input_ = tf.keras.layers.Input(input_shape, name="Input")

    x = tf.keras.layers.Rescaling(1. /255.0)(input_)

    # downsampling / encoder
    x, skip0 = downsample_block(x, 32, dropout)
    x, skip1 = downsample_block(x, 64, dropout)
    x, skip2 = downsample_block(x, 128, dropout)
    x, skip3 = downsample_block(x, 256, dropout)
    x, skip4 = downsample_block(x, 512, dropout)
    x, _ = downsample_block(x, 1024, dropout, max_pooling=False)

    # upsampling / decoder
    x = upsample_block(x, skip4, 512)
    x = upsample_block(x, skip3, 256)
    x = upsample_block(x, skip2, 128)
    x = upsample_block(x, skip1, 64)
    x = upsample_block(x, skip0, 32)

    # We specify the dtype on the final layer, if we do not, when using
    # mixed_precision, during training the loss and val_loss will be 0.
    if not is_segmentation:
        if mixed_precision:
            output = tf.keras.layers.Conv2D(num_output_channels,
                                            kernel_size=(1, 1),
                                            padding='valid',
                                            activation='linear',
                                            name="output", dtype=tf.float32)(x)
        else:
            output = tf.keras.layers.Conv2D(num_output_channels,
                                            kernel_size=(1, 1),
                                            padding='valid',
                                            activation='linear',
                                            name="output")(x)
    else:
        if mixed_precision:
            output = tf.keras.layers.Conv2D(num_output_channels,
                                            kernel_size=(1, 1),
                                            padding='valid',
                                            activation='sigmoid',
                                            name="output", dtype=tf.float32)(x)
        else:
            output = tf.keras.layers.Conv2D(num_output_channels,
                                            kernel_size=(1, 1),
                                            padding='valid',
                                            activation='sigmoid',
                                            name="output")(x)

    model = tf.keras.Model(input_, output, name="Encoder")

    return model


def get_unet_encoder(input_shape, dropout_prob = 0):
    """
    Builds the encoder part of a U-Net model.

    Parameters:
        input_shape (tuple): Shape of the input images.
        dropout_prob (float): Dropout probability for each downsampling block (default is 0).

    Returns:
        tuple: Encoder model and a list of skip connection tensors.
    """

    input_ = tf.keras.layers.Input(input_shape, name="Input")

    x = tf.keras.layers.Rescaling(1. / 255.0)(input_)

    x, skip0 = downsample_block(x, 32, dropout_prob)
    x, skip1 = downsample_block(x, 64, dropout_prob)
    x, skip2 = downsample_block(x, 128, dropout_prob)
    x, skip3 = downsample_block(x, 256, dropout_prob)
    x, skip4 = downsample_block(x, 512, dropout_prob)
    x, _ = downsample_block(x, 1024, dropout_prob, max_pooling=False)

    encoder = tf.keras.Model(input_, x, name='Encoder')

    return encoder, [skip0, skip1, skip2, skip3, skip4]

def get_unet_encoder_decoder(input_shape, dropout_prob = 0, last_float32 = True):
    """
    Builds a full U-Net model with both encoder and decoder.

    Parameters:
        input_shape (tuple): Shape of the input images.
        dropout_prob (float): Dropout probability for each downsampling block (default is 0).
        last_float32 (bool): Whether the final layer should use dtype float32 (default is True).

    Returns:
        keras.Model: Full U-Net model with encoder and decoder.
    """

    input_ = tf.keras.layers.Input(input_shape, name="Input")

    x = tf.keras.layers.Rescaling(1. / 255.0)(input_)

    x, skip0 = downsample_block(x, 32, dropout_prob)
    x, skip1 = downsample_block(x, 64, dropout_prob)
    x, skip2 = downsample_block(x, 128, dropout_prob)
    x, skip3 = downsample_block(x, 256, dropout_prob)
    x, skip4 = downsample_block(x, 512, dropout_prob)
    x, _ = downsample_block(x, 1024, dropout_prob, max_pooling=False)

    # upsampling / decoder
    x = upsample_block(x, skip4, 512)
    x = upsample_block(x, skip3, 256)
    x = upsample_block(x, skip2, 128)
    x = upsample_block(x, skip1, 64)
    x = upsample_block(x, skip0, 32)

    if last_float32:
        output = tf.keras.layers.Conv2D(1,
                                        kernel_size=(1, 1),
                                        padding='valid',
                                        name="output", dtype=tf.float32)(x)
    else:
        output = tf.keras.layers.Conv2D(1,
                                        kernel_size=(1, 1),
                                        padding='valid',
                                        name="output")(x)

    encoder = tf.keras.Model(input_, output, name='Encoder')

    return encoder

def get_unet_encoder_with_rescaling(input_shape, dropout_prob = 0):
    """
    Builds the encoder part of a U-Net model with input rescaling.

    Parameters:
        input_shape (tuple): Shape of the input images.
        dropout_prob (float): Dropout probability for each downsampling block (default is 0).

    Returns:
        tuple: Encoder model and a list of skip connection tensors.
    """

    input_ = tf.keras.layers.Input(input_shape, name="Input")

    x = tf.keras.layers.Rescaling(1. / 255.0)(input_)

    x, skip0 = downsample_block(x, 32, dropout_prob)
    x, skip1 = downsample_block(x, 64, dropout_prob)
    x, skip2 = downsample_block(x, 128, dropout_prob)
    x, skip3 = downsample_block(x, 256, dropout_prob)
    x, skip4 = downsample_block(x, 512, dropout_prob)
    x, _ = downsample_block(x, 1024, dropout_prob, max_pooling=False)

    encoder = tf.keras.Model(input_, x, name='Encoder')

    return encoder, [skip0, skip1, skip2, skip3, skip4]

def create_unet_model_reconstructive(encoder,
                    skip_connections,
                    input_shape,
                    num_output_channels,
                    lock_entire_enc,
                    lock_enc_batch_norm_layers,
                    lock_all_batch_norm_layers):
    """
    Creates a reconstructive U-Net model using a pre-trained encoder and skip connections.

    Parameters:
        encoder (keras.Model): Pre-trained encoder model.
        skip_connections (list): List of skip connection tensors from the encoder.
        input_shape (tuple): Shape of the input images.
        num_output_channels (int): Number of output channels.
        lock_entire_enc (bool): Whether to lock all layers of the encoder (default is False).
        lock_enc_batch_norm_layers (bool): Whether to lock only the encoder's batch normalization layers.
        lock_all_batch_norm_layers (bool): Whether to lock all batch normalization layers in the model.

    Returns:
        keras.Model: U-Net model with reconstructive capabilities.
    """

    skip0, skip1, skip2, skip3, skip4 = skip_connections

    #lock encoder batch norm layers
    for layer in encoder.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


    if lock_entire_enc:
        for layer in encoder.layers:
            layer.trainable = False

    x = None
    x = encoder.output
    x = upsample_block(x, skip4, 512)
    x = upsample_block(x, skip3, 256)
    x = upsample_block(x, skip2, 128)
    x = upsample_block(x, skip1, 64)
    x = upsample_block(x, skip0, 32)

    output = tf.keras.layers.Conv2D(num_output_channels,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    activation='sigmoid',
                                    name="output", dtype=tf.float32)(x)

    model = tf.keras.Model(inputs=encoder.input, outputs=output, name="Encoder")

    if lock_all_batch_norm_layers:
        # lock all batch norm layers
        for layer in model.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False

    return model
