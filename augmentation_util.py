import keras
import keras_cv
import tensorflow as tf
from tensorflow.keras import layers
import math

def get_augmenter(image_size, image_channels, min_area):
    """
    Creates a data augmentation pipeline with random translation, zoom, and rotation.

    Parameters:
       image_size (int): The height and width of the input image.
       image_channels (int): The number of channels in the input image (e.g., 3 for RGB).
       min_area (float): The minimum area fraction to retain during the zoom transformation.

    Returns:
       keras.Sequential: A sequential model containing the augmentation layers.
    """

    zoom_factor = 1.0 - math.sqrt(min_area)
    return keras.Sequential(
        [
            keras.Input(shape=(image_size, image_size, image_channels)),
            layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
            layers.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
            layers.RandomRotation(0.05),
        ]
    )

def get_augmenter_echo(image_size, image_channels, min_area, rotation):
    """
    Creates a data augmentation pipeline for echocardiographic images with random translation, zoom, and a custom rotation angle.

    Parameters:
        image_size (int): The height and width of the input image.
        image_channels (int): The number of channels in the input image (e.g., 3 for RGB).
        min_area (float): The minimum area fraction to retain during the zoom transformation.
        rotation (float): The maximum rotation angle in radians for the rotation transformation.

    Returns:
        keras.Sequential: A sequential model containing the augmentation layers.
    """

    zoom_factor = 1.0 - math.sqrt(min_area)
    return keras.Sequential(
        [
            keras.Input(shape=(image_size, image_size, image_channels)),
            layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
            layers.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
            layers.RandomRotation(rotation)
        ]
    )

def get_augmenter_zoom_trans(image_size, image_channels, min_area):
    """
    Creates a data augmentation pipeline with random translation and zoom, excluding rotation.

    Parameters:
        image_size (int): The height and width of the input image.
        image_channels (int): The number of channels in the input image (e.g., 3 for RGB).
        min_area (float): The minimum area fraction to retain during the zoom transformation.

    Returns:
        keras.Sequential: A sequential model containing the augmentation layers.
    """

    zoom_factor = 1.0 - math.sqrt(min_area)
    return keras.Sequential(
        [
            keras.Input(shape=(image_size, image_size, image_channels)),
            layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
            layers.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0))
        ]
    )

