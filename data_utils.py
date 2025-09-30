import tensorflow as tf
import os
import random
import math
import csv

from augmentation_util import get_augmenter_echo

images_folder = 'Images'  # The labelled images
labels_folder = 'Labels'  # The expert labels/annotations

IMAGE_SIZE = 512
IMAGE_CHANNELS = 1

AUG_MIN_AREA = 0.9
AUG_ROTATION = 0.05

def process_image_only(image_file):
    """
    Processes a single image file by reading, decoding (PNG or JPEG), converting to grayscale,
    and resizing if necessary.

    Parameters:
        image_file (str): The path to the image file.

    Returns:
        tf.Tensor: The processed image as a grayscale tensor of size (IMAGE_SIZE, IMAGE_SIZE).
    """

    ext = tf.strings.substr(image_file, tf.strings.length(image_file) - 3, 3)
    png_txt = tf.convert_to_tensor('png')
    img = tf.io.read_file(image_file)

    if (tf.strings.regex_full_match(ext, png_txt)):
        image = tf.image.decode_png(img, channels=3)
    else:
        image = tf.image.decode_jpeg(img, channels=3)

    image = tf.image.rgb_to_grayscale(image)

    h, w = image.shape[:2]

    if h!= IMAGE_SIZE and w!= IMAGE_SIZE:
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    return image

def process_image_and_labels_segmentation(image_file, label_file):
    """
    Processes an image and its corresponding label file for segmentation by reading, decoding (PNG or JPEG),
    and converting both to grayscale tensors.

    Parameters:
        image_file (str): The path to the image file.
        label_file (str): The path to the label file.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: The processed image and label tensors.
    """

    ext = tf.strings.substr(image_file, tf.strings.length(image_file) - 3, 3)
    png_txt = tf.convert_to_tensor('png')
    img = tf.io.read_file(image_file)

    if (tf.strings.regex_full_match(ext, png_txt)):
        image = tf.image.decode_png(img, channels=3)
    else:
        image = tf.image.decode_jpeg(img, channels=3)

    image = tf.image.rgb_to_grayscale(image)


    ext1 = tf.strings.substr(label_file, tf.strings.length(label_file) - 3, 3)
    png_txt1 = tf.convert_to_tensor('png')
    lbl = tf.io.read_file(label_file)

    if (tf.strings.regex_full_match(ext1, png_txt1)):
        label = tf.image.decode_png(lbl, channels=1)
    else:
        label = tf.image.decode_jpeg(lbl, channels=1)

    mask = label

    return image, mask

def process_image_and_labels_segmentation_resize(image_file, label_file):
    """
    Processes and resizes an image and its corresponding label file for segmentation by reading,
    decoding (PNG or JPEG), converting to grayscale, and resizing both to (IMAGE_SIZE, IMAGE_SIZE).

    Parameters:
        image_file (str): The path to the image file.
        label_file (str): The path to the label file.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: The resized image and label tensors.
    """

    ext = tf.strings.substr(image_file, tf.strings.length(image_file) - 3, 3)
    png_txt = tf.convert_to_tensor('png')
    img = tf.io.read_file(image_file)

    if (tf.strings.regex_full_match(ext, png_txt)):
        image = tf.image.decode_png(img, channels=3)
    else:
        image = tf.image.decode_jpeg(img, channels=3)

    image = tf.image.rgb_to_grayscale(image)

    # PROCESS LABEL FILE INTO TENSOR
    ext1 = tf.strings.substr(label_file, tf.strings.length(label_file) - 3, 3)
    png_txt1 = tf.convert_to_tensor('png')
    lbl = tf.io.read_file(label_file)

    if (tf.strings.regex_full_match(ext1, png_txt1)):
        label = tf.image.decode_png(lbl, channels=1)
    else:
        label = tf.image.decode_jpeg(lbl, channels=1)

    mask = label

    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    mask = tf.image.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))

    return image, mask

def prepare_tensor_dataset_segmentation(images, labels, batch_size, augmentation = False):
    """
    Prepares a TensorFlow dataset for segmentation by mapping image-label pairs, applying augmentation (if enabled),
    and batching the data.

    Parameters:
        images (list): List of image file paths.
        labels (list): List of label file paths.
        batch_size (int): The size of each batch.
        augmentation (bool): Whether to apply data augmentation. Default is False.

    Returns:
        tf.data.Dataset: The prepared dataset for segmentation.
    """

    image_files = tf.data.Dataset.from_tensor_slices((images, labels))
    tensor_dataset = image_files.map(process_image_and_labels_segmentation, num_parallel_calls=tf.data.AUTOTUNE)
    tensor_dataset = tensor_dataset.cache().shuffle(buffer_size=10 * batch_size).batch(batch_size)

    if augmentation:
        print('AUGMENTATION ENABLED.')
        aug = get_augmenter_echo(IMAGE_SIZE, IMAGE_CHANNELS, AUG_MIN_AREA, AUG_ROTATION)
        tensor_dataset = tensor_dataset.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    tensor_dataset = tensor_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return tensor_dataset

def prepare_tensor_dataset_segmentation_resize(images, labels, batch_size, augmentation = False):
    """
    Prepares a TensorFlow dataset for segmentation with resizing by mapping image-label pairs, resizing,
    applying augmentation (if enabled), and batching the data.

    Parameters:
        images (list): List of image file paths.
        labels (list): List of label file paths.
        batch_size (int): The size of each batch.
        augmentation (bool): Whether to apply data augmentation. Default is False.

    Returns:
        tf.data.Dataset: The prepared dataset for segmentation with resizing.
    """

    image_files = tf.data.Dataset.from_tensor_slices((images, labels))
    tensor_dataset = image_files.map(process_image_and_labels_segmentation_resize, num_parallel_calls=tf.data.AUTOTUNE)
    tensor_dataset = tensor_dataset.cache().shuffle(buffer_size=10 * batch_size).batch(batch_size)

    if augmentation:
        aug = get_augmenter_echo(IMAGE_SIZE, IMAGE_CHANNELS, AUG_MIN_AREA, AUG_ROTATION)
        tensor_dataset = tensor_dataset.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    tensor_dataset = tensor_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return tensor_dataset

def load_labelled_datasets_unity(dataset_path, batch_size, percentage_train = None):
    """
    Loads and prepares labeled datasets for training, validation, and testing
    from a Unity-based directory structure.

    Parameters:
        dataset_path (str): Path to the dataset directory.
        batch_size (int): Batch size for the prepared datasets.
        percentage_train (float, optional): Percentage of the training data to use. Default is None.

    Returns:
        Tuple: Prepared training and validation datasets along with lists of image and label file paths
               for training, validation, and testing, and their respective counts.
    """

    all_images, all_labels = [], []
    training_images, training_labels = [], []
    val_images, val_labels = [], []
    test_images, test_labels = [], []

    dataset = dataset_path

    train_labels_folder = os.path.join(os.path.join(dataset, labels_folder), 'training')
    val_labels_folder = os.path.join(os.path.join(dataset, labels_folder), 'validation')
    test_labels_folder = os.path.join(os.path.join(dataset, labels_folder), 'testing')

    set_labels = os.listdir(train_labels_folder)
    random.shuffle(set_labels)

    for file in set_labels:
        label_file = os.path.join(train_labels_folder, file)
        image_path = os.path.join(os.path.join(dataset, images_folder), 'training')
        image_file = os.path.join(image_path, file)
        training_labels.append(label_file)
        training_images.append(image_file)
        all_labels.append(label_file)
        all_images.append(image_file)

    set_labels = os.listdir(val_labels_folder)

    for file in set_labels:
        label_file = os.path.join(val_labels_folder, file)
        image_path = os.path.join(os.path.join(dataset, images_folder), 'validation')
        image_file = os.path.join(image_path, file)
        val_labels.append(label_file)
        val_images.append(image_file)
        all_labels.append(label_file)
        all_images.append(image_file)

    set_labels = os.listdir(test_labels_folder)
    set_labels.sort()

    for file in set_labels:
        label_file = os.path.join(test_labels_folder, file)
        image_path = os.path.join(os.path.join(dataset, images_folder), 'testing')
        image_file = os.path.join(image_path, file)
        test_labels.append(label_file)
        test_images.append(image_file)
        all_labels.append(label_file)
        all_images.append(image_file)

    if percentage_train is not None:
        num = int(len(training_images) * percentage_train/100)
        training_images = training_images[:num]
        training_labels = training_labels[:num]

    training_dataset = prepare_tensor_dataset_segmentation(training_images, training_labels, batch_size)
    validation_dataset = prepare_tensor_dataset_segmentation(val_images, val_labels, batch_size)

    return (training_dataset, validation_dataset,
            training_images, training_labels,
            val_images, val_labels,
            test_images, test_labels,
            len(training_images), len(val_images), len(test_images))

def load_labelled_datasets_unity_representative(dataset_path, representative_file, batch_size, percentage_train = None):
    """
    Loads and prepares labeled datasets for training, validation, and testing
    using a representative file to prioritize specific labels.

    Parameters:
        dataset_path (str): Path to the dataset directory.
        representative_file (str): Path to the representative file (CSV format).
        batch_size (int): Batch size for the prepared datasets.
        percentage_train (float, optional): Percentage of the training data to use. Default is None.

    Returns:
        Tuple: Prepared training and validation datasets along with lists of image and label file paths
               for training, validation, and testing, and their respective counts.
    """

    all_images, all_labels = [], []
    training_images, training_labels = [], []
    val_images, val_labels = [], []
    test_images, test_labels = [], []

    dataset = dataset_path

    train_labels_folder = os.path.join(os.path.join(dataset, labels_folder), 'training')
    val_labels_folder = os.path.join(os.path.join(dataset, labels_folder), 'validation')
    test_labels_folder = os.path.join(os.path.join(dataset, labels_folder), 'testing')

    set_labels = []
    with open(representative_file, 'r') as file:
        reader = csv.reader(file)
        cnt=0
        for row in reader:
            if cnt==0:
                cnt += 1
                continue
            set_labels.append(row[0])

    set_labels.reverse()  #least to most representative

    for file in set_labels:
        label_file = os.path.join(train_labels_folder, file)
        image_path = os.path.join(os.path.join(dataset, images_folder), 'training')
        image_file = os.path.join(image_path, file)
        if os.path.exists(label_file) and os.path.exists(image_file):
            training_labels.append(label_file)
            training_images.append(image_file)
            all_labels.append(label_file)
            all_images.append(image_file)

    set_labels = os.listdir(val_labels_folder)

    for file in set_labels:
        label_file = os.path.join(val_labels_folder, file)
        image_path = os.path.join(os.path.join(dataset, images_folder), 'validation')
        image_file = os.path.join(image_path, file)
        val_labels.append(label_file)
        val_images.append(image_file)
        all_labels.append(label_file)
        all_images.append(image_file)

    set_labels = os.listdir(test_labels_folder)
    set_labels.sort()

    for file in set_labels:
        label_file = os.path.join(test_labels_folder, file)
        image_path = os.path.join(os.path.join(dataset, images_folder), 'testing')
        image_file = os.path.join(image_path, file)
        test_labels.append(label_file)
        test_images.append(image_file)
        all_labels.append(label_file)
        all_images.append(image_file)

    if percentage_train is not None:
        num = int(len(training_images) * percentage_train/100)
        training_images = training_images[:num]
        training_labels = training_labels[:num]

    training_dataset = prepare_tensor_dataset_segmentation(training_images, training_labels, batch_size)
    validation_dataset = prepare_tensor_dataset_segmentation(val_images, val_labels, batch_size)

    return (training_dataset, validation_dataset,
            training_images, training_labels,
            val_images, val_labels,
            test_images, test_labels,
            len(training_images), len(val_images), len(test_images))

def load_unlabelled_datasets(unlabelled_dataset_path, batch_size, num_devices, num_train, exact_gpu_batch_split = True):
    """
    Loads and prepares an unlabeled dataset for training and validation.

    Parameters:
        unlabelled_dataset_path (str): Path to the directory containing unlabeled images.
        batch_size (int): Batch size for training and validation.
        num_devices (int): Number of devices (GPUs) to use for batch splitting.
        num_train (int): Number of training samples to load.
        exact_gpu_batch_split (bool): Whether to split batches exactly across GPUs. Default is True.

    Returns:
        Tuple: Unlabeled training and validation datasets, along with their respective lengths and a list of remaining frames.
    """

    files = os.listdir(unlabelled_dataset_path)
    random.shuffle(files)

    all_frames = []
    for file in files:
        all_frames.append(os.path.join(unlabelled_dataset_path, file))

    num = num_train
    per_step = num/float(batch_size)
    if exact_gpu_batch_split:
        num_multiple = int(per_step) * batch_size
        if num_multiple == 0:
            num_temp = int(num/num_devices)
            num = num_temp * num_devices
        else:
            num = math.ceil(per_step) * batch_size

    train_frames = all_frames[:num]

    val_percentage = 0.10
    if num_train > 1000 and num_train <= 60000:
        val_percentage = 0.05
    elif num_train > 60000:
        val_percentage = 0.025

    #calc val size
    num_val = int(len(train_frames) * val_percentage)
    per_step = num_val / float(batch_size)
    if exact_gpu_batch_split:
        num_multiple = int(per_step) * batch_size
        num_val = math.ceil(per_step) * batch_size
        # if num_multiple == 0:
        #     num_val = num
        # else:
        #     num_val = math.ceil(per_step) * batch_size

    val_frames = all_frames[num: num + num_val]

    balance_frames = all_frames[num + num_val:]

    # Load the unlabeled and labeled samples for training
    unlabelled_train = (
        tf.data.Dataset.from_tensor_slices((train_frames))
        .map(process_image_only, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=batch_size * 10)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Load the unlabeled and labeled samples for training
    unlabelled_val= (
        tf.data.Dataset.from_tensor_slices((val_frames))
        .map(process_image_only, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=batch_size * 10)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return unlabelled_train, len(train_frames), unlabelled_val, len(val_frames), balance_frames
