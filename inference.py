import json
import os

import cv2
import numpy as np

import drawhelper
from drawhelper import get_rgb_image, get_overlay_image
from measurements import get_dice_and_hd_from_masks
from utils import write_dict_to_json

import tensorflow as tf
from data_utils import process_image_only


def infer_test_segmentation(model, test_images, label_masks, save_folder, mask_shape, batch_size, inference_name):
    """
    Perform segmentation inference on test images and evaluate predictions using dice coefficient and Hausdorff distance.

    Args:
        model: Trained model for segmentation inference.
        test_images (list): List of paths to the test images.
        label_masks (list): List of ground truth label masks corresponding to the test images.
        save_folder (str): Path to the folder where results will be saved.
        mask_shape (tuple): Shape of the predicted mask (height, width, channels).
        batch_size (int): Number of images processed in each batch.
        inference_name (str): Name used for saving the inference results.

    Returns:
        dict: Dictionary containing average dice and Hausdorff distance scores.
    """

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    examples_path = os.path.join(save_folder, f'{inference_name} examples')
    if not os.path.exists(examples_path):
        os.makedirs(examples_path)

    frames_list = test_images
    all_frames_masks, _ = infer_images_in_batches_segmentation(model,
                                                                                    frames_list,
                                                                                    mask_shape,
                                                                                    batch_size)
    assert len(all_frames_masks) == len(frames_list)

    write_dict_to_json({'raw_outputs': all_frames_masks.tolist(),
                       'frames_list': frames_list},
                      save_folder,
                      f'{inference_name}_raw_model_outputs.json')

    # Save all predictions in example folder
    all_frames_masks_contours = []
    for i in range(len(frames_list)):
        image_file = frames_list[i]
        label_mask = np.array(label_masks[i])

        pred_mask = np.array(all_frames_masks[i])
        pred_mask = pred_mask[: ,: ,0]
        pred_mask = tf.greater(pred_mask, 0.5)
        pred_mask = tf.dtypes.cast(pred_mask, tf.uint8)
        pred_mask = pred_mask.numpy()

        if label_mask.shape[0] != pred_mask.shape[0] or label_mask.shape[1] != pred_mask.shape[1]:
            label_mask = label_mask[:, :, np.newaxis]
            label_mask_resize = tf.image.resize(label_mask, (pred_mask.shape[0], pred_mask.shape[1]))
            label_mask = label_mask_resize.numpy().astype(np.uint8)
            label_mask = label_mask[: ,: ,0]

        image_rgb = cv2.imread(image_file, cv2.IMREAD_COLOR)
        mask_rgb = get_rgb_image(pred_mask, (int(0) ,int(255) ,int(0)))

        if image_rgb.shape[0] != pred_mask.shape[0] or image_rgb.shape[1] != pred_mask.shape[1]:
            image_rgb = tf.image.resize(image_rgb, (pred_mask.shape[0], pred_mask.shape[1]))
            image_rgb = image_rgb.numpy().astype(np.uint8)

        dst_img = cv2.addWeighted(image_rgb, 1, mask_rgb, 0.3, 0)
        out_file = os.path.join(examples_path, os.path.basename(image_file).replace('.png' ,'' )+ '_1.png')
        cv2.imwrite(out_file, dst_img)

        img_curves, poly_pointsp = get_overlay_image(image_rgb, label_mask, pred_mask)
        out_file = os.path.join(examples_path, os.path.basename(image_file).replace('.png' ,'' )+ '_2.png')
        cv2.imwrite(out_file, img_curves)

        if poly_pointsp is not None:
            #build pred mask from points/contour
            contour_mask = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
            contour_mask = drawhelper.draw_poly_on_image(contour_mask, poly_pointsp, (255, 255, 255), False)
            contour_mask = drawhelper.fill_poly_on_image(contour_mask, poly_pointsp, (255, 255, 255))
            contour_mask = drawhelper.get_binary_image(contour_mask)
            all_frames_masks_contours.append(contour_mask)

            contour_mask_rgb = get_rgb_image(contour_mask, (int(0), int(255), int(0)))
            dst_img = cv2.addWeighted(image_rgb, 1, contour_mask_rgb, 0.3, 0)
            out_file = os.path.join(examples_path, os.path.basename(image_file).replace('.png', '') + '_3.png')
            cv2.imwrite(out_file, dst_img)
        else:
            contour_mask = all_frames_masks[i]
            contour_mask = tf.greater(contour_mask[: ,: ,0], 0.5)
            contour_mask = tf.dtypes.cast(contour_mask, tf.float32)
            all_frames_masks_contours.append(contour_mask.numpy())

    # Score predictions. Scores to be used to select pseudo-labels later on.
    # Score per frame
    frames_indexes = list(range(1, len(frames_list ) +1))
    dices_endo = []
    hds_endo = []
    frames_details = {}
    for i in range(len(frames_list)):
        img_path = frames_list[i]

        pred_mask = all_frames_masks[i]

        contour_mask = all_frames_masks_contours[i]

        label_mask = label_masks[i]

        pred_mask = tf.greater(pred_mask[: ,: ,0], 0.5)
        pred_mask = tf.dtypes.cast(pred_mask, tf.float32)
        pred_mask = pred_mask.numpy()

        if label_mask.shape[0] != pred_mask.shape[0] or label_mask.shape[1] != pred_mask.shape[1]:
            label_mask = label_mask[:, :, np.newaxis]
            label_mask_resize = tf.image.resize(label_mask, (pred_mask.shape[0], pred_mask.shape[1]))
            label_mask = label_mask_resize.numpy().astype(np.uint8)
            label_mask = label_mask[: ,: ,0]

        dice_endo, hd_endo = get_dice_and_hd_from_masks(label_mask, pred_mask)

        dice_endo1, hd_endo1 = get_dice_and_hd_from_masks(label_mask, contour_mask)

        dice, hd = dice_endo, hd_endo

        frame_info_dict = {}
        frame_info_dict['dice_endo'] = dice
        frame_info_dict['hd_endo'] = hd
        frames_details[img_path] = frame_info_dict

        dices_endo.append(dice)
        hds_endo.append(hd)

    avg_dice_endo = np.mean(np.asarray(dices_endo))
    avg_hd_endo = np.mean(np.asarray(hds_endo))

    frames_details['avg_dice_endo'] = float(avg_dice_endo)
    frames_details['avg_hd_endo'] = float(avg_hd_endo)

    file_name = os.path.join(save_folder, f'{inference_name}.json')
    with open(file_name, 'w') as fp:
        json.dump(frames_details, fp)

    avg_score_dict_endo = {'Dice' : avg_dice_endo,
                           'HD' : avg_hd_endo}

    frames_details.clear()

    return avg_score_dict_endo


def infer_test_segmentation_quick(model, test_images, label_masks, save_folder, mask_shape, batch_size, inference_name):
    """
    Perform a quicker segmentation inference on test images with minimal output saving for faster execution.

    Args:
        model: Trained model for segmentation inference.
        test_images (list): List of paths to the test images.
        label_masks (list): List of ground truth label masks corresponding to the test images.
        save_folder (str): Path to the folder where results will be saved.
        mask_shape (tuple): Shape of the predicted mask (height, width, channels).
        batch_size (int): Number of images processed in each batch.
        inference_name (str): Name used for saving the inference results.

    Returns:
        dict: Dictionary containing average dice and Hausdorff distance scores.
    """

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    examples_path = os.path.join(save_folder, f'{inference_name} examples')
    if not os.path.exists(examples_path):
        os.makedirs(examples_path)

    frames_list = test_images
    all_frames_masks, _ = infer_images_in_batches_segmentation(model,
                                                                frames_list,
                                                                mask_shape,
                                                                batch_size)
    assert len(all_frames_masks) == len(frames_list)

    # Save all predictions in example folder
    all_frames_masks_contours = []
    for i in range(len(frames_list)):
        image_file = frames_list[i]
        label_mask = np.array(label_masks[i])

        pred_mask = np.array(all_frames_masks[i])
        pred_mask = pred_mask[: ,: ,0]
        pred_mask = tf.greater(pred_mask, 0.5)
        pred_mask = tf.dtypes.cast(pred_mask, tf.uint8)
        pred_mask = pred_mask.numpy()

        if label_mask.shape[0] != pred_mask.shape[0] or label_mask.shape[1] != pred_mask.shape[1]:
            label_mask = label_mask[:, :, np.newaxis]
            label_mask_resize = tf.image.resize(label_mask, (pred_mask.shape[0], pred_mask.shape[1]))
            label_mask = label_mask_resize.numpy().astype(np.uint8)
            label_mask = label_mask[: ,: ,0]

        image_rgb = cv2.imread(image_file, cv2.IMREAD_COLOR)
        mask_rgb = get_rgb_image(pred_mask, (int(0) ,int(255) ,int(0)))

        if image_rgb.shape[0] != pred_mask.shape[0] or image_rgb.shape[1] != pred_mask.shape[1]:
            image_rgb = tf.image.resize(image_rgb, (pred_mask.shape[0], pred_mask.shape[1]))
            image_rgb = image_rgb.numpy().astype(np.uint8)

        img_curves, poly_pointsp = get_overlay_image(image_rgb, label_mask, pred_mask)
        out_file = os.path.join(examples_path, os.path.basename(image_file).replace('.png' ,'' )+ '_2.png')
        #cv2.imwrite(out_file, img_curves)

    # Score predictions. Scores to be used to select pseudo-labels later on.
    # Score per frame
    frames_indexes = list(range(1, len(frames_list ) +1))
    dices_endo = []
    hds_endo = []
    frames_details = {}
    for i in range(len(frames_list)):
        img_path = frames_list[i]

        pred_mask = all_frames_masks[i]

        label_mask = label_masks[i]

        pred_mask = tf.greater(pred_mask[: ,: ,0], 0.5)
        pred_mask = tf.dtypes.cast(pred_mask, tf.float32)
        pred_mask = pred_mask.numpy()

        if label_mask.shape[0] != pred_mask.shape[0] or label_mask.shape[1] != pred_mask.shape[1]:
            label_mask = label_mask[:, :, np.newaxis]
            label_mask_resize = tf.image.resize(label_mask, (pred_mask.shape[0], pred_mask.shape[1]))
            label_mask = label_mask_resize.numpy().astype(np.uint8)
            label_mask = label_mask[: ,: ,0]

        dice_endo, hd_endo = get_dice_and_hd_from_masks(label_mask, pred_mask)

        dice, hd = dice_endo, hd_endo

        frame_info_dict = {}
        frame_info_dict['dice_endo'] = dice
        frame_info_dict['hd_endo'] = hd
        frames_details[img_path] = frame_info_dict

        dices_endo.append(dice)
        hds_endo.append(hd)

    avg_dice_endo = np.mean(np.asarray(dices_endo))
    avg_hd_endo = np.mean(np.asarray(hds_endo))

    frames_details['avg_dice_endo'] = float(avg_dice_endo)
    frames_details['avg_hd_endo'] = float(avg_hd_endo)

    file_name = os.path.join(save_folder, f'{inference_name}.json')
    with open(file_name, 'w') as fp:
       json.dump(frames_details, fp)

    avg_score_dict_endo = {'Dice' : avg_dice_endo,
                           'HD' : avg_hd_endo}

    frames_details.clear()

    return avg_score_dict_endo


def prepare_inference_dataset(images, batch_size):
    """
    Prepare a TensorFlow dataset for inference, including preprocessing, caching, batching, and prefetching.

    Args:
        images (list): List of image paths to include in the dataset.
        batch_size (int): Number of images per batch.

    Returns:
        tf.data.Dataset: Prepared TensorFlow dataset.
    """

    ds = tf.data.Dataset.from_tensor_slices((images))
    dataset = ds.map(process_image_only, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def infer_images_in_batches_segmentation(model, images, mask_shape, batch_size):
    """
    Perform segmentation inference on a batch of images and return predicted masks.

    Args:
        model: Trained model for segmentation inference.
        images (list): List of paths to the images for inference.
        mask_shape (tuple): Shape of the predicted mask (height, width, channels).
        batch_size (int): Number of images processed in each batch.

    Returns:
        tuple: A tuple containing:
            - all_frames_masks (np.array): Array of predicted masks for all frames.
            - curve_points_per_frame (list): List of curve points per frame (currently unused).
    """

    curve_points_per_frame = []

    inference_dataset = prepare_inference_dataset(images, batch_size=batch_size)

    all_frames_masks = np.zeros((len(images), mask_shape[0], mask_shape[1], mask_shape[2]))

    offset_index = 0
    # loop through the video frames batch by batch
    for batch in inference_dataset:
        predicted_masks_batch = model.predict(batch)

        # loop through the predictions. This will generally have
        # shape (batchsize, 320,320,8) or if the batch size is larger than
        # the number of frames i.e., len(frames_list) then the shape will be
        # (len(frames_list), 320,320,8)
        # Hence we iterate by predicted_masks_batch.shape[0] which is the number
        # of batch predictions computed.
        for i in range(predicted_masks_batch.shape[0]):
            masks = predicted_masks_batch[i, :, :, :]
            # I assume the image for this mask is in order in frames_list
            # and that there is no shuffle when I prepared the dataset.
            image_index = i + offset_index
            all_frames_masks[image_index] = masks

            single_mask = masks[:, :, 0]  # Theres only the LV now

        offset_index += predicted_masks_batch.shape[0]

    return all_frames_masks, curve_points_per_frame