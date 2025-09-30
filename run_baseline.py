import random

import tensorflow_datasets as tfds

from inference import infer_test_segmentation
from architecture_unet import *
from plotting import *
from utils import *
from data_utils import *
from loss_funcs import *
import shutil

import time

#import pydevd

tfds.disable_progress_bar()

# Hyperparameters for the dataset
IMAGE_SIZE = 512
IMAGE_CHANNELS = 1

images_folder = 'Images'  # The labelled images
labels_folder = 'Labels'  # The expert labels/annotations

LOSS_FUNCTIONS = ['bce', 'dice', 'bce_dice']
LOSS = LOSS_FUNCTIONS[0]


def display_training_data(exp_folder, training_dataset):
    image_batch, label_batch = next(iter(training_dataset))

    # if my batch size was 4, then this will be (4,320,320,3)
    print(image_batch.shape)

    image_batch = image_batch.numpy()
    label_batch = tf.dtypes.cast(label_batch, tf.uint8)
    label_batch = label_batch.numpy()


    for image_idx in range(image_batch.shape[0]):

        image = image_batch[image_idx]
        masks = label_batch[image_idx]

        plt.figure(figsize=(10, 10), dpi=100)
        i = 0
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image)

        for i in range(1, masks.shape[2] + 1):
            idx = i - 1
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(masks[:, :, idx])

        #plt.show()

        plt.savefig(os.path.join(exp_folder, 'images_and_label_examples.png'))
        plt.clf()

        break

def get_unet_model():
    model = UNET((IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS), IMAGE_CHANNELS, dropout = 0, is_segmentation=True)
    return model

def get_loss():
    loss = None
    if LOSS == LOSS_FUNCTIONS[0]:
        loss = keras.losses.BinaryCrossentropy()
    elif LOSS == LOSS_FUNCTIONS[1]:
        loss_funcs = Semantic_loss_functions()
        loss = loss_funcs.dice_loss
    else:
        loss_funcs = Semantic_loss_functions()
        loss = loss_funcs.bce_dice_loss

    return loss


class LearningRateTracker(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        _lr = tf.keras.backend.get_value(optimizer.lr)
        print(f'\nEpoch {epoch+1}: Learning rate is {_lr:.5f}.')


def train_baseline_model(baseline_path, train_dataset, val_dataset,
                         num_epochs, batch_size, learning_rate,
                         num_train):
    learning_curve_path = os.path.join(baseline_path, 'learning_curve.png')
    csv_save_path = os.path.join(baseline_path, 'epoch_history.csv')
    model_save_path = os.path.join(baseline_path, 'model_weights.h5')
    model_plot_name = os.path.join(baseline_path, 'model.png')

    if not os.path.exists(baseline_path):
        os.makedirs(baseline_path)

    # Baseline supervised training with random initialization
    model = get_unet_model()
    model._name="Baseline_Model"

    if learning_rate is None:
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy())
    else:
        epochs = 100
        initial_learning_rate = 1e-4
        final_learning_rate = learning_rate
        learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / epochs)
        steps_per_epoch = int(num_train / batch_size)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=steps_per_epoch,
            decay_rate=learning_rate_decay_factor,
            staircase=True)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                      loss=get_loss())
                      #loss=keras.losses.BinaryCrossentropy())

    plot_model(model, model_plot_name, False)

    callbks = []
    # model checkpoint
    model_chpt = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    save_freq='epoch',
                                                    #period=5
                                                    )
    callbks.append(model_chpt)

    # csv logger
    csv_logger = tf.keras.callbacks.CSVLogger(csv_save_path)
    callbks.append(csv_logger)

    #start_from_epoch = 40  #for BCE loss
    start_from_epoch = 100  #for dice loss
    if num_train < 50:
        start_from_epoch = 100
    if num_train <= 20:          ##train fewer images for longer, more stable
        start_from_epoch = 150

    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=start_from_epoch,
                                                     restore_best_weights=True)
    callbks.append(early_stopper)

    callbks.append(LearningRateTracker())

    history = model.fit(train_dataset,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        callbacks=callbks,
                        validation_data=val_dataset
                        )

    plot_loss_per_epoch_skip(history, learning_curve_path, 10)

    return history

def test_baseline_model(baseline_path, model,
                        test_images, test_labels,
                        consensus_a2c_images, consensus_a2c_labels,
                        batch_size_infer):

    label_masks = []
    for label_file in test_labels:
        mask = cv2.imread(label_file)
        label_masks.append(mask[:, :, 0])

    avg_score_dict_endo = infer_test_segmentation(model, test_images, label_masks, baseline_path,
                                                  (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS), batch_size_infer,
                                                  'Test_dataset_results_all')

    write_dict_to_json({'Avg Dice': avg_score_dict_endo['Dice'], 'Avg HD':avg_score_dict_endo['HD']},
                       baseline_path, 'Test_dataset_results.json')

    print(f"Baseline Avg Dice: {avg_score_dict_endo['Dice']}")
    print(f"Baseline Avg HD: {avg_score_dict_endo['HD']}")

    consensus_label_masks = []
    for label_file in consensus_a2c_labels:
        mask = cv2.imread(label_file)
        consensus_label_masks.append(mask[:, :, 0])
    avg_score_dict_endo1 = infer_test_segmentation(model, consensus_a2c_images, consensus_label_masks, baseline_path,
                                                   (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS), batch_size_infer,
                                                   'Consensus_dataset_results_all')

    print(f"Baseline-consensus Avg Dice: {avg_score_dict_endo1['Dice']}")
    print(f"Baseline-consensus Avg HD: {avg_score_dict_endo1['HD']}")

    # get consensus scores for top 100 labels where the experts were in closest agreement.
    all_scores_dict = load_json(os.path.join(baseline_path, f'Consensus_dataset_results_all.json'))
    consensus_rankings = load_json(f'consensus rankings.json')
    dices_top100, hds_top100 = [], []
    rank_count = 0
    dataset_path = consensus_a2c_images[0]
    dataset_path = dataset_path[0:dataset_path.rindex('/')]
    for key, value in consensus_rankings.items():
        score_info = all_scores_dict[os.path.join(dataset_path, key)]
        dices_top100.append(score_info['dice_endo'])
        hds_top100.append(score_info['hd_endo'])
        rank_count += 1
        if (rank_count == 100):
            break

    avg_dice_top100 = np.mean(dices_top100)
    avg_hd_top100 = np.mean(hds_top100)
    print(f"Top100-consensus Avg Dice: {avg_dice_top100}")
    print(f"Top100-consensus Avg HD: {avg_hd_top100}")

    write_dict_to_json({'Avg Dice': avg_score_dict_endo1['Dice'], 'Avg HD': avg_score_dict_endo1['HD'],
                        'Avg Dice top100': avg_dice_top100, 'Avg HD top100': avg_hd_top100},
                       baseline_path, 'Consensus_dataset_results.json')

    return ([avg_score_dict_endo['Dice'], avg_score_dict_endo1['Dice'], avg_dice_top100],
            [avg_score_dict_endo['HD'], avg_score_dict_endo1['HD'], avg_hd_top100])


def establish_baseline_scores(baseline_folder, exp_folder, batch_size, percentage_labelled,
                              training_dataset, validation_dataset,
                              test_images, test_labels,
                              consensus_a2c_images, consensus_a2c_labels, consensus_dataset_folder,
                              num_train, num_val, num_test, num_consensus,
                              num_runs, train_baseline, test_baseline,
                              batch_size_infer):
    # BASELINE SUPERVISED TRAINING WITH RANDOM INITIALISATION

    dices_test, hd_test = [],[]
    dices_con, hd_con = [], []
    run_losses, run_epochs = [], []
    run_val_losses, run_val_epochs = [], []
    run_dices_consensus_top_100_expert_agreement = []
    run_hds_consensus_top_100_expert_agreement = []
    ef_errors, ed_errors, es_errors, count_no_efs = [], [], [], []
    training_times = []
    for i in range(num_runs):

        run_path = os.path.join(exp_folder, f'baseline_model_results_run_{i}')
        if not os.path.exists(run_path):
            os.makedirs(run_path)

        if train_baseline:
            start = time.time()
            history = train_baseline_model(run_path, training_dataset,
                                 validation_dataset,
                                 num_epochs=500,
                                 batch_size=batch_size,
                                 learning_rate=1e-5, num_train=num_train)
            end = time.time()
            ellapsed_time = end - start

            loss = np.array(history.history['loss']).astype(float)
            val_loss = np.array(history.history['val_loss']).astype(float)
            run_losses.append(np.min(loss))
            run_epochs.append(int(np.argmin(loss)) + 1)  # 0-based index but epoch is 1-based
            run_val_losses.append(np.min(val_loss))
            run_val_epochs.append(int(np.argmin(val_loss)) + 1)
            training_times.append(ellapsed_time)

        # TEST BASELINE
        if test_baseline:
            model = get_unet_model()
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                          loss=keras.losses.BinaryCrossentropy())
            model.load_weights(os.path.join(run_path, 'model_weights.h5')) #best model weights
            dices, hds = test_baseline_model(run_path, model, test_images, test_labels,
                                             consensus_a2c_images, consensus_a2c_labels,
                                             batch_size_infer)

            baseline_results_dict = {'dice_test': dices[0], 'hd_test': hds[0],
                                     'dice_consensus': dices[1], 'hd_consensus': hds[1],
                                     'dice_consensus_top100': dices[2], 'hd_consensus_top100': hds[2],
                                     'num_train': num_train, 'num_val': num_val,
                                     'num_test': num_test, 'num_consensus': len(consensus_a2c_images)}

            write_dict_to_json(baseline_results_dict, exp_folder,
                               f'baseline_results_run_{i}.json')

            #calculate ejection fraction for consensus
            (avg_ef_error,
             avg_ED_vol_error,
             avg_ES_vol_error,
             count_no_ef) = compute_ejection_fraction_consensus_set_for_baseline(run_path, consensus_dataset_folder)

            dices_test.append(dices[0])
            dices_con.append(dices[1])
            hd_test.append(hds[0])
            hd_con.append(hds[1])
            run_dices_consensus_top_100_expert_agreement.append(dices[2])
            run_hds_consensus_top_100_expert_agreement.append(hds[2])
            ef_errors.append(avg_ef_error)
            ed_errors.append(avg_ED_vol_error)
            es_errors.append(avg_ES_vol_error)
            count_no_efs.append(count_no_ef)

    results_dict = {'run_dices_test': dices_test, 'run_hds_test': hd_test,
                      'run_dices_consensus': dices_con, 'run_hds_consensus': hd_con,
                      'run_dices_consensus_top100': run_dices_consensus_top_100_expert_agreement,
                      'run_hds_consensus_top100': run_hds_consensus_top_100_expert_agreement,
                      'avg_ef_errors_per_run': ef_errors,
                      'avg_ed_vol_errors_per_run': ed_errors,
                      'avg_es_vol_errors_per_run': es_errors,
                      'count_no_ef': count_no_efs,
                      'num_train': num_train, 'percentage_train': percentage_labelled, 'num_val': num_val,
                      'num_test': num_test, 'num_consensus': num_consensus,
                      'run_losses': run_losses, 'run_epochs': run_epochs,
                      'run_val_losses': run_val_losses, 'run_val_epochs': run_val_epochs,
                      'training_times': training_times}
    write_dict_to_json(results_dict, baseline_folder,
                       f'baseline_results_{percentage_labelled}.json')



def run_baseline(baseline_folder, experiment_folder,
                 dataset_folder, consensus_dataset_folder,
                 percentage_labelled, runs, train=True, test=True,
                 representative = False):
    tf.config.run_functions_eagerly(False)

    if not os.path.exists(baseline_folder):
        os.makedirs(baseline_folder)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    gpus = ["/gpu:5"]
    single_gpu_labelled_batch_size = 8
    batch_size_labelled = single_gpu_labelled_batch_size * len(gpus)

    if not representative:
        (training_dataset, validation_dataset, training_images, training_labels, val_images, val_labels,
         test_images, test_labels, num_train, num_val, num_test) = load_labelled_datasets_unity(dataset_folder,
                                                                                                batch_size_labelled,
                                                                                                percentage_labelled)
    else:
        (training_dataset, validation_dataset, training_images, training_labels, val_images, val_labels,
         test_images, test_labels, num_train, num_val, num_test) = load_labelled_datasets_unity_representative(
            dataset_folder, 'classifier_output.csv',
            batch_size_labelled,
            percentage_labelled)

    display_training_data(experiment_folder, training_dataset)

    consensus_a2c_images, consensus_a2c_labels = get_list_of_images_and_labels_from_folder(
        consensus_dataset_folder,
        images_folder,
        labels_folder,
        False)

    if num_train <= 500:
        gpus = ["/gpu:5"]
        single_gpu_labelled_batch_size = 4
    batch_size_labelled = single_gpu_labelled_batch_size * len(gpus)
    batch_size_infer = len(gpus) * 512

    if not representative:
        (training_dataset, validation_dataset, training_images, training_labels, val_images, val_labels,
         test_images, test_labels, num_train, num_val, num_test) = load_labelled_datasets_unity(dataset_folder,
                                                                                                batch_size_labelled,
                                                                                                percentage_labelled)
    else:
        (training_dataset, validation_dataset, training_images, training_labels, val_images, val_labels,
         test_images, test_labels, num_train, num_val, num_test) = load_labelled_datasets_unity_representative(
            dataset_folder, 'classifier_output.csv',
            batch_size_labelled,
            percentage_labelled)

    write_dict_to_json({'train': num_train, 'val': num_val, 'test': num_test,
                        'consensus': len(consensus_a2c_images), 'perc_train': percentage_labelled,
                        'gpus':gpus, 'batch size':batch_size_labelled},
                       experiment_folder, f'labelled_dataset_counts.json')

    # Create a MirroredStrategy:
    strategy = tf.distribute.MirroredStrategy(devices=gpus)
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    print("-------------------------")

    with strategy.scope():
        establish_baseline_scores(baseline_folder, experiment_folder, batch_size_labelled, percentage_labelled,
                                  training_dataset, validation_dataset,
                                  test_images, test_labels,
                                  consensus_a2c_images, consensus_a2c_labels, consensus_dataset_folder,
                                  num_train, num_val, num_test, len(consensus_a2c_images),
                                  runs, train, test, batch_size_infer)

def main():
    #tf.config.run_functions_eagerly(True) for debugging step function

    main_folder = '.'

    dataset_folder = r'Data/unity final cleaning/'
    consensus_dataset_folder = 'Data/Expert Consensus dataset'

    tf.random.set_seed(777)
    np.random.seed(555)
    random.seed(444)

    global LOSS
    LOSS = LOSS_FUNCTIONS[0]

    representative = False

    train_baseline = True
    test_baseline = True

    percentages_for_downstream_training = [1, 2, 3, 4, 5, 10, 15, 25, 100]
    num_baseline_runs = 2

    # Baseline supervised training with randomly initialised weights
    if train_baseline or test_baseline:
        for perc_labelled in percentages_for_downstream_training:
            tf.random.set_seed(777)
            np.random.seed(555)
            random.seed(444)
            baseline_folder = os.path.join(main_folder, 'Baseline_444_Final')
            if representative:
                baseline_folder+='leastrep'
            experiment_folder = os.path.join(baseline_folder, f'Baseline_with_{perc_labelled}_percent_labelled_data')
            run_baseline(baseline_folder=baseline_folder, experiment_folder=experiment_folder,
                         dataset_folder=dataset_folder, consensus_dataset_folder=consensus_dataset_folder,
                         percentage_labelled=perc_labelled, runs=num_baseline_runs, train=train_baseline,
                         test=test_baseline, representative = representative)

    print('Complete.')

if __name__ == "__main__":
    main()