import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from lr_cosine_schedule import WarmUpCosine
from augmentation_util import *
from architecture_unet import *
from loss_funcs import *
from utils import *
from data_utils import *
from inference import infer_test_segmentation
import time

tfds.disable_progress_bar()

# Hyperparameters for the dataset
IMAGE_SIZE = 512
IMAGE_CHANNELS = 1

# Hyperparameters for the algorithm
NETWORK_WIDTH = 128
TEMPERATURE = 0.1
LARGE_NUM = 1e9


# Augmentations for contrastive and supervised training
AUG_CONTRASTIVE = {
    "image_size": IMAGE_SIZE,
    "image_channels": IMAGE_CHANNELS,
    "min_area": 0.6,#0.25,
    "brightness": 0.6,
    "jitter": 0.2
}
AUG_SEGMENTATION = {
    "image_size": IMAGE_SIZE,
    "image_channels": IMAGE_CHANNELS,
    "min_area": 0.75,
    "brightness": 0.3,
    "jitter": 0.1
}

AUTOTUNE = tf.data.AUTOTUNE

BACKBONES = ['unet-enc', 'unet']
BACKBONE = BACKBONES[0]

LOSS_FUNCTIONS = ['bce', 'dice']
LOSS = LOSS_FUNCTIONS[0]
lOCK_ENC_NORM_LAYERS = True

def set_backbone(idx):
    global BACKBONE
    BACKBONE = BACKBONES[idx]

def get_encoder_for_pretraining():
    enc = None
    if BACKBONE == BACKBONES[0]:
        enc,_ = get_unet_encoder((IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS), dropout_prob=0)
    elif BACKBONE == BACKBONES[1]:
        enc = get_unet_encoder_decoder((IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS), dropout_prob=0)
    return enc

def get_model_for_downstream_finetuning(model_file):
    finetuning_model = None
    if BACKBONE == BACKBONES[0]:
        encoder = tf.keras.models.load_model(model_file)

        transfer_enc = UNET(input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS),
                            num_output_channels=IMAGE_CHANNELS, dropout=0, is_segmentation=True)

        for i in range(len(encoder.layers)):
            from_layer = encoder.layers[i]
            to_layer = transfer_enc.layers[i]

            if isinstance(from_layer, tf.keras.layers.Conv2DTranspose):
                break

            if type(to_layer) == type(from_layer):
                to_layer.set_weights(from_layer.get_weights())

        finetuning_model = transfer_enc

    elif BACKBONE == BACKBONES[1]:
        encoder = tf.keras.models.load_model(model_file)

        transfer_enc = UNET(input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS),
                            num_output_channels=IMAGE_CHANNELS, dropout=0, is_segmentation=True)

        # -1 transfer all weights except for the output layer
        for i in range(len(encoder.layers)-1):
            from_layer = encoder.layers[i]
            to_layer = transfer_enc.layers[i]

            if type(to_layer) == type(from_layer):
                to_layer.set_weights(from_layer.get_weights())

        finetuning_model = transfer_enc

    return finetuning_model

def get_model_for_downstream_inference(model_file):
    model = None
    if BACKBONE == BACKBONES[0]:
        model = UNET(input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS),
                     num_output_channels=IMAGE_CHANNELS, dropout=0, is_segmentation=True)
        model.load_weights(model_file, by_name=True)
    elif BACKBONE == BACKBONES[1]:
        model = UNET(input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS),
                     num_output_channels=IMAGE_CHANNELS, dropout=0, is_segmentation=True)
        model.load_weights(model_file, by_name=True)

    return model

def get_loss():
    loss = None
    if LOSS == LOSS_FUNCTIONS[0]:
        loss = keras.losses.BinaryCrossentropy()
    else:
        loss_funcs = Semantic_loss_functions()
        loss = loss_funcs.dice_loss
    return loss

class ContrastiveModel(keras.Model):
    def __init__(self):
        super().__init__()

        self.temperature = TEMPERATURE
        self.contrastive_augmenter = get_augmenter(**AUG_CONTRASTIVE) #**AUG_SEGMENTATION
        self.encoder = get_encoder_for_pretraining()

        # Non-linear MLP as projection head
        encoder_output_shape = self.encoder.output_shape
        encoder_output_shape = (encoder_output_shape[1], encoder_output_shape[2], encoder_output_shape[3])

        self.projection_head = keras.Sequential(
            [
                keras.Input(shape=encoder_output_shape),  # output shape of the encoder
                layers.Flatten(),
                layers.Dense(NETWORK_WIDTH, activation="relu", dtype=tf.float32),
                layers.Dense(NETWORK_WIDTH, dtype=tf.float32),
            ],
            name="Project_Head",
        )

        self.encoder.summary()
        self.projection_head.summary()

    def compile(self, contrastive_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.contrastive_loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
        ]

    def contrastive_loss(self, projections_1, projections_2):

        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        hidden1, hidden2 = projections_1, projections_2
        batch_size = tf.shape(hidden1)[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)
        logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / TEMPERATURE
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / TEMPERATURE
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / TEMPERATURE
        logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / TEMPERATURE
        loss_a = tf.nn.softmax_cross_entropy_with_logits(
            labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = tf.nn.softmax_cross_entropy_with_logits(
            labels, tf.concat([logits_ba, logits_bb], 1))
        loss = tf.reduce_mean(loss_a + loss_b)

        return loss

    def train_step(self, data):
        images = data #first axis is the batch

        # Each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images, training=True)
        augmented_images_2 = self.contrastive_augmenter(images, training=True)

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)
            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)

        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images = data

        augmented_images_1 = self.contrastive_augmenter(images, training=False)
        augmented_images_2 = self.contrastive_augmenter(images, training=False)

        features_1 = self.encoder(augmented_images_1, training=False)
        features_2 = self.encoder(augmented_images_2, training=False)

        projections_1 = self.projection_head(features_1, training=False)
        projections_2 = self.projection_head(features_2, training=False)
        contrastive_loss = self.contrastive_loss(projections_1, projections_2)

        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        print(f'\n **SAVED MODEL: f{filepath} \n')
        self.encoder.save(filepath, overwrite, save_format)

    def save_weights(self, filepath, overwrite=True, save_format=None, **kwargs):
        print(f'\n **SAVED MODEL: f{filepath} \n')
        self.encoder.save(filepath, overwrite, save_format)


def contrastive_pretraining(exp_folder, train_dataset, unlabelled_val, num_unlabelled, batch_size_unlabelled, num_epochs):
    contrastive_training_path = os.path.join(exp_folder, 'contrastive_training')
    learning_curve_path = os.path.join(contrastive_training_path, 'learning_curve.png')
    csv_save_path = os.path.join(contrastive_training_path, 'epoch_history.csv')
    model_save_path_encoder_weights = os.path.join(contrastive_training_path, 'best_encoder_weights.h5')
    model_save_path_encoder = os.path.join(contrastive_training_path, 'best_encoder.h5')
    model_save_path_epoch = os.path.join(contrastive_training_path, 'model_ep_{epoch}.h5')

    if not os.path.exists(contrastive_training_path):
        os.makedirs(contrastive_training_path)

    callbks = []
    # csv logger
    csv_logger = tf.keras.callbacks.CSVLogger(csv_save_path)
    callbks.append(csv_logger)

    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     patience=10, start_from_epoch=50,
                                                     restore_best_weights=False)
    callbks.append(early_stopper)

    # model checkpoint to save best weights
    model_chpt = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path_encoder_weights,
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    save_freq='epoch',
                                                    #period=5
                                                    )
    callbks.append(model_chpt)

    # model checkpoint to save best weights
    model_chpt1 = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path_encoder,
                                                     monitor='val_loss',
                                                     verbose=1,
                                                     save_weights_only=False,
                                                     save_best_only=True,
                                                     save_freq='epoch',
                                                     # period=5
                                                     )
    callbks.append(model_chpt1)

    # COSINE DECAY
    EPOCHS = 100
    STEPS_PER_EPOCH = num_unlabelled // batch_size_unlabelled
    TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
    WARMUP_EPOCHS = int(EPOCHS * 0.10)
    WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)
    lr_decayed_fn = WarmUpCosine(
        learning_rate_base=1e-3,
        total_steps=EPOCHS * STEPS_PER_EPOCH,
        warmup_learning_rate=0.0,
        warmup_steps=WARMUP_STEPS
    )

    # Contrastive pretraining
    pretraining_model = ContrastiveModel()
    pretraining_model.compile(
            contrastive_optimizer=keras.optimizers.Adam(learning_rate=lr_decayed_fn)
        )

    start = time.time()
    pretraining_history = pretraining_model.fit(
        train_dataset, epochs=num_epochs, validation_data=unlabelled_val, callbacks=callbks
    )
    end = time.time()
    ellapsed_time = end - start

    plot_loss_per_epoch_skip(pretraining_history, learning_curve_path, 10)

    pretraining_model.encoder.save_weights(os.path.join(contrastive_training_path, f'last_epoch_encoder_weights.h5'))
    pretraining_model.encoder.save(os.path.join(contrastive_training_path, f'last_epoch_encoder.h5'))

    return pretraining_history, pretraining_model, ellapsed_time

def train_with_fine_tuning(model, model_file, train_dataset, val_dataset,
                           num_train, num_epochs, batch_size, learning_rate,
                           fine_tuning_path, index,
                           percentage_unlabelled, percentage_labelled):
    baseline_path = fine_tuning_path
    learning_curve_path = os.path.join(baseline_path, f'learning_curve_{index}.png')
    csv_save_path = os.path.join(baseline_path, f'epoch_history_{index}.csv')
    model_save_path = os.path.join(baseline_path, f'best_finetuned_model_{index}.h5')
    model_weights_save_path = os.path.join(baseline_path, f'best_finetuned_model_weights_{index}.h5')
    model_plot_name = os.path.join(baseline_path, 'model.png')

    if not os.path.exists(baseline_path):
        os.makedirs(baseline_path)

    if learning_rate is None:
        model.compile(optimizer=keras.optimizers.Adam(), loss=get_loss())
    else:
        #COSINE DECAY
        EPOCHS = 120
        start_from = 100

        if 15 <= percentage_labelled < 20:
            EPOCHS = 110
        elif percentage_labelled == 20:
            EPOCHS = 100
            start_from = 80
        elif percentage_labelled == 25:
            EPOCHS = 90
            start_from = 60
        elif percentage_labelled == 50:
            EPOCHS = 45
            start_from=40
        elif percentage_labelled == 100:
            EPOCHS = 30
            start_from=30

        STEPS_PER_EPOCH = num_train // batch_size
        TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
        WARMUP_EPOCHS = int(EPOCHS * 0.10)
        WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)

        lr_decayed_fn = WarmUpCosine(
            learning_rate_base=1e-4,  #1e-4 normal, #1e-3 is for unity frames
            total_steps=EPOCHS * STEPS_PER_EPOCH,
            warmup_learning_rate=0.0,
            warmup_steps=WARMUP_STEPS
        )

        x, y =[],[]
        cnt=1
        for l in range(1, EPOCHS * STEPS_PER_EPOCH):
            val = lr_decayed_fn(l)
            val = val.numpy()
            if l % STEPS_PER_EPOCH == 0:
                x.append(cnt)
                y.append(val)
                cnt+=1

        plt.plot(x, y)
        plt.ylabel("Learning Rate")
        plt.xlabel("Train Step")
        plt.savefig(os.path.join(baseline_path, 'Learning rate.png'))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_decayed_fn),
                     loss=get_loss())

    plot_model(model, model_plot_name, False)


    callbks = []
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=start_from,
                                                     restore_best_weights=False)
    callbks.append(early_stopper)

    model_chpt1 = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_weights_only=False,
                                                    save_best_only=True,
                                                    save_freq='epoch',
                                                    #period=2
                                                     )
    callbks.append(model_chpt1)

    model_chpt2 = tf.keras.callbacks.ModelCheckpoint(filepath=model_weights_save_path,
                                                     monitor='val_loss',
                                                     verbose=1,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     save_freq='epoch',
                                                     # period=2
                                                     )
    callbks.append(model_chpt2)

    # csv logger
    csv_logger = tf.keras.callbacks.CSVLogger(csv_save_path)
    callbks.append(csv_logger)

    history = model.fit(train_dataset,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        callbacks=callbks,
                        validation_data=val_dataset
                        )

    plot_loss_per_epoch_skip(history, learning_curve_path, skip=10)
    model.save_weights(os.path.join(baseline_path, f'last_epoch_finetuned_model_weights_{index}.h5'))

    return history

def test_fine_tuned_model(folder_path, model, test_images, test_labels, test_name):
    baseline_path = folder_path

    if not os.path.exists(baseline_path):
        os.makedirs(baseline_path)

    label_masks = []
    for label_file in test_labels:
        mask = cv2.imread(label_file)
        label_masks.append(mask[:, :, 0])

    inference_batch_size = 512
    if BACKBONE == BACKBONES[1]:
        inference_batch_size = 20

    #use infer_test_segmentation_quick for testing purposes
    avg_score_dict_endo = infer_test_segmentation(model, test_images, label_masks, save_folder=baseline_path,
                                                  mask_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS), batch_size=inference_batch_size,
                                                  inference_name=test_name)

    json_name = test_name.replace('all', 'avg')
    write_dict_to_json({'Avg Dice': avg_score_dict_endo['Dice'], 'Avg HD':avg_score_dict_endo['HD']},
                       baseline_path, f"{json_name}.json")

    print(f"{json_name} Avg Dice: {avg_score_dict_endo['Dice']}")
    print(f"{json_name} Avg HD: {avg_score_dict_endo['HD']}")

    return avg_score_dict_endo['Dice'], avg_score_dict_endo['HD']


def test_pretrained_model_exp(exp_folder, training_dataset, validation_dataset,
                                 test_images, test_labels,
                                 consensus_a2c_images, consensus_a2c_labels,
                                 num_unlabelled, percentage_for_unlabelled_training,
                                 num_train, num_val, num_test, num_consensus,
                                 num_fine_tuning_runs, batch_size,
                                 train_fine_tuned_model, test_fine_tune, percentage_labelled, gpus):
    plt.clf()
    models_path = os.path.join(exp_folder, 'contrastive_training')
    #Either use one of the 4
    model_file_path_encoder = os.path.join(models_path, 'best_encoder.h5')
    model_file_path_encoder_weights = os.path.join(models_path, 'best_encoder_weights.h5')
    file_path_encoder_weights_last_epoch = os.path.join(models_path, f'last_epoch_encoder_weights.h5')
    file_path_encoder_last_epoch = os.path.join(models_path, f'last_epoch_encoder.h5')

    main_fine_tuning_path = os.path.join(exp_folder, f'finetuning_{percentage_labelled}')
    if not os.path.exists(main_fine_tuning_path):
        os.makedirs(main_fine_tuning_path)

    run_dices_test, run_dices_consensus = [], []
    run_hds_test, run_hds_consensus = [], []
    run_losses, run_epochs = [], []
    run_val_losses, run_val_epochs = [], []
    run_dices_consensus_top_100_expert_agreement = []
    run_hds_consensus_top_100_expert_agreement = []
    training_times = []

    # run fine-tuning a few times for a more accurate result
    for p in range(num_fine_tuning_runs):

        tf.keras.backend.clear_session()
        strategy = tf.distribute.MirroredStrategy(devices=gpus)
        with strategy.scope():

            if train_fine_tuned_model:
                finetuning_model = get_model_for_downstream_finetuning(model_file_path_encoder)

                start = time.time()
                history = train_with_fine_tuning(finetuning_model, model_file_path_encoder,
                                       training_dataset,
                                       validation_dataset,
                                       num_train,
                                       num_epochs=500,  # fine_tuning_epochs[i],   500 wih early stopping
                                       batch_size=batch_size,
                                       learning_rate=1e-5,
                                       fine_tuning_path = main_fine_tuning_path,
                                       index=p,
                                       percentage_unlabelled=percentage_for_unlabelled_training,
                                       percentage_labelled=percentage_labelled)

                end = time.time()
                ellapsed_time = end - start

                loss = np.array(history.history['loss']).astype(float)
                val_loss = np.array(history.history['val_loss']).astype(float)
                run_losses.append(np.min(loss))
                run_epochs.append(int(np.argmin(loss)) + 1)  #0-based index but epoch is 1-based
                run_val_losses.append(np.min(val_loss))
                run_val_epochs.append(int(np.argmin(val_loss)) + 1)
                training_times.append(ellapsed_time)

        tf.keras.backend.clear_session()
        strategy = tf.distribute.MirroredStrategy(devices=gpus)
        with strategy.scope():
            if test_fine_tune:

                fine_tuned_model_path = os.path.join(main_fine_tuning_path, f'best_finetuned_model_{p}.h5')
                model = tf.keras.models.load_model(fine_tuned_model_path, custom_objects={'WarmUpCosine': WarmUpCosine,
                                                                                          #'dice_loss':Semantic_loss_functions().dice_loss
                                                                                          })

                dice, hd = test_fine_tuned_model(main_fine_tuning_path, model,
                                                 test_images, test_labels,
                                                 test_name=f'Test_dataset_results_all_{p}')
                run_dices_test.append(dice)
                run_hds_test.append(hd)

                dice1, hd1 = test_fine_tuned_model(main_fine_tuning_path, model,
                                                   consensus_a2c_images, consensus_a2c_labels,
                                                   test_name=f'Consensus_dataset_results_all_{p}')
                run_dices_consensus.append(dice1)
                run_hds_consensus.append(hd1)

                #get consensus scores for top 100 labels where the experts were in closest agreement.
                all_scores_dict = load_json(os.path.join(main_fine_tuning_path, f'Consensus_dataset_results_all_{p}.json'))
                consensus_rankings = load_json(f'consensus rankings.json')
                dices_top100, hds_top100 = [], []
                rank_count=0
                dataset_path = consensus_a2c_images[0]
                dataset_path = dataset_path[0:dataset_path.rindex('/')]
                for key, value in consensus_rankings.items():
                    score_info = all_scores_dict[os.path.join(dataset_path, key)]
                    dices_top100.append(score_info['dice_endo'])
                    hds_top100.append(score_info['hd_endo'])
                    rank_count+=1
                    if(rank_count == 100):
                        break

                avg_dice_top100 = np.mean(dices_top100)
                avg_hd_top100 = np.mean(hds_top100)
                print(f"Consensus_top100_{p} Avg Dice: {avg_dice_top100}")
                print(f"Consensus_top100_{p} Avg HD: {avg_hd_top100}")

                run_dices_consensus_top_100_expert_agreement.append(avg_dice_top100)
                run_hds_consensus_top_100_expert_agreement.append(avg_hd_top100)

    fine_tune_results_dict = {'run_dices_test': run_dices_test, 'run_hds_test': run_hds_test,
                              'run_dices_consensus': run_dices_consensus, 'run_hds_consensus': run_hds_consensus,
                              'run_dices_consensus_top100': run_dices_consensus_top_100_expert_agreement,
                              'run_hds_consensus_top100': run_hds_consensus_top_100_expert_agreement,
                              'num_unlabelled': num_unlabelled, 'percentage_unlabelled': percentage_for_unlabelled_training,
                              'num_train': num_train, 'percentage_train': percentage_labelled, 'num_val': num_val,
                              'num_test': num_test, 'num_consensus': num_consensus,
                              'run_losses': run_losses, 'run_epochs': run_epochs,
                              'run_val_losses': run_val_losses, 'run_val_epochs': run_val_epochs,
                              'training_times': training_times}
    write_dict_to_json(fine_tune_results_dict, main_fine_tuning_path,
                       f'encoder_fine_tune_results.json')



def run_ssl(mixed_precision, single_gpu_unlabelled_batch_size, experiment_folder, unlabelled_dataset_path, percentage_unlabelled):

    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    mixed = mixed_precision
    print(f'The current precision policy is: {keras.mixed_precision.global_policy()}')
    if mixed:
        keras.mixed_precision.set_global_policy('mixed_float16')
        print(f'Precision policy changed to: {keras.mixed_precision.global_policy()}')

    files = os.listdir(unlabelled_dataset_path)
    num_unlabelled = len(files)

    if num_unlabelled <= 500:
        gpus = ["/gpu:5"]
    if num_unlabelled <= 2500:
        gpus = ["/gpu:3"]#, "/gpu:4"]
    else:
        gpus = ["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5"]

    batch_size_unlabelled = single_gpu_unlabelled_batch_size * len(gpus)

    (unlabelled_train_dataset,
     num_unlabelled,
     unlabelled_val,
     num_unlabelled_val,
     bal_frames) = load_unlabelled_datasets(unlabelled_dataset_path=unlabelled_dataset_path,
                                            batch_size=batch_size_unlabelled,
                                            num_devices=len(gpus),
                                            num_train=percentage_unlabelled)

    write_dict_to_json({'train': num_unlabelled, 'val': num_unlabelled_val,
                        'perc_train': percentage_unlabelled, 'gpus':gpus, 'batch_size':batch_size_unlabelled},
                       experiment_folder, f'unlabelled_dataset_counts.json')

    # Create a MirroredStrategy:
    strategy = tf.distribute.MirroredStrategy(devices=gpus)
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    print("-------------------------")
    with strategy.scope():
        contrastive_training_path = os.path.join(experiment_folder, 'contrastive_training')
        pretraining_history, pretraining_model, ellapsed_time = contrastive_pretraining(experiment_folder,
                                                                                        unlabelled_train_dataset,
                                                                                         unlabelled_val,
                                                                                         num_unlabelled,
                                                                                         batch_size_unlabelled,
                                                                                         num_epochs=200)

        loss = np.array(pretraining_history.history['loss']).astype(float)
        val_loss = np.array(pretraining_history.history['val_loss']).astype(float)
        run_loss = np.min(loss)
        run_epochs = int(np.argmin(loss)) + 1  # 0-based index but epoch is 1-based
        run_val_loss = np.min(val_loss)
        run_val_epochs = int(np.argmin(val_loss)) + 1

        write_dict_to_json({'train': num_unlabelled, 'val': num_unlabelled_val,
                            'perc_train': percentage_unlabelled, 'gpus': gpus, 'batch_size': batch_size_unlabelled,
                            'training_loss': run_loss, 'training_epoch': run_epochs,
                            'val_loss': run_val_loss, 'val_epoch': run_val_epochs, 'run_time':ellapsed_time},
                           experiment_folder, f'unlabelled_dataset_counts.json')

    return bal_frames


def run_downstream(experiment_folder, dataset_path, percentage_labelled, runs, percentage_unlabelled,
                   consensus_dataset_path = None, train=True, test=True, representative = False):

    keras.mixed_precision.set_global_policy('float32')

    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    gpus = ["/gpu:2"]
    single_gpu_labelled_batch_size = 4
    batch_size_labelled = single_gpu_labelled_batch_size * len(gpus)

    if not representative:
        (training_dataset, validation_dataset, training_images, training_labels, val_images, val_labels,
         test_images, test_labels, num_train, num_val, num_test) = load_labelled_datasets_unity(dataset_path,
            batch_size_labelled, percentage_labelled)
    else:
        (training_dataset, validation_dataset, training_images, training_labels, val_images, val_labels,
         test_images, test_labels, num_train, num_val, num_test) = load_labelled_datasets_unity_representative(dataset_path, 'classifier_output.csv',
                                                                                                batch_size_labelled,
                                                                                                percentage_labelled)

    consensus_a2c_images, consensus_a2c_labels = [],[]
    if consensus_dataset_path is not None:
        consensus_a2c_images, consensus_a2c_labels = get_list_of_images_and_labels_from_folder(
            consensus_dataset_path,
            images_folder,
            labels_folder,
            False)

    unlabelled_counts_dict = load_json(os.path.join(experiment_folder, 'unlabelled_dataset_counts.json'))

    batch_size_labelled = single_gpu_labelled_batch_size * len(gpus)

    if not representative:
        (training_dataset, validation_dataset, training_images, training_labels, val_images, val_labels,
         test_images, test_labels, num_train, num_val, num_test) = load_labelled_datasets_unity(dataset_path,
            batch_size_labelled, percentage_labelled)
    else:
        (training_dataset, validation_dataset, training_images, training_labels, val_images, val_labels,
         test_images, test_labels, num_train, num_val, num_test) = load_labelled_datasets_unity_representative(dataset_path,
                                                                                                               'classifier_output.csv',
                                                                                                               batch_size_labelled,
                                                                                                               percentage_labelled)

    percent_downstream = percentage_labelled
    if percentage_labelled is None:
        percent_downstream = 100
    test_pretrained_model_exp(experiment_folder, training_dataset, validation_dataset,
                              test_images, test_labels,
                              consensus_a2c_images, consensus_a2c_labels,
                              unlabelled_counts_dict['train'], percentage_unlabelled,
                              num_train, num_val, num_test, len(consensus_a2c_images),
                              runs, batch_size_labelled,
                              train, test, percent_downstream, gpus)


def get_batch_size(mixed_precision):
    single_gpu_unlabelled_batch_size = 64#64 #20
    if not mixed_precision:
        single_gpu_unlabelled_batch_size = 32

    if BACKBONE == BACKBONES[1]:
        single_gpu_unlabelled_batch_size = 20  # 16#20#8
        if not mixed_precision:
            single_gpu_unlabelled_batch_size = 16

    return single_gpu_unlabelled_batch_size


def main():
    #tf.config.run_functions_eagerly(True) for debugging step function

    tf.random.set_seed(777)
    np.random.seed(555)
    random.seed(444)

    unlabelled_dataset_path_a2c = 'Data/@-securion-a2c-dual-layer-CYBER-frames-resized'     # 136,593 unlabelled A2C LV
    unlabelled_dataset_path = 'Data/1 A4CH_LV resized'                                      # 105,451 unlabelled A4C LV
    #unlabelled_dataset_path = r'Data/unity final cleaning/Images/training/'  # Unity
    #unlabelled_dataset_path = 'Data/Unity Frames'  # UnityALL
    labelled_data_set_path = r'Data/unity final cleaning/'
    consensus_dataset_folder = 'Data/Expert Consensus dataset'

    global BACKBONE
    BACKBONE = BACKBONES[1]
    global LOSS
    LOSS = LOSS_FUNCTIONS[0]
    global lOCK_ENC_NORM_LAYERS

    ssl_mixed_precision = True
    single_gpu_batch_size = get_batch_size(ssl_mixed_precision)

    representative = False

    main_folder = 'run_SimCLR_A4CHLV_1900'#'run_SimCLR_A4CHLV'#'run_SimCLR_A4CHLV_1900'
    if representative:
        main_folder += '_leastrep'
    main_folder = f'{main_folder}_{BACKBONE}_{LOSS}_{ssl_mixed_precision}_batch{single_gpu_batch_size}'

    train_ssl = False                   #train a model on a pretext task using only unlabelled images
    train_fine_tuned_model = False      #fine tune a model using the pre-trained ssl model
    test_fine_tune = False

    percentages_for_unlabelled_training = [1900]##[1000, 30000, 100000]
    percentages_for_downstream_training = [1]#[1, 2, 3, 4, 5, 10, 15, 25, 100]

    num_fine_tuning_runs = 2  # Fine k models and take the average dice/hd scores of the k

    # SSL training
    if train_ssl:
        for perc_unlabelled in percentages_for_unlabelled_training:
            experiment_folder = os.path.join(main_folder, f'Exp_with_{perc_unlabelled}_percent_unlabelled_data')
            run_ssl(mixed_precision=ssl_mixed_precision,
                    single_gpu_unlabelled_batch_size=single_gpu_batch_size,
                    experiment_folder=experiment_folder,
                    unlabelled_dataset_path=unlabelled_dataset_path,
                    percentage_unlabelled=perc_unlabelled)

    #Fine-tuning with SSL models trained above
    if train_fine_tuned_model or test_fine_tune:
        for perc_labelled in percentages_for_downstream_training:
            for perc_unlabelled in percentages_for_unlabelled_training: #just so we know which pretrained encoder to use

                #This resets the seed for each percentage that is being tested.
                #Allows for a fair comparison of performances between percentages.
                #This also ensures that that the training set of succeeding downstream percentages will contain the
                #image files of preceeding downstream percentages, i.e., 12% will contain all images of 7% and so on.
                tf.random.set_seed(777)
                np.random.seed(555)
                random.seed(444)

                if perc_labelled == 100:
                    lOCK_ENC_NORM_LAYERS = False
                else:
                    lOCK_ENC_NORM_LAYERS = True

                experiment_folder = os.path.join(main_folder, f'Exp_with_{perc_unlabelled}_percent_unlabelled_data')
                run_downstream(experiment_folder=experiment_folder, dataset_path=labelled_data_set_path, percentage_labelled=perc_labelled,
                               percentage_unlabelled=perc_unlabelled, runs=num_fine_tuning_runs,
                               consensus_dataset_path=consensus_dataset_folder,
                               train=train_fine_tuned_model, test=test_fine_tune,
                               representative = representative)

    print('Complete.')

if __name__ == "__main__":
    main()