import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy

beta = 0.25
alpha = 0.25
gamma = 2
epsilon = 1e-5
smooth = 1

class Semantic_loss_functions(object):
    def __init__(self):
        print ("semantic loss functions initialized")

    def dice_coef(self, y_true, y_pred):
        """
        Compute the Dice coefficient, a measure of similarity between two sets.

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.

        Returns:
            Dice coefficient value.
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + K.epsilon()) / (
                    K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

    def sensitivity(self, y_true, y_pred):
        """
        Compute sensitivity (recall), the ratio of true positives to the sum of true positives and false negatives.

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.

        Returns:
            Sensitivity value.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def specificity(self, y_true, y_pred):
        """
        Compute specificity, the ratio of true negatives to the sum of true negatives and false positives.

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.

        Returns:
            Specificity value.
        """
        true_negatives = K.sum(
            K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())

    def convert_to_logits(self, y_pred):
        """
        Convert probabilities to logits.

        Args:
           y_pred: Predicted probabilities tensor.

        Returns:
           Logits tensor.
        """
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))

    def weighted_cross_entropyloss(self, y_true, y_pred):
        """
        Compute the weighted cross-entropy loss.

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.

        Returns:
            Weighted cross-entropy loss value.
        """
        y_pred = self.convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
                                                        targets=y_true,
                                                        pos_weight=pos_weight)
        return tf.reduce_mean(loss)

    def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
        """
        Compute focal loss for imbalanced datasets, focusing more on hard examples.

        Args:
            logits: Logits tensor.
            targets: Ground truth tensor.
            alpha: Weighting factor for positive examples.
            gamma: Focusing parameter for modulating factor.
            y_pred: Predicted tensor.

        Returns:
            Focal loss value.
        """
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(
            -logits)) * (weight_a + weight_b) + logits * weight_b

    def focal_loss(self, y_true, y_pred):
        """
        Compute the focal loss based on logits and predicted probabilities.

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted probabilities tensor.

        Returns:
            Focal loss value.
        """
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))

        loss = self.focal_loss_with_logits(logits=logits, targets=y_true,
                                      alpha=alpha, gamma=gamma, y_pred=y_pred)

        return tf.reduce_mean(loss)

    def depth_softmax(self, matrix):
        """
        Compute a depth-wise softmax over the input matrix.

        Args:
            matrix: Input tensor.

        Returns:
            Softmax normalized tensor.
        """
        sigmoid = lambda x: 1 / (1 + K.exp(-x))
        sigmoided_matrix = sigmoid(matrix)
        softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
        return softmax_matrix

    def generalized_dice_coefficient(self, y_true, y_pred):
        """
        Compute the generalized Dice coefficient for multi-class segmentation.

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.

        Returns:
            Generalized Dice coefficient value.
        """
        smooth = 1.
        y_true_f = tf.cast(K.flatten(y_true), dtype=tf.float32)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
                    K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score

    def dice_loss(self, y_true, y_pred):
        """
        Compute the Dice loss, which is 1 - Dice coefficient.

        Args:
           y_true: Ground truth tensor.
           y_pred: Predicted tensor.

        Returns:
           Dice loss value.
        """
        loss = 1 - self.generalized_dice_coefficient(y_true, y_pred)
        return loss

    def dice_loss1(self, y_true, y_pred, smooth=1e-5):
        """
        Compute an alternative Dice loss function.

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.
            smooth: Smoothing factor to avoid division by zero.

        Returns:
            Dice loss value.
        """
        y_true = tf.cast(y_true, dtype=tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
        sum_of_squares_pred = tf.reduce_sum(tf.square(y_pred), axis=(1, 2, 3))
        sum_of_squares_true = tf.reduce_sum(tf.square(y_true), axis=(1, 2, 3))
        dice = 1 - (2 * intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true + smooth)
        return dice

    def bce_dice_loss(self, y_true, y_pred):
        """
        Compute a combination of binary cross-entropy loss and Dice loss.

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.

        Returns:
            Average of binary cross-entropy and Dice loss.
        """
        loss = binary_crossentropy(y_true, y_pred) + \
               self.dice_loss(y_true, y_pred)
        return loss / 2.0

    def confusion(self, y_true, y_pred):
        """
        Compute precision and recall values based on the predicted and ground truth tensors.

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.

        Returns:
            A tuple containing precision and recall values.
        """

        smooth = 1
        y_pred_pos = K.clip(y_pred, 0, 1)
        y_pred_neg = 1 - y_pred_pos
        y_pos = K.clip(y_true, 0, 1)
        y_neg = 1 - y_pos
        tp = K.sum(y_pos * y_pred_pos)
        fp = K.sum(y_neg * y_pred_pos)
        fn = K.sum(y_pos * y_pred_neg)
        prec = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        return prec, recall

    def true_positive(self, y_true, y_pred):
        """
        Compute the true positive rate (sensitivity/recall).

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.

        Returns:
            True positive rate value.
        """
        smooth = 1
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pos = K.round(K.clip(y_true, 0, 1))
        tp = (K.sum(y_pos * y_pred_pos) + smooth) / (K.sum(y_pos) + smooth)
        return tp

    def true_negative(self, y_true, y_pred):
        """
        Compute the true negative rate (specificity).

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.

        Returns:
            True negative rate value.
        """
        smooth = 1
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos
        tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth)
        return tn

    def tversky_index(self, y_true, y_pred):
        """
        Compute the Tversky index, a generalization of Dice coefficient with adjustable weighting for false positives and false negatives.

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.

        Returns:
            Tversky index value.
        """
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (
                    1 - alpha) * false_pos + smooth)

    def tversky_loss(self, y_true, y_pred):
        """
        Compute the Tversky loss, which is 1 - Tversky index.

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.

        Returns:
            Tversky loss value.
        """
        return 1 - self.tversky_index(y_true, y_pred)

    def focal_tversky(self, y_true, y_pred):
        """
        Compute the focal Tversky loss, which applies a focusing parameter to the Tversky index for improved performance on imbalanced data.

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.

        Returns:
            Focal Tversky loss value.
        """
        pt_1 = self.tversky_index(y_true, y_pred)
        gamma = 0.75
        return K.pow((1 - pt_1), gamma)

    def log_cosh_dice_loss(self, y_true, y_pred):
        """
        Compute the log-cosh Dice loss, a smoothed version of Dice loss that reduces the impact of outliers.

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted tensor.

        Returns:
            Log-cosh Dice loss value.
        """
        x = self.dice_loss(y_true, y_pred)
        return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)
