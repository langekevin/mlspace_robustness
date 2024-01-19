"""This module defines different error metrics
used for training and validating the neural network
"""

from tensorflow.python.keras import backend as K


epsilon = 0.0000001


def jacc_coef(y_true, y_pred):
    """Calculates the Jaccard coefficient which is the intersection of the
    ground truth and the predicted value divided by the sum of the predicted
    and ground truth positions that are no part of the intersection

    Args:
        y_true (np.ndarray): The ground truth value (e. g. the image mask)
        y_pred (np.ndarray): The predicted value (e. g. the image mask)

    Returns:
        float: A number between 0 and 1 where 0 indicates a perect match
            between `y_true` and `y_pred` and 1 being completely different.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((intersection + epsilon) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + epsilon))


def precision(y_true, y_pred):
    """Calculates the precision value which describes the sum of all
    true positives devided by the sum of predicted positives.

    Args:
        y_true (np.ndarray): The ground truth value (e. g. the image mask)
        y_pred (np.ndarray): The predicted value (e. g. the image mask)

    Returns:
        float: A number between 0 and 1 where 1 indicates a perfect match
            between the true positives and the predicted positives and 0
            having no precision (e. g. true positives) at all.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Calculates the recall value which describes the sum of all
    true positives divided by the sum of all possible positive values

    Args:
        y_true (np.ndarray): The ground truth value (e. g. the image mask)
        y_pred (np.ndarray): The predicted value (e. g. the image mask)

    Returns:
        float: A number between 0 and 1 where 1 indicates a perfect match
            between the true positives and the possible positive values and
            0 having not covered the possible positives at all.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def overall_accuracy(y_true, y_pred):
    """Calculates the accuracy value which describes the ratio between
    the true positives and negatives and the sum of all predicted values
    (e. g. true positive/negative and false positive/negative).

    Args:
        y_true (np.ndarray): The ground truth value (e. g. the image mask)
        y_pred (np.ndarray): The predicted value (e. g. the image mask)

    Returns:
        float: A number between 0 and 1 where 1 indicates a perfect match
            between `y_true` and `y_pred` and 0 indicates a complete wrong
            prediction.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    false_positives = K.sum(K.round(K.clip((y_true * -1 + 1) * y_pred, 0, 1)))
    true_negatives = K.sum(K.round(K.clip((y_true * -1 + 1) * (y_pred * -1 + 1), 0, 1)))
    false_negatives = K.sum(K.round(K.clip(y_true * (y_pred * -1 + 1), 0, 1)))
    return (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
