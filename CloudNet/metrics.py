from tensorflow.python.keras import backend as K


epsilon = 0.0000001


def jacc_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((intersection + epsilon) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + epsilon))


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def overall_accuracy(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    false_positives = K.sum(K.round(K.clip((y_true * -1 + 1) * y_pred, 0, 1)))
    true_negatives = K.sum(K.round(K.clip((y_true * -1 + 1) * (y_pred * -1 + 1), 0, 1)))
    false_negatives = K.sum(K.round(K.clip(y_true * (y_pred * -1 + 1), 0, 1)))
    return (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
