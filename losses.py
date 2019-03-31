import numpy as np
from keras import backend as K
import tensorflow as tf
import numpy as np


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def dice(y_true, y_pred):
    """
    Average dice across all samples
    """
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1-dice(y_true, y_pred)


def dice_and_focal(y_true, y_pred):
    gamma = 4.
    alpha = .25
    dice_l = 1-dice(y_true, y_pred)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    focal_l = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - \
        K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return dice_l + focal_l


def dice_and_binary_crossentropy(y_true, y_pred):
    bce_l = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    dice_l = 1-dice(y_true, y_pred)
    return bce_l + dice_l


def iou_and_binary_crossentropy(y_true, y_pred):
    bce_l = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    smooth = 1e-6
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth + bce_l


def dice_iou_and_binary_crossentropy(y_true, y_pred):
    bce_l = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    smooth = 1e-6
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    dice_l = 1-dice(y_true, y_pred)
    return (1 - jac) * smooth + bce_l + dice_l


def dice_and_iou(y_true, y_pred):
    smooth = 1e-6
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    dice_l = 1-dice(y_true, y_pred)
    return (1 - jac) * smooth + dice_l


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss


def jaccard_distance_loss(y_true, y_pred):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    smooth = 1
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def single_class_accuracy(interesting_class_id):
    def sing_acc(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(
            K.equal(class_id_preds, interesting_class_id), 'int32')
        class_acc_tensor = K.cast(
            K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / \
            K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return sing_acc
