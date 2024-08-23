from typing import Union

import numpy as np


def is_essential_agreement(
        label: Union[np.ndarray, list],
        predicted: Union[np.ndarray, list],
        min_after_log2: float | None = None,
        max_after_log2: float | None = None
):
    """
    Check if the predicted value is within the essential agreement range
    :param label:
    :param predicted:
    :param min_after_log2:
    :param max_after_log2:
    :return:
    """
    ea = np.zeros(len(label), dtype=bool)

    if isinstance(label, list):
        label = np.array(label)
    if isinstance(predicted, list):
        predicted = np.array(predicted)

    for i in range(len(label)):
        pred = predicted[i]
        ceil = np.ceil(pred)
        floor = np.floor(pred)

        mid = (2 ** ceil + 2 ** floor) / 2
        if 2 ** pred < mid:
            pred = floor
        else:
            pred = ceil


        if min_after_log2 is not None and label[i] <= min_after_log2:
            ea[i] = pred <= min_after_log2
        elif max_after_log2 is not None and label[i] >= max_after_log2:
            ea[i] = pred >= max_after_log2
        else:
            ea[i] = abs(label[i] - pred) <= 1

    return ea


def essential_agreement_cus_metric(preds, dtrain):
    labels = dtrain.get_label()
    agreements = is_essential_agreement(labels, preds)
    agreement_rate = np.mean(agreements)
    return 'essential_agreement', agreement_rate


def standardize_data(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test


def normalize_output(
        output,
        min_output,
        max_output
):
    return (output - min_output) / (max_output - min_output)


def denormalize_output(
        output,
        min_output,
        max_output
):
    return output * (max_output - min_output) + min_output
