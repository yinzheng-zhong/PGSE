from typing import Union

import numpy as np


def is_essential_agreement(
        label: Union[np.ndarray, list],
        predicted: Union[np.ndarray, list],
):
    if isinstance(label, list):
        label = np.array(label)
    if isinstance(predicted, list):
        predicted = np.array(predicted)

    return np.abs(label - predicted) <= 1


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
