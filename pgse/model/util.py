from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle


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


def oversample_minority_class(X, y, shuffle_data=True):
    # Initialize the RandomOverSampler
    ros = RandomOverSampler(random_state=42)

    # Perform oversampling
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Shuffle the data if shuffle_data is True
    if shuffle_data:
        X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=42)

    return X_resampled, y_resampled

