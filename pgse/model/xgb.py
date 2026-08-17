import math
from typing import Optional, Sequence, cast

import numpy as np
import pandas as pd
import xgboost as xgb
import ray
from tqdm import tqdm

from time import time
from pgse.log import logger
from pgse.model.label_scaler import LabelScaler
from pgse.result.predictions import as_matrix, as_target, result_frame
from pgse.validation import Metric

# Cores allocated to each partition's XGBoost worker when partitioning is on.
# Ray therefore runs about (workers // CORES_PER_PARTITION) partitions at once,
# each XGBoost using this many threads. The final, non-partitioned model is the
# exception: it gets the whole worker pool instead (see __init__).
CORES_PER_PARTITION = 8

# One tree per label per round. The alternative, multi_output_tree, reports neither
# gain nor cover, which feature selection ranks the segments by.
MULTI_STRATEGY = 'one_output_per_tree'

# Columns of the importance frame the trained partitions report.
FEATURE_COLUMN = 'Feature'
LABEL_COLUMN = 'Label'
IMPORTANCE_COLUMN = 'Importance'


class XGBoost:
    def __init__(
            self,
            partition_size_target: int,
            boost_rounds: int = 250,
            max_depth: int = 3,
            base_learning_rate: float = 0.05,
            importance_type: str = 'gain',
            use_partition: bool = False,
            num_cpu_per_node: int = 8,
            custom_metric: Optional[Metric] = None,
            early_stopping_rounds: int = 20,
            device: str = 'cpu',
            binary: bool = False,
            label_names: Optional[Sequence[str]] = None,
            scaler: Optional[LabelScaler] = None
    ):
        """
        Initialize XGBoost model with parameters. `binary` trains a 0/1 classifier
        (`binary:logistic`) whose predictions are probabilities, instead of a regressor.
        Several `label_names` train one output per label, and `scaler` standardises the
        labels for training and returns the predictions in the units of the dataset.
        """
        self.boost_rounds = boost_rounds
        self.max_depth = max_depth
        self.base_learning_rate = base_learning_rate
        self.importance_type = importance_type
        self.use_partition = use_partition
        self.partition_size_target = partition_size_target
        self.num_cpu_per_node = num_cpu_per_node
        self.custom_metric = custom_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.binary = binary
        self.label_names = list(label_names) if label_names else []
        self.scaler = scaler

        # num_cpu_per_node is the whole Ray pool (the --workers value). When we
        # partition, cap each partition to a small fixed number of cores so many
        # run concurrently; the single final model keeps the whole pool. Clamp to
        # the pool size so a partition never reserves more cores than exist, which
        # would leave the task pending forever.
        self.cores_per_task = (
            min(CORES_PER_PARTITION, num_cpu_per_node) if use_partition else num_cpu_per_node
        )

        self.params = {
            'objective': 'binary:logistic' if binary else 'reg:squarederror',
            'eval_metric': 'logloss' if binary else 'rmse',
            'max_depth': max_depth,
            'tree_method': 'hist',
            'multi_strategy': MULTI_STRATEGY,
            'device': device,
            'learning_rate': base_learning_rate,
            'nthread': self.cores_per_task  # threads per partition worker
        }

    def _create_dmatrix(self, data: np.ndarray, label: np.ndarray) -> xgb.DMatrix:
        """
        Helper method to create DMatrix for training and testing.
        """
        return xgb.DMatrix(data, label=as_target(label))

    def _adaptive_learning_rate(self, train_x: np.ndarray) -> float:
        """
        Calculate the adaptive learning rate based on the number of features.
        """
        return self.base_learning_rate / math.sqrt(self.partition_size_target / train_x.shape[1])

    def _scale(self, labels: np.ndarray) -> np.ndarray:
        """Standardise the labels, when the run asked for it.

        Args:
            labels: Labels in the units of the dataset.
        """
        return labels if self.scaler is None else self.scaler.transform(labels)

    def _unscale(self, predictions: np.ndarray) -> np.ndarray:
        """Return the predictions in the units of the dataset.

        Args:
            predictions: Predictions on the scale the model was trained on.
        """
        return predictions if self.scaler is None else self.scaler.inverse_transform(predictions)

    def _label_names(self, n_labels: int) -> list[str]:
        """The name of every label, filled in with positions when none were given.

        Args:
            n_labels: Number of labels the model predicts.
        """
        if len(self.label_names) == n_labels:
            return self.label_names

        return [str(index) for index in range(n_labels)]

    def _partition_importance(
            self,
            model: xgb.Booster,
            feature_indices: np.ndarray,
            n_labels: int
    ) -> list[tuple[int, int, float]]:
        """The importance of every feature the model split on, per label it was split for.

        Args:
            model: The booster trained on the partition.
            feature_indices: Count-matrix column of each feature the partition holds.
            n_labels: Number of labels the model predicts.
        """
        if n_labels == 1:
            score = model.get_score(importance_type=self.importance_type)
            return [
                (int(feature_indices[int(name[1:])]), 0, float(cast(float, value)))
                for name, value in score.items()
            ]

        # One tree per label per round, in label order, so a tree's position gives its label.
        trees = model.trees_to_dataframe()
        splits = trees[trees[FEATURE_COLUMN] != 'Leaf'].copy()
        splits[LABEL_COLUMN] = splits['Tree'] % n_labels
        gains = splits.groupby([FEATURE_COLUMN, LABEL_COLUMN], as_index=False)['Gain'].mean()

        return [
            (int(feature_indices[int(name[1:])]), int(label), float(gain))
            for name, label, gain in zip(gains[FEATURE_COLUMN], gains[LABEL_COLUMN], gains['Gain'])
        ]

    @ray.remote(
        num_cpus=1,
        # num_gpus=0
    )
    def _train_one_partition(
            self,
            train_x: np.ndarray,
            train_y: np.ndarray,
            test_x: np.ndarray,
            test_y: np.ndarray,
            feature_indices: np.ndarray,
            verbose: int = 50
    ):
        """
        Train XGBoost model on a subset of features (a partition).
        """
        # if self.use_partition:  # min-max scaling
        #     scaler = MinMaxScaler()
        #     train_x = scaler.fit_transform(train_x)
        #     test_x = scaler.transform(test_x)

        n_labels = as_matrix(train_y).shape[1]

        dtrain = self._create_dmatrix(train_x, self._scale(train_y))
        dtest = self._create_dmatrix(test_x, self._scale(test_y))

        # Update learning rate based on the number of features
        # self.params['learning_rate'] = self._adaptive_learning_rate(train_x)

        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        model = xgb.train(
            self.params, dtrain, self.boost_rounds, evals=watchlist,
            # custom_metric=self.custom_metric,
            verbose_eval=verbose,
            early_stopping_rounds=self.early_stopping_rounds
        )

        start = time()
        predictions = self._unscale(model.predict(dtest))
        if not self.use_partition:
            logger.info(f'Inference time: {time() - start:.3f} seconds')

        if self.custom_metric is not None:
            score = self.custom_metric(predictions, test_y)
            logger.info(f'{self.custom_metric.name}: {score}')

        importance = self._partition_importance(model, feature_indices, n_labels)

        return predictions, importance, None if self.use_partition else model

    def _create_partitions(self, feature_count: int) -> list:
        """
        Split features based on partition size.
        """
        num_partitions = feature_count // self.partition_size_target if self.partition_size_target > 0 else 1
        # use just 1 partition if we are not using partitioning e.g. testing/inference
        num_partitions = max(num_partitions, 1) if self.use_partition else 1
        return np.array_split(np.arange(feature_count), num_partitions)

    def _gather_results(self, tasks: list) -> tuple:
        """
        Gather and combine results from the Ray tasks.
        """
        results = [ray.get(task) for task in tqdm(tasks, desc='Training partitions')]

        combined_predictions = np.mean([res[0] for res in results], axis=0)
        all_importance = [imp for res in results for imp in res[1]]
        trained_model = None if self.use_partition else results[0][2]

        return combined_predictions, all_importance, trained_model

    def _importance_frame(
            self,
            entries: list[tuple[int, int, float]],
            n_labels: int
    ) -> pd.DataFrame:
        """Rank the features by importance, giving every label an equal say.

        Each label's gains are read as its share of the gain that label explains, so a
        label whose values are on a larger scale, and whose splits therefore carry larger
        gains, cannot crowd the others out of the selection.

        Args:
            entries: The (feature, label, importance) triples of every partition.
            n_labels: Number of labels the model predicts.
        """
        frame = pd.DataFrame(entries, columns=[FEATURE_COLUMN, LABEL_COLUMN, IMPORTANCE_COLUMN])

        if n_labels > 1 and not frame.empty:
            totals = frame.groupby(LABEL_COLUMN)[IMPORTANCE_COLUMN].transform('sum').replace(0.0, 1.0)
            frame[IMPORTANCE_COLUMN] = frame[IMPORTANCE_COLUMN] / totals
            frame = frame.groupby(FEATURE_COLUMN, as_index=False)[IMPORTANCE_COLUMN].sum()

        return frame[[FEATURE_COLUMN, IMPORTANCE_COLUMN]].sort_values(
            by=IMPORTANCE_COLUMN, ascending=False
        )

    def _calculate_rmse(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error (RMSE).
        """
        return np.sqrt(np.mean((predictions - actuals) ** 2))

    def _log_rmse(self, predictions: np.ndarray, actuals: np.ndarray, label_names: list[str]):
        """Log the RMSE of every label.

        Args:
            predictions: Predictions, one row per sample and one column per label.
            actuals: True labels, in the same shape as predictions.
            label_names: Name of each label, in the order of the label columns.
        """
        if len(label_names) == 1:
            logger.info(f'Root Mean Square Error: {self._calculate_rmse(predictions, actuals)}')
            return

        per_label = ', '.join(
            f'{name}: {self._calculate_rmse(predictions[:, index], actuals[:, index])}'
            for index, name in enumerate(label_names)
        )
        logger.info(f'Root Mean Square Error: {per_label}')

    def run(
            self,
            train_x: np.ndarray,
            test_x: np.ndarray,
            train_y: np.ndarray,
            test_y: np.ndarray,
    ) -> tuple[pd.DataFrame, pd.DataFrame, xgb.Booster]:
        """
        Run the training and testing process.
        """

        feature_partitions = self._create_partitions(train_x.shape[1])
        label_names = self._label_names(as_matrix(train_y).shape[1])

        logger.info(
            f'Training {len(feature_partitions)} partitions of features, '
            f'{self.cores_per_task} core(s) each '
            f'(~{max(self.num_cpu_per_node // self.cores_per_task, 1)} concurrent), '
            f'predicting {len(label_names)} label(s)'
        )

        tasks = []
        for split in feature_partitions:
            train_x_split = train_x[:, split]
            test_x_split = test_x[:, split]

            task_ref = self._train_one_partition.options(
                num_cpus=self.cores_per_task,
                # num_gpus=1
            ).remote(
                self,
                train_x_split,
                train_y,
                test_x_split,
                test_y,
                split,
                verbose=0 if self.use_partition else 50
            )
            tasks.append(task_ref)

        combined_predictions, all_importance, trained_model = self._gather_results(tasks)

        results_df = result_frame(test_y, combined_predictions, label_names)
        importance_df = self._importance_frame(all_importance, len(label_names))

        self._log_rmse(as_matrix(combined_predictions), as_matrix(test_y), label_names)

        return results_df, importance_df, trained_model
