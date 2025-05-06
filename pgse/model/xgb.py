import math

import numpy as np
import pandas as pd
import xgboost as xgb
import ray
from tqdm import tqdm

from time import time
from pgse.log import logger


class XGBoost:
    def __init__(
            self,
            partition_size: int,
            boost_rounds: int = 250,
            max_depth: int = 3,
            base_learning_rate: float = 0.05,
            importance_type: str = 'gain',
            use_partition: bool = False,
            num_cpu_per_node: int = 8,
            custom_metric=None,
            early_stopping_rounds: int = 20,
            device: str = 'cpu'
    ):
        """
        Initialize XGBoost model with parameters
        """
        self.boost_rounds = boost_rounds
        self.max_depth = max_depth
        self.base_learning_rate = base_learning_rate
        self.importance_type = importance_type
        self.use_partition = use_partition
        self.partition_size = partition_size
        self.num_cpu_per_node = num_cpu_per_node
        self.custom_metric = custom_metric
        self.early_stopping_rounds = early_stopping_rounds

        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': max_depth,
            'tree_method': 'hist',
            'device': device,
            'learning_rate': base_learning_rate,
            'nthread': num_cpu_per_node  # Use multiple threads per worker
        }

    def _create_dmatrix(self, data: np.ndarray, label: np.ndarray) -> xgb.DMatrix:
        """
        Helper method to create DMatrix for training and testing.
        """
        return xgb.DMatrix(data, label=label)

    def _adaptive_learning_rate(self, train_x: np.ndarray) -> float:
        """
        Calculate the adaptive learning rate based on the number of features.
        """
        return self.base_learning_rate / math.sqrt(self.partition_size / train_x.shape[1])

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

        dtrain = self._create_dmatrix(train_x, train_y)
        dtest = self._create_dmatrix(test_x, test_y)

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
        predictions = model.predict(dtest)
        if not self.use_partition:
            logger.info(f'Inference time: {time() - start:.3f} seconds')

        results = {
            'Prediction': predictions,
            'Actual': dtest.get_label()
        }

        if self.custom_metric is not None:
            ea = self.custom_metric(predictions, dtest)
            logger.info(f'Essential Agreement: {ea}')

        importance = model.get_score(importance_type=self.importance_type)
        importance_mapped = {feature_indices[int(k[1:])]: v for k, v in importance.items()}

        return results, list(importance_mapped.items()), None if self.use_partition else model

    def _create_partitions(self, feature_count: int) -> list:
        """
        Split features based on partition size.
        """
        num_partitions = feature_count // self.partition_size if self.partition_size > 0 else 1
        # use just 1 partition if we are not using partitioning e.g. testing/inference
        num_partitions = max(num_partitions, 1) if self.use_partition else 1
        return np.array_split(np.arange(feature_count), num_partitions)

    def _gather_results(self, tasks: list) -> tuple:
        """
        Gather and combine results from the Ray tasks.
        """
        results = [ray.get(task) for task in tqdm(tasks, desc='Training partitions')]

        combined_predictions = np.mean([res[0]['Prediction'] for res in results], axis=0)
        all_importance = [imp for res in results for imp in res[1]]
        trained_model = None if self.use_partition else results[0][2]

        return combined_predictions, all_importance, trained_model

    def _calculate_rmse(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error (RMSE).
        """
        return np.sqrt(np.mean((predictions - actuals) ** 2))

    def _log_rmse(self, rmse: float):
        """
        Log the RMSE value.
        """
        logger.info(f'Root Mean Square Error: {rmse}')

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

        logger.info(f'Training {len(feature_partitions)} partitions of features')

        tasks = []
        for split in feature_partitions:
            train_x_split = train_x[:, split]
            test_x_split = test_x[:, split]

            task_ref = self._train_one_partition.options(
                num_cpus=self.num_cpu_per_node,
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

        results_df = pd.DataFrame({
            'Prediction': combined_predictions,
            'Actual': test_y
        })

        importance_df = pd.DataFrame(all_importance, columns=['Feature', 'Importance'])
        importance_df.sort_values(by='Importance', ascending=False, inplace=True)

        rmse = self._calculate_rmse(results_df['Prediction'], results_df['Actual'])
        self._log_rmse(rmse)

        return results_df, importance_df, trained_model
