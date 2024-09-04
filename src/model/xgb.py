import numpy as np
import pandas as pd
import xgboost as xgb
import ray
from tqdm import tqdm

from src.log import logger

class XGBoost:
    def __init__(
            self,
            boost_rounds: int = 250,
            max_depth: int = 4,
            learning_rate: float = 0.05,
            importance_type: str = 'gain',
            partition_size: int = 0,
            num_cpu_per_node: int = 8,
            custom_metric=None
    ):
        """
        Initialize XGBoost model with parameters
        """
        self.boost_rounds = boost_rounds
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.importance_type = importance_type
        self.partition_size = partition_size
        self.num_cpu_per_node = num_cpu_per_node
        self.custom_metric = custom_metric

        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'nthread': num_cpu_per_node  # Use multiple threads per worker
        }

    def _create_dmatrix(self, data: np.ndarray, label: np.ndarray) -> xgb.DMatrix:
        """
        Helper method to create DMatrix for training and testing.
        """
        return xgb.DMatrix(data, label=label)

    @ray.remote(num_cpus=1)
    def _train_one_partition(
            self,
            train_x: np.ndarray,
            train_y: np.ndarray,
            test_x: np.ndarray,
            test_y: np.ndarray,
            feature_indices: np.ndarray,
            verbose: int = 100
    ):
        """
        Train XGBoost model on a subset of features (a split).
        """
        dtrain = self._create_dmatrix(train_x, train_y)
        dtest = self._create_dmatrix(test_x, test_y)

        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        model = xgb.train(
            self.params, dtrain, self.boost_rounds, watchlist,
            custom_metric=self.custom_metric,
            verbose_eval=verbose
        )

        predictions = model.predict(dtest)
        results = {
            'Prediction': predictions,
            'Actual': dtest.get_label()
        }

        importance = model.get_score(importance_type=self.importance_type)
        importance_mapped = {feature_indices[int(k[1:])]: v for k, v in importance.items()}

        return results, list(importance_mapped.items())

    def _create_partitions(self, feature_count: int) -> list:
        """
        Split features based on partition size.
        """
        num_splits = feature_count // self.partition_size if self.partition_size > 0 else 1
        num_splits = max(num_splits, 1)
        return np.array_split(np.arange(feature_count), num_splits)

    def _gather_results(self, tasks: list) -> tuple:
        """
        Gather and combine results from the Ray tasks.
        """
        results = [ray.get(task) for task in tqdm(tasks, desc='Training partitions')]

        combined_predictions = np.mean([res[0]['Prediction'] for res in results], axis=0)
        all_importance = [imp for res in results for imp in res[1]]

        return combined_predictions, all_importance

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
    ) -> tuple:
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
                num_cpus=self.num_cpu_per_node
            ).remote(
                self,
                train_x_split,
                train_y,
                test_x_split,
                test_y,
                split,
                verbose=100 if not self.partition_size else 0
            )
            tasks.append(task_ref)

        combined_predictions, all_importance = self._gather_results(tasks)

        results_df = pd.DataFrame({
            'Prediction': combined_predictions,
            'Actual': test_y
        })

        importance_df = pd.DataFrame(all_importance, columns=['Feature', 'Importance'])
        importance_df.sort_values(by='Importance', ascending=False, inplace=True)

        rmse = self._calculate_rmse(results_df['Prediction'], results_df['Actual'])
        self._log_rmse(rmse)

        return results_df, importance_df
