import os

import numpy as np
import pandas as pd
import xgboost as xgb

from src.dataset.loader import Loader
from src.model.util import is_essential_agreement, essential_agreement_cus_metric
from src.log import logger
from xgboost_ray import RayDMatrix, RayParams, train


class XGBoost:
    def __init__(
            self,
            boost_rounds: int = 250,
            num_nodes=1,
            num_cpu_per_node=8,
    ):
        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'nthread': num_cpu_per_node,
        }

        self.boost_rounds = boost_rounds
        self.num_nodes = num_nodes
        self.num_cpu_per_node = num_cpu_per_node
        self.max_cpu_per_actor = 8 if num_cpu_per_node > 8 else num_cpu_per_node

    def run(
            self,
            train_x: np.ndarray,
            test_x: np.ndarray,
            train_y: np.ndarray,
            test_y: np.ndarray,
    ):
        """
        Train and test the model using the initial parameters
        :param train_x:
        :param test_x:
        :param train_y:
        :param test_y:
        :return:
        """

        ray_params = None
        # Create DMatrix for training and testing
        if self.num_nodes > 1:
            # Use RayDMatrix for better performance
            dtrain = RayDMatrix(train_x, train_y)
            dtest = RayDMatrix(test_x, test_y)

            total_cpus = self.num_nodes * self.num_cpu_per_node
            num_actors = total_cpus // self.max_cpu_per_actor
            ray_params = RayParams(max_actor_restarts=1, num_actors=num_actors,
                                   cpus_per_actor=self.max_cpu_per_actor)
        else:
            dtrain = xgb.DMatrix(train_x, label=train_y)
            dtest = xgb.DMatrix(test_x, label=test_y)

        # Watchlist to observe the training and testing performance
        watchlist = [(dtrain, 'train'), (dtest, 'test')]

        logger.info('Training the model...')
        if self.num_nodes > 1:
            model = train(params=self.params, dtrain=dtrain, evals=watchlist, num_boost_round=self.boost_rounds,
                          ray_params=ray_params, custom_metric=essential_agreement_cus_metric)
        else:
            model = xgb.train(self.params, dtrain, self.boost_rounds, watchlist, custom_metric=essential_agreement_cus_metric)

        # Print the final evaluation results
        evals_result = model.eval(dtest)
        print(evals_result)

        # Predict using the trained model
        predictions = model.predict(dtest)

        # Create a dataframe with predictions and actual labels
        results_df = pd.DataFrame({
            'Prediction': predictions,
            'Actual': dtest.get_label(),
            'Essential Agreement': list(is_essential_agreement(dtest.get_label(), predictions))
        })

        # Get feature importance
        feature_importance = model.get_score(importance_type='gain')
        # Convert dictionary to pandas dataframe
        importance_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
        # Sort by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        return results_df, importance_df
