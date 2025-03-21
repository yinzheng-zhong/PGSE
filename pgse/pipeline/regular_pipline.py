import pandas as pd
import ray

from pgse.environment.ray_env import RayEnvManager
from pgse.log import logger
from pgse.model.model_trainer import ModelTrainer
from pgse.dataset.file_label import FileLabel
from pgse.dataset.loader import Loader
from pgse.segment import seg_pool


class Pipeline:
    def __init__(
            self,
            data_dir: str,
            label_file: str,
            pre_kfold_info_file: str = None,
            export_file: str = './default.export',
            k: int = 6,
            ext: int = 2,
            folds: int = 0,
            ea_min: float = None,
            ea_max: float = None,
            num_rounds: int = 1500,
            lr: float = 0.03,
            dist: bool = False,
            nodes: int = 1,
            workers: int = 8,
    ):
        self.data_dir = data_dir
        self.label_file = label_file
        self.pre_kfold_info_file = pre_kfold_info_file
        self.export_file = export_file
        self.k = k
        self.ext = ext
        self.folds = folds
        self.ea_min = ea_min
        self.ea_max = ea_max
        self.num_rounds = num_rounds
        self.lr = lr
        self.dist = dist
        self.nodes = nodes
        self.workers = workers

        self.file_label = FileLabel(self.label_file, self.data_dir, self.pre_kfold_info_file)

    def run(self):
        RayEnvManager.initialize(self.dist, self.nodes, self.workers)

        accumulated_results = pd.DataFrame()

        # Use k-mer data only without any feature selection or partitioning
        for i in range(self.folds if self.folds > 0 else 1):
            logger.info(f'==================== Fold {i + 1} ====================')
            loader = Loader(
                self.file_label,
                folds=self.folds,
                fold_index=i
            )

            model_trainer = ModelTrainer(
                loader,
                self.num_rounds,
                self.workers,
                self.lr,
                self.ea_min,
                self.ea_max
            )

            # Load k-mer dataset
            seg_pool.clear()
            seg_pool.add_all_kmer(self.k, self.ext)
            train_kmer, test_kmer, train_labels, test_labels = loader.get_dataset_from_pool()

            # Run XGBoost without partitioning or custom metrics
            custom_metric = model_trainer.custom_essential_agreement_metric()
            fold_results, importance_df, trained_model = model_trainer.run_xgboost(
                train_kmer, test_kmer, train_labels, test_labels, use_partition=False, custom_metric=custom_metric
            )

            logger.info(fold_results)
            logger.info("Feature importance:")
            logger.info(str(importance_df.head(20)))

            # Append fold results
            accumulated_results = pd.concat([accumulated_results, fold_results], ignore_index=True)
            trained_model.save_model(f'{self.export_file}_regular_xgboost_fold_{i}')


        # Export final results and shutdown Ray
        accumulated_results.to_csv(f'{self.export_file}_regular_xgboost.csv', index=False)
        ray.shutdown()
