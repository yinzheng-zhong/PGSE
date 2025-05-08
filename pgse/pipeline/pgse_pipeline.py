import os
import ray

from xgboost import  Booster

from pgse.environment.ray_env import RayEnvManager
from pgse.log import logger
from pgse.model.model_trainer import ModelTrainer
from pgse.dataset.file_label import FileLabel
from pgse.dataset.loader import Loader
from pgse.pipeline.progress_manager import ProgressManager
from pgse.segment.extender import Extender
from pgse.segment import seg_pool


class PGSEPipelineOutput:
    def __init__(self, models, segments, results):
        self.models = models
        self.segments = segments
        self.results = results


class Pipeline:
    def __init__(
            self,
            data_dir: str,
            label_file: str | dict,
            pre_kfold_info_file: str = None,
            save_file: str = '',
            export_file: str = './default.export',
            k: int = 6,
            ext: int = 2,
            target: int = 70,
            features: int = 10000,
            folds: int = 0,
            ea_min: float = None,
            ea_max: float = None,
            num_rounds: int = 1500,
            lr: float = 0.03,
            dist: bool = False,
            nodes: int = 1,
            workers: int = 8,
            device: str = 'cpu'
    ):
        self.data_dir = data_dir
        self.label_file = label_file
        self.pre_kfold_info_file = pre_kfold_info_file
        self.save_file = save_file
        self.export_file = export_file
        self.k = k
        self.ext = ext
        self.target = target
        self.features = features
        self.folds = folds
        self.ea_min = ea_min
        self.ea_max = ea_max
        self.dist = dist
        self.nodes = nodes
        self.num_rounds = num_rounds
        self.lr = lr
        self.workers = workers
        self.device = device

        self.file_label = FileLabel(self.label_file, self.data_dir, self.pre_kfold_info_file)
        self.extender = Extender()
        self.progress_manager = ProgressManager(self.save_file, self.k, self.ext)
        self.model_trainer = None

        self.models: list[Booster] = []
        self.segments: list[str] = []
        self.results: list[dict] = []
        self._suppress_write: bool = False

    def extend_segments(self):
        try:
            self.extender.extend_all_segs(self.ext)
        except ValueError:
            logger.error("No segments could be extended. Stopping.")
            return False

        return True

    def run(self):
        RayEnvManager.initialize(self.dist, self.nodes, self.workers)

        start_fold, accumulated_results = self.progress_manager.load_fold_progress()

        for i in range(start_fold, self.folds if self.folds > 0 else 1):
            logger.info(f'==================== Fold {i + 1} ====================')
            loader = Loader(
                self.file_label,
                folds=self.folds,
                fold_index=i
            )

            self.model_trainer = ModelTrainer(
                loader,
                self.num_rounds,
                self.workers,
                self.lr,
                self.features,
                self.ea_min,
                self.ea_max,
                device=self.device
            )

            train_kmer, test_kmer, train_labels, test_labels = self.progress_manager.load_round_progress(loader)

            while True:
                logger.info(f'==================== Feature Selection ====================')

                # Step 1: Run XGBoost for feature selection
                xgb_result = self.model_trainer.run_xgboost(train_kmer, test_kmer, train_labels, test_labels)
                self.model_trainer.perform_feature_selection(xgb_result)

                # Step 2: Attempt to extend segments
                if seg_pool.get_current_max_length() >= self.target or not self.extend_segments():
                    break

                if not self._suppress_write:
                    self.progress_manager.save_round_progress()

                train_kmer, test_kmer, train_labels, test_labels = loader.get_dataset_from_pool()

            # Step 3: Train and test with selected segments
            logger.info(f'==================== Training & testing with selected segments ====================')
            train_kmer, test_kmer, train_labels, test_labels = loader.get_dataset_from_pool()

            # Run XGBoost with custom metric
            custom_metric = self.model_trainer.custom_essential_agreement_metric()
            fold_results, importance_df, trained_model = self.model_trainer.run_xgboost(
                train_kmer, test_kmer, train_labels, test_labels,
                use_partition=False, custom_metric=custom_metric
            )

            logger.info(fold_results)

            # Append fold results
            accumulated_results = self.progress_manager.append_results(fold_results, accumulated_results)

            # Update containers after each fold
            self.models.append(trained_model)
            self.segments.append(seg_pool.get_copy())
            self.results.append(fold_results)

            # Save progress after each fold
            if not self._suppress_write:
                self.progress_manager.save_fold_progress(i + 1, accumulated_results)

                try:
                    os.remove(self.save_file)
                except FileNotFoundError as e:
                    logger.error(e)

                importance_scores = importance_df['Importance']
                seg_pool.export(self.export_file + f'_fold_{i}_segs.csv', importance_scores=importance_scores)
                trained_model.save_model(self.export_file + f'_fold_{i}.json')

        # Export final results and shutdown Ray
        if not self._suppress_write:
            accumulated_results.to_csv(f'{self.export_file}.csv')
        ray.shutdown()

    def train(self):
        self._suppress_write = True

        self.run()

        return(
            PGSEPipelineOutput(
                models=self.models,
                segments=self.segments,
                results=self.results
            )
        )
