import os
import ray

from src.enviromnet.ray_env import RayEnvManager
from src.log import logger
from src.model.model_trainer import ModelTrainer
from src.enviromnet import args
from src.dataset.file_label import FileLabel
from src.dataset.loader import Loader
from src.pipeline.progress_manager import ProgressManager
from src.segment.extender import Extender
from src.segment import seg_pool


class Pipeline:
    def __init__(self):
        self.file_label = FileLabel(args.label_file, args.data_dir)
        self.extender = Extender()
        self.progress_manager = None
        self.model_trainer = None

    def extend_segments(self):
        try:
            self.extender.extend_all_segs(args.ext)
        except ValueError:
            logger.error("No segments could be extended. Stopping.")
            return False

        return True

    def run(self):
        RayEnvManager.initialize()

        start_fold, accumulated_results = ProgressManager.load_fold_progress()

        for i in range(start_fold, args.folds if args.folds > 0 else 1):
            logger.info(f'==================== Fold {i + 1} ====================')
            loader = Loader(
                self.file_label,
                folds=args.folds,
                fold_index=i
            )

            self.progress_manager = ProgressManager(loader)
            self.model_trainer = ModelTrainer(loader)

            train_kmer, test_kmer, train_labels, test_labels = self.progress_manager.load_progress()

            while True:
                logger.info(f'==================== Feature Selection ====================')

                # Step 1: Run XGBoost for feature selection
                xgb_result = self.model_trainer.run_xgboost(train_kmer, test_kmer, train_labels, test_labels)
                self.model_trainer.perform_feature_selection(xgb_result)

                # Step 2: Attempt to extend segments
                if seg_pool.get_current_max_length() >= args.target or not self.extend_segments():
                    break

                seg_pool.save(args.save_file)
                train_kmer, test_kmer, train_labels, test_labels = loader.get_dataset_from_pool(no_consecutive=False)

            # Step 3: Train and test with selected segments
            logger.info(f'==================== Training & testing with selected segments ====================')
            train_kmer, test_kmer, train_labels, test_labels = loader.get_dataset_from_pool(no_consecutive=False)

            # Run XGBoost with custom metric
            custom_metric = self.model_trainer.custom_essential_agreement_metric()
            fold_results, _ = self.model_trainer.run_xgboost(
                train_kmer, test_kmer, train_labels, test_labels,
                use_partition=False, custom_metric=custom_metric
            )

            logger.info(fold_results)

            # Append fold results
            accumulated_results = ProgressManager.append_results(fold_results, accumulated_results)
            # Save progress after each fold
            self.progress_manager.save_fold_progress(i + 1, accumulated_results)

            # Remove saved segments
            try:
                os.remove(args.save_file)
            except FileNotFoundError as e:
                logger.error(e)

        # Export final results and shutdown Ray
        accumulated_results.to_csv(f'{args.export_file}.csv')
        seg_pool.export(args.export_file)
        ray.shutdown()
