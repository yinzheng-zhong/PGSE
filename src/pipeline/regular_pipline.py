import pandas as pd
import ray

from src.enviromnet.ray_env import RayEnvManager
from src.log import logger
from src.model.model_trainer import ModelTrainer
from src.enviromnet import args
from src.dataset.file_label import FileLabel
from src.dataset.loader import Loader
from src.segment import seg_pool


class Pipeline:
    def __init__(self):
        self.file_label = FileLabel(args.label_file, args.data_dir, args.pre_kfold_info_file)

    def run(self):
        RayEnvManager.initialize()

        accumulated_results = pd.DataFrame()

        # Use k-mer data only without any feature selection or partitioning
        for i in range(args.folds if args.folds > 0 else 1):
            logger.info(f'==================== Fold {i + 1} ====================')
            loader = Loader(
                self.file_label,
                folds=args.folds,
                fold_index=i
            )

            model_trainer = ModelTrainer(loader)

            # Load k-mer dataset
            seg_pool.clear()
            seg_pool.add_all_kmer(args.k, args.ext)
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
            trained_model.save_model(f'{args.export_file}_regular_xgboost_fold_{i}')


        # Export final results and shutdown Ray
        accumulated_results.to_csv(f'{args.export_file}_regular_xgboost.csv', index=False)
        ray.shutdown()
