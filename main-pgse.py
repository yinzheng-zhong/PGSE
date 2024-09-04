import math

from src.log import logger
import os

from src.model.util import essential_agreement_cus_metric

os.environ["RAY_LOG_TO_STDERR"] = "0"
os.environ["RAY_LOG_LEVEL"] = "ERROR"

from src.enviromnet import args

from src.dataset.file_label import FileLabel
from src.dataset.loader import Loader
from src.segment.extender import Extender
from src.model.xgb import XGBoost
from src.segment import seg_pool
import ray



file_label = FileLabel(args.label_file, args.data_dir)
extender = Extender()

# Initialize Ray
if args.dist:
    ray.init(address='auto', log_to_driver=True)
    logger.warning(
        f'Connected to Ray cluster with {args.nodes} nodes and {args.workers} workers per node.\n'
        f'Sometimes the progress bar may seem frozen, but it is still running.'
    )
else:
    ray.init(num_cpus=args.workers, log_to_driver=True)

loader = Loader(file_label)

# check if the save file exists
try:
    seg_pool.load(args.save_file)
    train_kmer, test_kmer, train_labels, test_labels = loader.get_dataset_from_pool(no_consecutive=False)

    print(train_kmer)
except FileNotFoundError:
    seg_pool.add_all_kmer(args.k, args.ext)
    train_kmer, test_kmer, train_labels, test_labels = loader.get_kmer_dataset(args.k, no_consecutive=False)

while True:
    if args.k > args.target:
        break

    logger.info(f'==================== Feature Selection ====================')
    xgb = XGBoost(
        boost_rounds=1000,
        num_cpu_per_node=args.workers,
        partition_size=5000,
        learning_rate=args.lr,
    )

    results_df, importance_df = xgb.run(train_kmer, test_kmer, train_labels, test_labels)

    logger.info(str(importance_df.head(20)))

    index = list(map(int, importance_df['Feature'].values))[:args.features]
    seg_pool.use_subset(index)
    # do the pruning first otherwise only longer segments will be kept
    seg_pool.redundant_elimination(range(len(index)))
    #seg_pool.n_gram_grafting()

    # test with single training
    logger.info(f'==================== Training & testing with selected segments ====================')
    train_kmer, test_kmer, train_labels, test_labels = loader.get_dataset_from_pool(no_consecutive=False)
    xgb = XGBoost(
        boost_rounds=1000,
        num_cpu_per_node=args.workers,
        learning_rate=args.lr,
        custom_metric=lambda x, y : essential_agreement_cus_metric(
            x, y,
            min_after_log2=math.log2(args.ea_min) if args.ea_min is not None else None,
            max_after_log2=math.log2(args.ea_max) if args.ea_max is not None else None
        )
    )

    _, _ = xgb.run(train_kmer, test_kmer, train_labels, test_labels)

    # This will change the order
    try:
        extender.extend_all_segs(args.ext)
    except ValueError:
        break

    seg_pool.save(args.save_file)

    train_kmer, test_kmer, train_labels, test_labels = loader.get_dataset_from_pool(no_consecutive=False)

    args.k += args.ext

seg_pool.export(args.export_file)
ray.shutdown()
