from src.log import logger
import os

os.environ["RAY_LOG_TO_STDERR"] = "0"
os.environ["RAY_LOG_LEVEL"] = "ERROR"

import argparse

from src.dataset.file_label import FileLabel
from src.dataset.loader import Loader
from src.segment.extender import Extender
from src.model.xgb import XGBoost
from src.segment import seg_pool
import ray

parser = argparse.ArgumentParser()
parser.add_argument('--label-file', type=str, required=True,
                    help="Path to the CSV file containing the labels")
parser.add_argument('--data-dir', type=str, required=True,
                    help="Directory containing the data files")
parser.add_argument('--save-file', type=str, required=True,
                    help="File path to save the selected segments. Used to recover the progress.")
parser.add_argument('--export-file', type=str, required=True,
                    help="File path to save the results")
parser.add_argument('--k', type=int, default=8,
                    help="Initial size of k-mers")
parser.add_argument('--ext', type=int, default=2,
                    help="Length of extensions to add in each iteration")
parser.add_argument('--target', type=int, default=70,
                    help="Target length of segments to reach")
parser.add_argument('--workers', type=int, default=38,
                    help="Number of CPU workers to allocate per node.")
parser.add_argument('--nodes', type=int, default=os.environ.get('SLURM_JOB_NUM_NODES', 1),
                    help="Number of nodes allocated. Used with distributed processing only.")
parser.add_argument('--features', type=int, default=10000,
                    help="Number of top features to select based on importance")
parser.add_argument('--lr', type=float, default=0.03,
                    help="Learning rate for the XGBoost model")
parser.add_argument('--dist', type=int, default=0,
                    help="Flag to enable distributed processing")
args = parser.parse_args()

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

    xgb = XGBoost(
        boost_rounds=1000,
        num_cpu_per_node=args.workers,
        partition_size=10,
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
    train_kmer, test_kmer, train_labels, test_labels = loader.get_dataset_from_pool(no_consecutive=False)
    xgb = XGBoost(
        boost_rounds=1000,
        num_cpu_per_node=args.workers,
        partition_size=1,
        learning_rate=args.lr,
    )

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
