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
parser.add_argument('--label-file', type=str, default='../volatile/e_coli_mic_label.csv',
                    help="Path to the CSV file containing the labels")
parser.add_argument('--data-dir', type=str, default='../volatile/e_coli_mic/',
                    help="Directory containing the data files")
parser.add_argument('--save-file', type=str, default='../volatile/var/test-70.txt',
                    help="File path to save the selected segments. Used to recover the progress.")
parser.add_argument('--export-file', type=str, default='../volatile/var/pgse-result.txt',
                    help="File path to save the results")
parser.add_argument('--k', type=int, default=8,
                    help="Initial size of k-mers")
parser.add_argument('--ext', type=int, default=2,
                    help="Length of extensions to add in each iteration")
parser.add_argument('--target', type=int, default=70,
                    help="Target length of segments to reach")
parser.add_argument('--workers', type=int, default=38,
                    help="Number of CPU workers to allocate. Used with single node only.")
parser.add_argument('--features', type=int, default=1000,
                    help="Number of top features to select based on importance")
parser.add_argument('--dist', type=int, default=0,
                    help="Flag to enable distributed processing")
args = parser.parse_args()

file_label = FileLabel(args.label_file, args.data_dir)
extender = Extender()

# Initialize Ray
if args.dist:
    ray.init(address='auto')
else:
    ray.init(num_cpus=args.workers)

loader = Loader(file_label)

# check if the save file exists
try:
    seg_pool.load(args.save_file)
    train_kmer, test_kmer, train_labels, test_labels = loader.get_extended_dataset()

    print(train_kmer)
except FileNotFoundError:
    seg_pool.add_all_kmer(args.k)
    train_kmer, test_kmer, train_labels, test_labels = loader.get_kmer_dataset(args.k)

while True:
    if args.k > args.target:
        break

    xgb = XGBoost()
    results_df, importance_df = xgb.run(train_kmer, test_kmer, train_labels, test_labels)

    print(importance_df)

    index = list(map(int, importance_df['Feature'].str.replace('f', '').values))[:args.features]
    seg_pool.use_subset(index)
    # do the pruning first otherwise only longer segments will be kept
    seg_pool.redundant_elimination(range(len(index)))

    # This will change the order
    try:
        extender.extend_all_segs(args.ext)
    except ValueError:
        break

    seg_pool.save(args.save_file)

    train_kmer, test_kmer, train_labels, test_labels = loader.get_extended_dataset()

    args.k += args.ext

seg_pool.save(args.save_file)
ray.shutdown()