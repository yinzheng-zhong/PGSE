import os


os.environ["RAY_LOG_TO_STDERR"] = "0"
os.environ["RAY_LOG_LEVEL"] = "ERROR"

import argparse

from src.dataset.file_label import FileLabel
from src.dataset.loader import Loader
from src.segment.extender import Extender
from src.model.xgb import XGBoost
from src.segment import seg_manager
from src.etc.config import config
import ray

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=8)
parser.add_argument('--ext', type=int, default=2)
parser.add_argument('--target', type=int, default=70)
parser.add_argument('--save-file', type=str, default='src-70.txt')
parser.add_argument('--workers', type=int, default=38)
parser.add_argument('--features', type=int, default=1000)
parser.add_argument('--dist', type=int, default=0)
args = parser.parse_args()

K = args.k
EXTENSIONS = args.ext
TARGET = args.target
SAVE_FILE = args.save_file
NUM_WORKERS = args.workers
MAX_FEATURES = args.features

file_label = FileLabel(config['label_file'], config['data_dir'])
extender = Extender()

# Initialize Ray
if args.dist:
    ray.init(address='auto')
else:
    ray.init(num_cpus=NUM_WORKERS)

loader = Loader(file_label)

# check if the save file exists
try:
    seg_manager.load(SAVE_FILE)
    train_kmer, test_kmer, train_labels, test_labels = loader.get_extended_dataset()

    print(train_kmer)
except FileNotFoundError:
    seg_manager.add_all_kmer(K)
    train_kmer, test_kmer, train_labels, test_labels = loader.get_kmer_dataset(K)

while True:
    if K > TARGET:
        break

    xgb = XGBoost()
    results_df, importance_df = xgb.run(train_kmer, test_kmer, train_labels, test_labels)

    print(importance_df)

    index = list(map(int, importance_df['Feature'].str.replace('f', '').values))[:MAX_FEATURES]
    seg_manager.use_subset(index)
    # do the pruning first otherwise only longer segments will be kept
    seg_manager.segments_pruning(range(len(index)))
    extender.extend_all_segs(EXTENSIONS)

    seg_manager.save(SAVE_FILE)

    train_kmer, test_kmer, train_labels, test_labels = loader.get_extended_dataset()

    K += EXTENSIONS
