import argparse
import os

from pgse.etc.alphabet import AUTO, DNA_CHARS


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str,
                        help="Path to the CSV file containing the labels")
    parser.add_argument('--data-dir', type=str,
                        help="Directory containing the data files")
    parser.add_argument('--pre-kfold-info-file', type=str, default=None,
                        help="Path to the JSON file containing the pre-defined k-fold indices")
    parser.add_argument('--save-file', type=str,
                        default='',
                        help="File path to save the selected segments. Used to recover the progress.")
    parser.add_argument('--export-file', type=str,
                        default='./default-export',
                        help="File path to save the results")
    parser.add_argument('--model-file', type=str,
                        default=None,
                        help="File path to save the trained model")
    parser.add_argument('--segments-file', type=str,
                        default=None,
                        help="File path to save the segments")
    parser.add_argument('--k', type=int, default=10,
                        help="Initial size of k-mers")
    parser.add_argument('--ext', type=int, default=2,
                        help="Length of extensions to add in each iteration")
    parser.add_argument('--target', type=int, default=70,
                        help="Target length of segments to reach")
    parser.add_argument('--workers', type=int, default=8,
                        help="Number of CPU workers to allocate per node.")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to use for training. Options: 'cpu', 'gpu', 'gpu:0', etc.")
    parser.add_argument('--nodes', type=int, default=os.environ.get('SLURM_JOB_NUM_NODES', 1),
                        help="Number of nodes allocated. Used with distributed processing only.")
    parser.add_argument('--features', type=int, default=10000,
                        help="Number of top features to select based on importance")
    parser.add_argument('--lr', type=float, default=0.03,
                        help="Learning rate for the XGBoost model")
    parser.add_argument('--dist', type=int, default=0,
                        help="Flag to enable distributed processing")
    parser.add_argument('--ea-max', type=float, default=None,
                        help="Maximum value of MIC (>)")
    parser.add_argument('--ea-min', type=float, default=None,
                        help="Minimum value of MIC (<)")
    parser.add_argument('--num-rounds', type=int, default=1500,
                        help="Number of boosting rounds")
    parser.add_argument('--folds', type=int, default=0,
                        help="Number of folds for cross-validation")
    parser.add_argument('--oversample', type=int, default=0,
                        help="Flag to enable oversampling of the minority class")
    parser.add_argument('--verbose', type=int, default=0,
                        help="0: Error, 1: Warning, 2: Info, 3: Debug")
    parser.add_argument('--alphabet', type=str, default=DNA_CHARS,
                        help="The set of characters the sequences are made of, given as a single "
                             "string. Defaults to DNA ('atgc'). Anything outside the alphabet is "
                             "dropped when reading the input.")
    parser.add_argument('--case-sensitive', type=int, default=0,
                        help="Flag to treat upper and lower case as distinct characters. "
                             "0 (default) folds everything to lower case.")
    parser.add_argument('--complement', type=str, default=AUTO,
                        help="Complement used to canonicalise segments, given as one character per "
                             "character of --alphabet (e.g. 'tacg' for 'atgc'). Defaults to reverse "
                             "complementing for DNA and to no canonicalisation otherwise. "
                             "Pass an empty string to switch it off.")
    parser.add_argument('--uint16', type=int, default=0,
                        help="Flag to store segment counts as uint16 instead of float32, halving the "
                             "count-matrix memory. Lossless for counts up to 65535 (saturated above).")
    parser.add_argument('--sparse', type=int, default=0,
                        help="Flag to store the count matrix as a sparse CSR matrix. Big saving for "
                             "short/sparse sequences (e.g. SMILES). The same flag must be used for "
                             "training and prediction.")
    return parser