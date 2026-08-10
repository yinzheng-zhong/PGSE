import os

from pgse.environment.args import get_parser
from pgse import TrainingPipeline, InferencePipeline
from pgse.log import logger

SEQUENCE_SUFFIXES = ('.fna', '.fasta')
MAX_LOGGED_PREDICTIONS = 20


def train():
    parser = get_parser()
    args = parser.parse_args()

    if args.log_file:
        logger.add_file_handler(args.log_file)

    pipeline = TrainingPipeline(
        args.data_dir,
        args.label_file,
        args.pre_kfold_info_file,
        args.save_file,
        args.export_file,
        args.k,
        args.ext,
        args.target,
        args.features,
        args.folds,
        args.ea_min,
        args.ea_max,
        args.num_rounds,
        args.lr,
        args.dist,
        args.nodes,
        args.workers,
        device=args.device,
        alphabet=args.alphabet,
        case_sensitive=bool(args.case_sensitive),
        complement=args.complement,
        uint16=bool(args.uint16),
        sparse=bool(args.sparse),
        partition_size_target=args.partition_size_target,
        metric=args.metric,
        binary=bool(args.binary),
        table_file=args.table_file,
        data_column=args.data_column,
        label_column=args.label_column
    )
    pipeline.run()

def predict():
    import pandas as pd
    parser = get_parser()
    args = parser.parse_args()

    if args.log_file:
        logger.add_file_handler(args.log_file)

    if args.model_file is None:
        raise ValueError("Model file must be specified for prediction.")
    if args.segments_file is None:
        raise ValueError("Segments file must be specified for prediction.")

    pipeline = InferencePipeline(
        args.model_file,
        args.segments_file,
        workers=args.workers,
        alphabet=args.alphabet,
        case_sensitive=bool(args.case_sensitive),
        complement=args.complement,
        uint16=bool(args.uint16),
        sparse=bool(args.sparse)
    )

    if args.table_file:
        if not args.data_column:
            raise ValueError("--data-column must be specified alongside --table-file.")
        inputs = pd.read_csv(args.table_file)[args.data_column].astype(str).tolist()
        results = pipeline.run(sequences=inputs)
        input_column = args.data_column
    else:
        inputs = [
            os.path.join(args.data_dir, f)
            for f in os.listdir(args.data_dir) if f.endswith(SEQUENCE_SUFFIXES)
        ]
        results = pipeline.run(files=inputs)
        input_column = 'file'

    _log_predictions(inputs, results)

    # save a csv file with the results
    output_file = args.export_file
    if output_file is None:
        logger.warning("Export file not specified. Results will not be saved.")

    if not output_file.endswith('.csv'):
        output_file += '.csv'

    df = pd.DataFrame({
        input_column: inputs,
        'prediction': results
    })

    df.to_csv(output_file, index=False)


def _log_predictions(inputs: list[str], results) -> None:
    """Log the first predictions, and how many more there are.

    Args:
        inputs: The file paths, or the sequences, that were scored.
        results: The prediction of each input, in the same order.
    """
    formatted_results = [
        f"Input: {sample}, Prediction: {result}"
        for sample, result in zip(inputs[:MAX_LOGGED_PREDICTIONS], results[:MAX_LOGGED_PREDICTIONS])
    ]

    if len(inputs) > MAX_LOGGED_PREDICTIONS:
        formatted_results.append(f"... and {len(inputs) - MAX_LOGGED_PREDICTIONS} more")

    logger.info("\n".join(formatted_results))
