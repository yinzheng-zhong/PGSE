import os

from pgse.environment.args import get_parser
from pgse import TrainingPipeline, InferencePipeline
from pgse.log import logger


def train():
    parser = get_parser()
    args = parser.parse_args()

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
        device=args.device
    )
    pipeline.run()

def predict():
    import pandas as pd
    parser = get_parser()
    args = parser.parse_args()

    if args.model_file is None:
        raise ValueError("Model file must be specified for prediction.")
    if args.segments_file is None:
        raise ValueError("Segments file must be specified for prediction.")

    input_dir = args.data_dir

    # discover the list of files in the input directory
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.fna') or f.endswith('.fasta')]

    pipeline = InferencePipeline(args.model_file, args.segments_file, workers=args.workers)
    results = pipeline.run(input_files)

    # format the results for printing
    formatted_results = []
    for file, result in zip(input_files, results):
        formatted_results.append(f"File: {file}, Prediction: {result}")

    logger.info(
        "\n".join(formatted_results)
    )

    # save a csv file with the results
    output_file = args.export_file
    if output_file is None:
        logger.warning("Export file not specified. Results will not be saved.")

    if not output_file.endswith('.csv'):
        output_file += '.csv'

    df = pd.DataFrame({
        'file': input_files,
        'prediction': results
    })

    df.to_csv(output_file, index=False)