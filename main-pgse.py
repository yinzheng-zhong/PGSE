from pgse.environment.args import get_parser
from pgse import TrainingPipeline

if __name__ == "__main__":
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
        device=args.device,
        alphabet=args.alphabet,
        case_sensitive=bool(args.case_sensitive),
        complement=args.complement,
    )
    pipeline.run()