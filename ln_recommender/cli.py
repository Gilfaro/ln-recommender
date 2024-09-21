import argparse
from types import SimpleNamespace


def setup_advanced_cli(parser):
    sp = parser.add_subparsers(
        help="Generate difficulty estimation and some stats",
        required=True,
        dest="command",
    )
    sync = sp.add_parser(
        "eval",
        help="Evaluate text files",
    )

    gen = sp.add_parser(
        "train",
        help="Train estimation model",
    )

    cluster = sp.add_parser(
        "cluster",
        help="Cluster training data",
    )

    main_group = sync.add_argument_group("Main arguments")
    gen_main_group = gen.add_argument_group("Main arguments")
    cluster_main_group = cluster.add_argument_group("Main arguments")

    # Sources
    main_group.add_argument(
        "-d",
        "--dirs",
        dest="dirs",
        default=[],
        required=False,
        type=str,
        nargs="+",
        help="List of folders to pull text from",
    )
    main_group.add_argument(
        "-t",
        "--text",
        nargs="+",
        default=[],
        required=False,
        help="List of paths to single files",
    )
    main_group.add_argument(
        "-f",
        "--freq",
        default="freq.txt",
        required=False,
        help="Path to the frequency file",
    )
    main_group.add_argument(
        "-ml",
        "--model-load",
        default=None,
        required=False,
        help="Path to the model file to load",
    )
    main_group.add_argument(
        "-ms",
        "--model-save",
        default=None,
        required=False,
        help="Path to the model file to save",
    )
    main_group.add_argument(
        "-td",
        "--training-data",
        default="data.csv",
        required=False,
        help="Path to the training data file",
    )
    main_group.add_argument(
        "-it",
        "--iterations",
        default=10000,
        type=int,
        required=False,
        help="Number of iterations to train for",
    )
    main_group.add_argument(
        "-ev",
        "--eval-mode",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use 80%% of data to train and 20%% to evaluate during training",
    )
    main_group.add_argument(
        "-o",
        "--output-csv",
        default=None,
        help="Output csv of analysis and estimation",
    )

    gen_main_group.add_argument(
        "-ms",
        "--model-save",
        default="model.cbm",
        required=False,
        help="Path to the model file to save",
    )
    gen_main_group.add_argument(
        "-td",
        "--training-data",
        default="data.csv",
        required=False,
        help="Path to the training data file",
    )
    gen_main_group.add_argument(
        "-it",
        "--iterations",
        default=10000,
        type=int,
        required=False,
        help="Number of iterations to train for",
    )
    gen_main_group.add_argument(
        "-ev",
        "--eval-mode",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use 80%% of data to train and 20%% to evaluate during training",
    )

    cluster_main_group.add_argument(
        "-td",
        "--training-data",
        default="data.csv",
        required=False,
        help="Path to the training data file",
    )
    cluster_main_group.add_argument(
        "-mcs",
        "--min-cluster-size",
        default=2,
        type=int,
        required=False,
        help="""The minimum number of samples in a group for that group to be considered a cluster;
                groupings smaller than this size will be left as noise""",
    )
    cluster_main_group.add_argument(
        "-ms",
        "--min-samples",
        default=None,
        type=int,
        required=False,
        help="""The number of samples in a neighborhood for a point to be considered as a core point;
                this includes the point itself""",
    )

    return parser


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate difficulty estimation and some stats"
    )

    parser = setup_advanced_cli(parser)
    args = parser.parse_args()
    return args


def get_inputs():
    args = get_args()
    if args.command == "eval":
        inputs = SimpleNamespace(
            command=args.command,
            dirs=args.dirs,
            text=args.text,
            freq=args.freq,
            iterations=args.iterations,
            eval_mode=args.eval_mode,
            training_data=args.training_data,
            model_load=args.model_load,
            model_save=args.model_save,
            output_csv=args.output_csv,
        )
    elif args.command == "train":
        inputs = SimpleNamespace(
            command=args.command,
            iterations=args.iterations,
            eval_mode=args.eval_mode,
            training_data=args.training_data,
            model_save=args.model_save,
        )
    elif args.command == "cluster":
        inputs = SimpleNamespace(
            command=args.command,
            training_data=args.training_data,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
        )

    return inputs
