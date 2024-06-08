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

    main_group = sync.add_argument_group("Main arguments")
    gen_main_group = gen.add_argument_group("Main arguments")

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
        "-ev",
        "--eval-mode",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use 80%% of data to train and 20%% to evaluate during training",
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
            eval_mode=args.eval_mode,
            training_data=args.training_data,
            model_load=args.model_load,
            model_save=args.model_save,
            output_csv=args.output_csv,
        )
    elif args.command == "train":
        inputs = SimpleNamespace(
            command=args.command,
            eval_mode=args.eval_mode,
            training_data=args.training_data,
            model_save=args.model_save,
        )

    return inputs
