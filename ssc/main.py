import argparse
import json
from pathlib import Path
from typing import List

from loguru import logger

from ssc.model import LabelSet, Stride
from ssc.pipeline import annotate_and_evaluate_files
from ssc.train import run_training
from utils.constants import datasets_folder

datasets = [dir for dir in datasets_folder.iterdir() if dir.is_dir()]
test_datasets = [dir for dir in datasets if "test" in dir.name]


def parse_label_set(value: str) -> LabelSet:
    try:
        return LabelSet[value]
    except KeyError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid {LabelSet.__name__}")


def main():
    parser = argparse.ArgumentParser(description="Train a language model with specific parameters.")

    parser.add_argument('--context_size', type=int, required=True, help='The context size for the training.')
    parser.add_argument('--eval_batch_size', type=int, required=True, help='Batch size for evaluation.')
    parser.add_argument('--label_set', type=parse_label_set, required=True, help='Label set for the datasets.', )
    parser.add_argument('--model_name', type=str, required=True, help='The name of the model.')
    parser.add_argument('--num_train_epochs', type=int, required=True, help='Number of training epochs.')
    parser.add_argument('--only_last_layer', action='store_true', help='If true, only the last layer will be trained.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where the output will be saved.')
    parser.add_argument('--small_ds', action='store_true', help='If true, use a smaller dataset.')
    parser.add_argument('--stride', type=Stride, required=True, help='Stride for the input sequences.')
    parser.add_argument('--test_files', type=Path, required=True, help='List of test files.')
    parser.add_argument('--train_batch_size', type=int, required=True, help='Batch size for training.')
    parser.add_argument('--train_files', type=Path, required=True, help='List of training files.')
    parser.add_argument('--tokenizer_name', type=str, required=True, help='Name of the tokenizer to be used.')
    parser.add_argument('--has_mems', action='store_true', help='If true, the model has memory states.')
    parser.add_argument('--train_embedding_model', action='store_true', help='If true, train the embedding model.')
    parser.add_argument('--lstm_num_layers', type=int, help='Number of LSTM layers.')
    parser.add_argument('--lstm_hidden_size', type=int, help='Hidden size of the LSTM layers.')
    parser.add_argument("--drop_noninformative", action="store_true",
                        help="Drop non-informative samples with no scene borders from the dataset.")
    parser.add_argument("--random_seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument("--no_mlp", action="store_true", help="Do not use the MLP layer for classification.")

    args = parser.parse_args()
    model = run_training(
        context_size=args.context_size,
        eval_batch_size=args.eval_batch_size,
        label_set=args.label_set,
        model_name=args.model_name,
        num_train_epochs=args.num_train_epochs,
        only_last_layer=args.only_last_layer,
        output_dir=args.output_dir,
        stride=args.stride,
        test_folder_or_files=args.test_files,
        train_batch_size=args.train_batch_size,
        train_folder_or_files=args.train_files,
        tokenizer_name=args.tokenizer_name,
        has_mems=args.has_mems,
        train_embedding_model=args.train_embedding_model,
        lstm_num_layers=args.lstm_num_layers,
        lstm_hidden_size=args.lstm_hidden_size,
        drop_noninformative=args.drop_noninformative,
        random_seed=args.random_seed,
        no_mlp=args.no_mlp,
    )

    for test_ds in test_datasets:
        logger.info(f"Testing on {test_ds.name}...")
        files = [file for file in test_ds.iterdir() if ".xmi.zip" in file.name]
        output_dir = Path(args.output_dir) / test_ds.name
        r = annotate_and_evaluate_files(files, model, coarse=True, tolerance=3, output_dir=output_dir)
        (output_dir / "results.json").write_text(json.dumps(r))
        logger.info(f"Results: {r}")


if __name__ == "__main__":
    main()
