from argparse import ArgumentParser
from pathlib import Path

from loguru import logger

from llama.train_unsloth import fourbit_models
from utils import seed_everything

from llama.cot import CoTConfigs
from ssc.pipeline import annotate_and_evaluate_files
from unsloth import FastLanguageModel

parser = ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--test_set', type=str, required=True)
parser.add_argument('--cot_config', type=str, required=False, default="no_cot")


def notify(message):
    print(message)


args = parser.parse_args()

model_identifier = args.model
cot_config = CoTConfigs[args.cot_config]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_identifier,
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

seed_everything(42)

FastLanguageModel.for_inference(model)

dataset_path = datasets_folder

if args.test_set == "all":
    test_folder = dataset_path / "test_full"
    test_documents = [file for file in Path(test_folder).iterdir() if file.name.endswith(".xmi.zip")]

    if model_identifier in fourbit_models:
        output_dir = Path(model_identifier).absolute() / cot_config.name / test_folder.name
    else:
        output_dir = Path(model_identifier).absolute() / test_folder.name
    logger.info(f"Output directory: {output_dir}")
    results = annotate_and_evaluate_files(test_documents, model, coarse=True, tolerance=3,
                                          output_dir=output_dir,
                                          tokenizer=tokenizer, cot_config=cot_config.value)

    print(results)
    for folder in dataset_path.iterdir():
        if folder.is_dir() and "test" in folder.name:
            test_folder = dataset_path / folder.name
            test_documents = [file for file in Path(test_folder).iterdir() if file.name.endswith(".xmi.zip")]

            if model_identifier in fourbit_models:
                output_dir = Path(model_identifier).absolute() / cot_config.name / test_folder.name
            else:
                output_dir = Path(model_identifier).absolute() / test_folder.name
            logger.info(f"Output directory: {output_dir}")
            results = annotate_and_evaluate_files(test_documents, model, coarse=True, tolerance=3,
                                                  output_dir=output_dir,
                                                  tokenizer=tokenizer, cot_config=cot_config.value)

            print(results)

else:
    test_folder = dataset_path / args.test_set
    test_documents = [file for file in Path(test_folder).iterdir() if file.name.endswith(".xmi.zip")]

    if model_identifier in fourbit_models:
        output_dir = Path(model_identifier).absolute() / cot_config.name / test_folder.name
    else:
        output_dir = Path(model_identifier).absolute() / test_folder.name
    logger.info(f"Output directory: {output_dir}")
    results = annotate_and_evaluate_files(test_documents, model, coarse=True, tolerance=3,
                                          output_dir=output_dir,
                                          tokenizer=tokenizer, cot_config=cot_config.value)

    print(results)

notify("Evaluation finished.")
