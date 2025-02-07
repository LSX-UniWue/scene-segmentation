import os
import shutil
from pathlib import Path
from typing import List, Union
from collections import Counter

import numpy as np
import torch
import transformers
from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

from transformers import PreTrainedTokenizer, AutoTokenizer, TrainingArguments, Trainer, set_seed
from datasets import Dataset, Sequence

from ssc.dataset import SSCDataset
from ssc.model import SSCModel, SSCModelConfig, Stride, LabelSet, EmbeddingModelType
from utils.constants import datasets_folder

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.device("mps") else "cpu"

set_seed(42)


def tokenize_function(tokenizer: PreTrainedTokenizer, sep_token: str, examples):
    data = tokenizer(sep_token.join(examples["sentences"]) + sep_token, padding="max_length", truncation=False)
    sep_token_indices = examples.data["sep_token_indices"]
    if sep_token_indices is None:
        str(examples)
    sep_token_indices = examples.data["sep_token_indices"]
    all_sep_ids = {data.data["input_ids"][i] for i in sep_token_indices}
    assert len(
        all_sep_ids) == 1, f"Found multiple sep ids: {all_sep_ids}. Should all be the same. Something went wrong in tokenisation! {data.data['input_ids']}"
    return data


def compute_metrics(pred):
    labels = pred.label_ids.flatten()

    preds = pred.predictions[0].argmax(-1).flatten()
    preds = [preds[i] for i in range(len(labels)) if labels[i] != -100]
    labels = [labels[i] for i in range(len(labels)) if labels[i] != -100]
    no_border_class = Counter(labels).most_common(1)[0][0]
    if len(set(labels)) == 2:
        border_class = 1 - no_border_class
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary",
                                                                   pos_label=border_class)
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="micro",
                                                                   labels=list(set(labels) - {no_border_class}))
    acc = accuracy_score(labels, preds)
    metrics = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

    return metrics


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        # logits = outputs.get('logits')
        loss = outputs.get('loss')
        # compute custom loss
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1, 0.3]).half().to(DEVICE))
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def run_training(context_size: int, eval_batch_size: int, label_set: LabelSet, model_name: str, num_train_epochs: int,
                 only_last_layer: bool, output_dir: str, stride: Stride,
                 test_folder_or_files: Union[Path, List[Path]], train_batch_size: int,
                 train_folder_or_files: Union[Path, List[Path]], tokenizer_name: str,
                 has_mems: bool, train_embedding_model: bool, lstm_num_layers: int, lstm_hidden_size: int,
                 drop_noninformative: bool = False, random_seed: int = 42, no_mlp: bool = False, ):
    logger.info(f"Setting random seed to {random_seed}")
    transformers.set_seed(random_seed)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    additional_heads = []

    logger.add(output_dir + "/train.log", rotation="10 MB", backtrace=True, diagnose=True, level="DEBUG")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "dataset.txt").write_text(f"{train_folder_or_files}")

    model_config = SSCModelConfig(embedding_model_name=model_name,
                                  stride=stride,
                                  only_last_layer=only_last_layer,
                                  additional_heads=additional_heads,
                                  label_set=label_set.value,
                                  context_size=context_size,
                                  tokenizer_name=tokenizer_name,
                                  has_mems=has_mems,
                                  train_embedding_model=train_embedding_model,
                                  lstm_num_layers=lstm_num_layers,
                                  lstm_layer_size=lstm_hidden_size,
                                  no_mlp=no_mlp,
                                  )
    model = SSCModel(model_config)

    if isinstance(test_folder_or_files, Path):
        train_files = [file for file in train_folder_or_files.iterdir() if ".xmi" in file.name]
    else:
        train_files = test_folder_or_files
    if isinstance(test_folder_or_files, Path):
        test_files = [file for file in test_folder_or_files.iterdir() if ".xmi" in file.name]
    else:
        test_files = test_folder_or_files

    train_ds, raw_train_ds = get_dataset(model_config, stride, tokenizer, train_files,
                                         drop_noninformative=drop_noninformative, coarse=label_set == LabelSet.Coarse)

    model_config.class_weights = compute_class_weight("balanced", classes=np.array([1, 0]),
                                                      y=[0] * raw_train_ds.num_borders + [
                                                          1] * raw_train_ds.num_nonborders)

    test_ds, raw_test_ds = get_dataset(model_config, stride, tokenizer, test_files, drop_noninformative=False,
                                       coarse=label_set == LabelSet.Coarse)

    if model_config.model_type == EmbeddingModelType.LLM:
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            evaluation_strategy="epoch",
            label_names=["labels"] + (["additional_head_labels"] if additional_heads else []),
            num_train_epochs=num_train_epochs,
            use_mps_device=False,
            logging_steps=50,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            optim="adamw_bnb_8bit",
            learning_rate=1e-6,
            # bf16=True,
            fp16=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
        )

    else:
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            evaluation_strategy="epoch",
            label_names=["labels"] + (["additional_head_labels"] if additional_heads else []),
            num_train_epochs=num_train_epochs,
            use_mps_device=False,
            logging_steps=50,
            learning_rate=1e-6 if model_config.train_embedding_model else 1e-3,
            past_index=3 if model_config.has_mems else -1,
            save_total_limit=2,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    logger.info(f"Training starts with {model_config}")
    logger.info(f"Training on {train_folder_or_files} with {len(train_ds)} samples.")
    logger.info(f"Evaluating on {test_folder_or_files} with {len(test_ds)} samples.")

    trainer.train()
    trainer.save_model()
    print(trainer.evaluate(test_ds))

    return model


def get_dataset(model_config, stride, tokenizer, files, drop_noninformative=False, coarse=True):
    raw_ds = SSCDataset(files, tokenizer=tokenizer, context_size=model_config.context_size, stride=stride,
                        coarse=coarse, drop_noninformative=drop_noninformative)
    ds = Dataset.from_generator(raw_ds.yield_samples)
    ds = ds.cast_column("labels", Sequence(model_config.label_set))
    ds = ds.map(lambda samples: tokenize_function(tokenizer, model_config.sep_token, samples),
                batched=False, keep_in_memory=True)
    ds = ds.remove_columns(["sentences"])
    return ds, raw_ds


def main():
    small_ds = DEVICE != "cuda"
    _model = "bert"
    output_dir = "temp/test_trainer"

    train_batch_size = 1
    eval_batch_size = 1
    num_train_epochs = 5
    stride = Stride.Full
    label_set = LabelSet.Fine
    only_last_layer = False
    has_mems = False
    train_embedding_model = True
    lstm_num_layers = 0
    lstm_hidden_size = 512
    drop_noninformative = False
    no_mlp = True

    if _model == "llama":
        if small_ds:
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        else:
            model_name = "meta-llama/Meta-Llama-3-8B"
        model_name = "astronomer/Llama-3-8B-Special-Tokens-Adjusted"
        tokenizer_name = "astronomer/Llama-3-8B-Special-Tokens-Adjusted"
    elif _model == "bert":
        if small_ds:
            model_name = "deepset/gbert-base"
        else:
            model_name = "deepset/gbert-large"
        tokenizer_name = "deepset/gbert-large"
    elif _model == "xlnet":
        model_name = "/Users/zehe/PycharmProjects/scenes-general/lms/XLitNet-large-slow"
        tokenizer_name = "xlnet-base-cased"
    else:
        raise NotImplementedError
    context_size = 512

    train_folder = datasets_folder / "stss_train"
    test_folder = datasets_folder / "stss_test_1"

    files_small = [datasets_folder / "ood_test" / "Harry Potter und der Halbblutprinz - Kapitel Schleim*.xmi.zip"]
    if small_ds:
        train_folder = files_small
        test_folder = files_small

    run_training(context_size=context_size,
                 eval_batch_size=eval_batch_size,
                 label_set=label_set,
                 model_name=model_name,
                 num_train_epochs=num_train_epochs,
                 only_last_layer=only_last_layer,
                 output_dir=output_dir,
                 stride=stride,
                 test_folder_or_files=test_folder,
                 train_batch_size=train_batch_size,
                 train_folder_or_files=train_folder,
                 tokenizer_name=tokenizer_name,
                 has_mems=has_mems,
                 train_embedding_model=train_embedding_model,
                 lstm_num_layers=lstm_num_layers,
                 lstm_hidden_size=lstm_hidden_size,
                 drop_noninformative=drop_noninformative,
                 no_mlp=no_mlp, )


if __name__ == '__main__':
    main()
