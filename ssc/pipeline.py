import json
import re
import shutil
from ast import literal_eval
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Dict, Union

from peft import PeftModelForCausalLM
from tqdm import tqdm

try:
    from unsloth import FastLanguageModel
except ImportError:
    # no unsloth in development environment
    from transformers import AutoModelForCausalLM as FastLanguageModel

import torch
from loguru import logger

from ssc.evaluation import evaluate_predictions, evaluate_predictions_iou

torch.backends.mps.is_available = lambda: False

from datasets import Dataset, Sequence
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, Trainer, AutoTokenizer, pipeline, DataCollatorForLanguageModeling, \
    LlamaForCausalLM, AutoModelForCausalLM
from wuenlp.impl.UIMANLPStructs import UIMADocument, DocumentList

from ssc.dataset import doc_to_window_samples, SSCDataset, get_label, xmi_to_llama_samples, doc_to_llama_samples
from llama.cot import CoTConfigs, CoTConfig
from ssc.model import SSCModel, LabelSet, Stride, DEVICE
from ssc.train import tokenize_function
from utils.constants import datasets_folder

import ssc.json_encoder


def annotate_and_evaluate_files(files: List[Path],
                                model: Union[SSCModel, FastLanguageModel],
                                coarse: bool = False,
                                tolerance: int = 1,
                                output_dir: Optional[Path] = None,
                                context_size: int = None,
                                tokenizer=None,
                                cot_config: CoTConfig = CoTConfigs.no_cot.value) -> Dict[str, Dict[str, float]]:
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for file in files:
        results[file.name] = annotate_and_evaluate_file(file, model, coarse, tolerance,
                                                        output_file=(output_dir / file.name) if output_dir else None,
                                                        context_size=context_size, tokenizer=tokenizer,
                                                        cot_config=cot_config)

        logger.info(f"Results for {file.name}: {results[file.name]}")

    results["average_f1"] = sum([result["f1"] for result in results.values()]) / len(results)

    if output_dir:
        with open(output_dir / "results.json", "w") as f:
            f.write(json.dumps(results))
    return results


def annotate_and_evaluate_file(file: Path, model: Union[SSCModel, FastLanguageModel], coarse: bool = False,
                               tolerance: int = 1,
                               output_file: Optional[Path] = None,
                               context_size: int = None,
                               tokenizer=None,
                               cot_config: CoTConfig = CoTConfigs.no_cot.value) -> Dict[str, float]:
    if output_file:
        potential_full_dir = output_file.parent.parent / "test_full"
        full_file = potential_full_dir / file.name
    else:
        full_file = None
    if output_file and output_file.is_file():
        logger.warning(f"Output file {output_file} already exists. Using existing file.")
        doc = UIMADocument.from_xmi(output_file)
    elif full_file and full_file.is_file():
        shutil.copy(full_file, output_file)
        logger.info(f"Using file {full_file} as output file.")
        doc = UIMADocument.from_xmi(output_file)
    else:
        doc = UIMADocument.from_xmi(file)
        doc = annotate_document(doc, model, context_size, tokenizer, cot_config=cot_config)
    if output_file:
        doc.serialize(output_file)

    predictions = evaluate_document(coarse, doc, tolerance)
    return predictions


def evaluate_file(file: Path, coarse: bool = False, tolerance: int = 1, iou: bool = False):
    logger.info(f"Evaluating file {file}")
    doc = UIMADocument.from_xmi(file)
    predictions = evaluate_document(coarse, doc, tolerance, iou=iou)
    return predictions


def evaluate_document(coarse, doc, tolerance, iou: bool = False):
    true_labels = [get_label(sentence, coarse) for sentence in doc.sentences]
    predicted_labels = [get_label(sentence, coarse, get_predicted_label=True) for sentence in doc.sentences]
    if iou:
        results = evaluate_predictions_iou(true_labels, predicted_labels)
    else:
        results = evaluate_predictions(true_labels, predicted_labels, tolerance=tolerance)
    return results


def evaluate_files(files: List[Path], coarse: bool = False, tolerance: int = 1, iou: bool = False):
    results = {}
    for file in files:
        results[file.name] = evaluate_file(file, coarse, tolerance, iou=iou)
    if iou is False:
        results["average_f1"] = sum([result["f1"] for result in results.values()]) / len(results)
    else:
        results["average_iou"] = sum([result for result in results.values()]) / len(results)
    return results


def annotate_and_evaluate_document(coarse,
                                   doc: UIMADocument,
                                   model: Union[SSCModel, FastLanguageModel],
                                   tolerance: int,
                                   output_file: Optional[Path] = None,
                                   cot: bool = False) -> Dict[str, float]:
    if output_file and output_file.is_file():
        logger.warning(f"Output file {output_file} already exists. Using existing file.")
        doc = UIMADocument.from_xmi(output_file)
    else:
        doc = annotate_document(doc, model, cot=cot)
    if output_file:
        doc.serialize(output_file)
    results = evaluate_document(coarse, doc, tolerance)
    return results


def annotate_files(files: List[Path], model: SSCModel, output_folder: Path, context_size: int = None,
                   tokenizer=None):
    for file in files:
        annotate_file(file, model, output_folder, context_size=context_size, tokenizer=tokenizer)


def annotate_file(file, model,
                  output_folder: Path,
                  context_size: int = None,
                  tokenizer=None,
                  cot_config: CoTConfig = CoTConfigs.no_cot.value):
    doc = UIMADocument.from_xmi(file)
    doc = annotate_document(doc, model, context_size, tokenizer, cot_config=cot_config)
    doc.serialize(output_folder / file.name)


Prediction = namedtuple("Prediction", ["predicted_label", "border_distance", "logits"])


class InvalidOutputError(Exception):
    pass


def parse_unsloth_output(outputs: List[str], sample: Dict, tokenizer, cot_config: CoTConfig) -> (List[bool], List[str]):
    predictions, reasons = [], []
    for i, output in enumerate(outputs):
        response_start = '<|start_header_id|>assistant<|end_header_id|>\n\n'
        if response_start not in output:
            logger.error(
                f"Output does not contain response start: {output}\nInput was: {''.join(tokenizer.batch_decode(sample['input_ids'][i]))}")
        lines = output.split(response_start)[-1].split(",")
        if cot_config is CoTConfigs.no_cot.value:
            _parse_non_cot_unsloth(output, predictions, reasons, response_start, lines)
        else:
            _parse_cot_unsloth(",".join(lines), predictions, reasons, response_start, lines, cot_config)

    return predictions, reasons


def _parse_cot_unsloth(output, predictions, reasons, response_start, lines, cot_config):
    match = cot_config.regex.search(output)
    if not match:
        logger.warning(f"Output does not match expected pattern: {output}")
        predictions.append(False)
        reasons.append("Invalid output from Unsloth model")
    else:
        prediction = "therefore the sentence starts a new scene" in match.group("border")
        reason = output

        predictions.append(prediction)
        reasons.append(reason)


def _parse_non_cot_unsloth(output, predictions, reasons, response_start, lines):
    try:
        assert not ("False" in lines[0] and "True" in lines[0]), f"Output contains both True and False: {output}"
        assert "False" in lines[0] or "True" in lines[0], f"Output does not contain True or False: {output}"
        prediction = "False" not in lines[0]
        reason = ", ".join(lines).strip()

        predictions.append(prediction)
        reasons.append(reason.replace("<|end_of_text|>", ""))
    except Exception as e:
        if not isinstance(e, AssertionError):
            logger.error(f"Invalid output from Unsloth model: {output}")
        logger.error(e)
        predictions.append(False)
        reasons.append("Invalid output from Unsloth model")


def annotate_document(doc: UIMADocument,
                      model: Union[SSCModel, PeftModelForCausalLM],
                      context_size: int = None,
                      tokenizer=None,
                      cot_config: CoTConfig = CoTConfigs.no_cot.value,
                      overwrite: bool = True):
    if len(doc.system_scenes) > 0:
        if overwrite:
            logger.warning("Document already contains system scenes. Overwriting.")
            for ss in doc.system_scenes:
                doc.remove_annotation(ss)
        else:
            logger.warning("Document already contains system scenes. Skipping.")
            return doc
    is_unsloth = False
    for t in (PeftModelForCausalLM, LlamaForCausalLM):
        if isinstance(model, t):
            is_unsloth = True
            break
    if is_unsloth:
        if context_size is None:
            context_size = 512
        return _annotate_document_with_unsloth(doc, model, context_size, tokenizer, cot_config=cot_config)
    print(type(model))
    assert tokenizer is None, "Tokenizer must be None for SSCModel"
    assert context_size is None, "Context size must be None for SSCModel"
    return _annotate_document_with_bert(doc, model)


def _annotate_document_with_unsloth(doc: UIMADocument,
                                    model: PeftModelForCausalLM,
                                    context_size,
                                    tokenizer,
                                    cot_config: CoTConfig = CoTConfigs.no_cot.value):
    test_llama_samples = {"llama_sentences": doc_to_llama_samples(
        doc, context_size=context_size, tokenizer=tokenizer, labels=False, cot_config=cot_config)["llama_sentences"]}
    test_dataset = Dataset.from_list(
        [dict(zip(test_llama_samples.keys(), values)) for values in zip(*test_llama_samples.values())])

    def tokenize_fn(sample):
        t = tokenizer(sample["llama_sentences"], return_tensors="pt", padding="max_length",
                      max_length=context_size, truncation=True).to(DEVICE)
        return t

    test_dataset = test_dataset.map(tokenize_fn, batched=False, remove_columns=["llama_sentences"])

    def collator(samples):
        return {"input_ids": torch.tensor([s["input_ids"][0] for s in samples]).to(DEVICE),
                "attention_mask": torch.tensor([s["attention_mask"][0] for s in samples]).to(DEVICE)}

    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=collator)

    prev_end = 0
    sentence_id = -1
    next_reason = "Begin of text"

    for sample in tqdm(test_dataloader, total=len(test_dataloader)):
        # logger.debug(f"Generating for sample {tokenizer.batch_decode(sample['input_ids'])}")
        outputs = model.generate(**sample, max_new_tokens=64, use_cache=True)
        output = tokenizer.batch_decode(outputs)

        predictions, reasons = parse_unsloth_output(output, sample, tokenizer, cot_config=cot_config)
        for i, prediction, reason in zip(range(len(predictions)), predictions, reasons):
            sentence_id += 1
            print(i, prediction, reason)
            if prediction:
                if sentence_id == 0:
                    continue
                end = doc.sentences[sentence_id].previous.end
                ss = doc.create_system_scene(prev_end, end, scene_type="Scene",
                                             add_to_document=True)
                ss.additional_features["reason"] = next_reason
                next_reason = reason
                prev_end = end + 1

    ss = doc.create_system_scene(prev_end, len(doc.text), scene_type="Scene", add_to_document=True)
    ss.additional_features["reason"] = next_reason

    return doc


def _annotate_document_with_bert(doc: UIMADocument, model: SSCModel):
    logger.info(f"Annotating document with model {model.config}")
    if doc.system_scenes:
        logger.warning("Document already contains system scenes. Overwriting.")
        for ss in doc.system_scenes:
            doc.remove_annotation(ss)

    tokenizer = AutoTokenizer.from_pretrained(model.config.tokenizer_name)
    stride = model.config.stride

    dataset = Dataset.from_generator(SSCDataset(files_or_docs=[doc], context_size=model.config.context_size,
                                                tokenizer=tokenizer, stride=stride,
                                                coarse=model.config.label_set == LabelSet.Coarse.value,
                                                labels=False).yield_samples)

    dataset = dataset.cast_column("labels", Sequence(model.config.label_set))
    dataset = dataset.map(lambda samples: tokenize_function(tokenizer, model.config.sep_token, samples), batched=False,
                          keep_in_memory=True)
    dataset = dataset.remove_columns(["sentences"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    dataset = dataset.map(collator)
    dataset.set_format(type='torch')
    dataloader = DataLoader(dataset)

    all_preds: Dict[int, Prediction] = {}

    if torch.cuda.is_available():
        model = model.to("cuda")

    logger.info(f"Starting prediction for document {doc.path}")

    for sample in tqdm(dataloader, total=len(dataloader)):
        sample = {k: v.to(model.device) for k, v in sample.items()}
        try:
            output = model(**sample)
            logits = output["logits"].squeeze()
            sentence_indices = output["sentence_indices"].flatten()
        except Exception as e:
            logger.error(f"Error in sample {tokenizer.batch_decode(sample['input_ids'])}: {e}")
            logger.warning(f"Using dummy logits for sample")
            sentence_indices = sample["sentence_indices"].flatten()
            logits = torch.zeros((len(sentence_indices), 2)).to(model.device)
            logits[:, 0] = 1
        preds = logits.argmax(-1).flatten()
        min_sentence_index = min(sentence_indices)
        max_sentence_index = max(sentence_indices)
        for sentence_index, pred in zip(sentence_indices, preds):
            min_distance_to_border = min(sentence_index - min_sentence_index, max_sentence_index - sentence_index)
            sentence_index = int(sentence_index)
            if stride == Stride.Full and sentence_index in all_preds:
                logger.warning(f"Found multiple predictions for sentence {sentence_index}, even though stride is full.")
            if sentence_index not in all_preds or min_distance_to_border > all_preds[sentence_index][1]:
                all_preds[sentence_index] = Prediction(int(pred), int(min_distance_to_border),
                                                       logits[
                                                           sentence_index - min_sentence_index].detach().cpu().numpy().tolist())

    prev_end = 0

    sentences = doc.sentences
    for sentence_index, prediction in all_preds.items():
        if prediction.predicted_label == 0:
            if sentence_index == 0:
                continue
            end = sentences[sentence_index].previous.end
            ss = doc.create_system_scene(prev_end, end, scene_type="Scene",
                                         add_to_document=True)
            ss.additional_features["distance_to_window_border"] = prediction.border_distance
            ss.additional_features["logits"] = prediction.logits
            prev_end = end + 1
        sentences[sentence_index].additional_features["logits"] = prediction.logits
        sentences[sentence_index].additional_features["predicted_label"] = prediction.predicted_label
        sentences[sentence_index].additional_features["distance_to_window_border"] = prediction.border_distance

    doc.create_system_scene(prev_end, len(doc.text), scene_type="Scene", add_to_document=True)

    return doc


def load_model(pretrained_model_name_or_path, unsloth):
    if unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, load_in_4bit=True)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    return model, tokenizer
