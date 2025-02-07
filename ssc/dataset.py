import itertools
import random
from dataclasses import dataclass
from typing import List, Union, Dict

from loguru import logger
from transformers import PreTrainedTokenizerBase, AutoTokenizer, PreTrainedTokenizer
from wuenlp.impl.UIMANLPStructs import UIMADocument, UIMASentence, UIMAScene, UIMASystemScene
from pathlib import Path

from llama.cot import CoTConfig
from ssc.model import Stride
from utils.constants import SCENE_TYPES, NONSCENE_TYPES, Label

try:
    from unsloth.chat_templates import get_chat_template
except ImportError:
    pass


@dataclass
class Sample:
    left_context: List[UIMASentence]
    right_context: List[UIMASentence]
    sentence: UIMASentence
    label: Label


def build_sample_around_sentence(doc: UIMADocument, sentence: UIMASentence, context_size: int,
                                 tokenizer: PreTrainedTokenizerBase, coarse: bool) -> Sample:
    context_left: List[UIMASentence] = []
    context_right: List[UIMASentence] = []
    context_len = 0
    prev_sentence: UIMASentence = sentence
    next_sentence: UIMASentence = sentence
    while True:
        next_sentence = next_sentence.next if next_sentence is not None else None
        prev_sentence = prev_sentence.previous if prev_sentence is not None else None

        if prev_sentence is not None:
            prev_sentence_length = len(tokenizer.encode(prev_sentence.text, add_special_tokens=False)) + 1
            if context_len + prev_sentence_length <= context_size:
                context_left.append(prev_sentence)
                context_len += prev_sentence_length
            else:
                break

        if next_sentence is not None:
            next_sentence_length = len(tokenizer.encode(next_sentence.text, add_special_tokens=False)) + 1
            if context_len + next_sentence_length <= context_size:
                context_right.append(next_sentence)
                context_len += next_sentence_length
            else:
                break

        if prev_sentence is None and next_sentence is None:
            break

    label = get_label(sentence, coarse=coarse)

    return Sample(left_context=list(reversed(context_left)), right_context=context_right, sentence=sentence,
                  label=label)


def get_scene(sentence: UIMASentence, get_predicted_scene: bool = False):
    SceneType = UIMASystemScene if get_predicted_scene else UIMAScene
    covering_scenes = sentence.covering(SceneType)
    try:
        current_scene: SceneType = covering_scenes[0]
    except:
        logger.warning(
            f"Sentence {sentence} is not covered by any scene! Checking for two scenes overlapping this sentence")
        overlapping_scenes = sentence.overlapping(SceneType)
        if len(overlapping_scenes) == 2:
            current_scene: SceneType = overlapping_scenes[1]
            logger.info(
                f"Found two scenes overlapping the sentence. Using the second one, since it starts with this sentence")
        elif len(overlapping_scenes) > 2:
            raise RuntimeError(f"More than two scenes overlapping this sentence found!")
        else:
            raise RuntimeError(f"No scenes covering this sentence found!")
    return current_scene


def get_label_reason(sentence, coarse: bool, get_predicted_label: bool = False):
    current_scene = get_scene(sentence, get_predicted_scene=get_predicted_label)
    reason_annotation = frozenset()
    try:
        for reason_annotation in ("Grund für Wechsel", "Grund_für_Wechsel"):
            if reason_annotation in current_scene.additional_features["features"]:
                reason_annotation = frozenset(current_scene.additional_features["features"][reason_annotation].split())
    except:
        reason_annotation = frozenset(["one of the important dimensions"])
    return reason_annotation


def get_label(sentence, coarse: bool, get_predicted_label: bool = False):
    try:
        current_scene = get_scene(sentence, get_predicted_scene=get_predicted_label)
        previous_scene = current_scene.previous
        label = Label.NOBORDER
        if current_scene.covered(UIMASentence)[0] == sentence:
            prev_scene_type = previous_scene.scene_type.lower() if previous_scene is not None else "scene"
            curr_scene_type = current_scene.scene_type.lower()
            assert prev_scene_type in SCENE_TYPES, f"Invalid scene type {prev_scene_type}"
            assert curr_scene_type in SCENE_TYPES, f"Invalid scene type {curr_scene_type}"
            if prev_scene_type in NONSCENE_TYPES and curr_scene_type in NONSCENE_TYPES:
                logger.warning(f"Detected border from Nonscene to Nonscene, which is illegal, at {sentence}")
            if coarse:
                label = Label.BORDER
            else:
                if prev_scene_type in NONSCENE_TYPES:
                    label = Label.NS2S
                elif curr_scene_type in NONSCENE_TYPES:
                    label = Label.S2NS
                else:
                    label = Label.S2S
    except Exception as e:
        logger.error(f"Error determining label for sentence {sentence}: {e}")
        label = Label.NOBORDER
    return label


def doc_to_window_samples(doc: UIMADocument, context_size: int, tokenizer: PreTrainedTokenizer, stride: Stride,
                          coarse: bool, drop_noninformative: bool, labels: bool = True) -> List:
    samples = []
    sentence = doc.sentences[0]
    if "xlnet" in tokenizer.name_or_path or "XLitNet" in tokenizer.name_or_path:
        curr_len = -1
    else:
        curr_len = 0
    sample_sentences = []
    sample_labels = []
    sample_sep_positions = []
    sample_sentence_indices = []

    while sentence is not None:
        sentence_len = len(tokenizer.encode(sentence.text, add_special_tokens=False)) + 1
        if curr_len + sentence_len >= context_size - 1:
            if not drop_noninformative or [label for label in sample_labels if label != Label.NOBORDER]:
                samples.append(
                    {"sentences": sample_sentences, "labels": sample_labels, "sep_token_indices": sample_sep_positions,
                     "sentence_indices": sample_sentence_indices})
            if stride == Stride.Half:
                try:
                    sentence = sample_sentences[len(sample_sentences) // 2 + 1]
                except IndexError:
                    pass
                sentence_len = len(tokenizer.encode(sentence.text, add_special_tokens=False)) + 1
            sample_sentences = []
            sample_sentence_indices = []
            sample_labels = []
            sample_sep_positions = []
            if "xlnet" in tokenizer.name_or_path or "XLitNet" in tokenizer.name_or_path:
                curr_len = -1
            else:
                curr_len = 0
        sample_sentences.append(sentence)
        sample_sentence_indices.append(sentence.position_within(doc.document_annotation))
        sample_labels.append(get_label(sentence, coarse=coarse) if labels else Label.NOBORDER)
        curr_len += sentence_len
        sample_sep_positions.append(curr_len)
        sentence = sentence.next

    # append last batch
    samples.append(
        {"sentences": sample_sentences, "labels": sample_labels, "sep_token_indices": sample_sep_positions,
         "sentence_indices": sample_sentence_indices})

    if not drop_noninformative:
        if stride == Stride.Full:
            for i in range(1, len(samples)):
                assert (samples[i - 1]["sentences"][-1].end - samples[i]["sentences"][
                    0].begin - 1) <= 2, f"Sample {i} does not start right after sample {i - 1}"
        else:
            for i in range(1, len(samples)):
                prev_sentences = samples[i - 1]["sentences"]
                assert (prev_sentences[len(prev_sentences) // 2].end - samples[i]["sentences"][
                    0].begin - 1) <= 2, f"Sample {i} does not start in the middle of sample {i - 1}"

    for sample in samples:
        sample["sentences"] = [sentence.text for sentence in sample["sentences"]]

    return samples


def doc_to_llama_samples(doc: UIMADocument, context_size: int, tokenizer: PreTrainedTokenizer,
                         labels: bool, cot_config: CoTConfig) -> Dict:
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        map_eos_token=True,  # Maps <|im_end|> to </s> instead
    )

    EOS_TOKEN = tokenizer.eos_token
    if EOS_TOKEN is None:
        logger.warning(f"Tokenizer {tokenizer.name_or_path} does not have an EOS token. Using a period instead.")
        EOS_TOKEN = "."

    def build_sentence_sample(sentence: UIMASentence, prompt: str, exp_output: str, context_size: int) -> str:
        context = ["<sentence>" + sentence.text + "</sentence>"]
        prev_sentence = sentence.previous
        next_sentence = sentence.next

        temp_c = (" ".join(context + [prev_sentence.text if prev_sentence is not None else "",
                                      next_sentence.text if next_sentence is not None else ""]))
        while len(get_llama_input(exp_output, temp_c, True, True, prompt)) < context_size:
            if prev_sentence is not None:
                context.insert(0, prev_sentence.text)
                prev_sentence = prev_sentence.previous
            if next_sentence is not None:
                context.append(next_sentence.text)
                next_sentence = next_sentence.next

            temp_c = (" ".join(context + [prev_sentence.text if prev_sentence is not None else "",
                                          next_sentence.text if next_sentence is not None else ""]))
            if prev_sentence is None and next_sentence is None:
                break
        context = " ".join(context)
        text = get_llama_input(exp_output, context, exp_output == "", tokenize=False, prompt=prompt)
        return text

    def get_llama_input(exp_output, history, add_generation_prompt, tokenize, prompt):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": history},
            ] + ([
                     {"role": "assistant", "content": exp_output}
                 ] if exp_output else []), tokenize=tokenize, add_generation_prompt=add_generation_prompt)

    samples = []
    sentence = doc.sentences[0]
    sample_sentences = []
    sample_labels = []
    sample_sentence_indices = []
    sample_llama_sentences = []

    while sentence is not None:

        if labels:
            map_label_to_bool = {Label.BORDER: "True", Label.NOBORDER: "False"}
            map_reason_to_en = {"Handlung": "narrative action", "Raum": "location", "Zeit": "time",
                                "Figuren": "characters"}
            scene_label = map_label_to_bool[get_label(sentence, coarse=True)]
            label_reason = get_label_reason(sentence, coarse=True)
            first_sentence = sentence.previous is None
            sentence_output = get_llama_target(label_reason, map_reason_to_en, scene_label, cot_config,
                                               first_sentence=first_sentence)
            llama_sentence_input = build_sentence_sample(sentence, cot_config.prompt, sentence_output, context_size)
        else:
            llama_sentence_input = build_sentence_sample(sentence, cot_config.prompt, "", context_size)

        sample_llama_sentences.append(llama_sentence_input)
        sample_sentences.append(sentence)
        sample_sentence_indices.append(sentence.position_within(doc.document_annotation))
        sample_labels.append(get_label(sentence, coarse=True))
        sentence = sentence.next

    # append last batch
    samples = {"sentences": sample_sentences, "labels": sample_labels, "sentence_indices": sample_sentence_indices,
               "llama_sentences": sample_llama_sentences}

    samples["sentences"] = [sentence.text for sentence in samples["sentences"]]

    return samples


def get_llama_target(label_reason: frozenset, map_reason_to_en: dict, scene_label: str, cot_config: CoTConfig,
                     first_sentence: bool):
    return cot_config.get_target(label=scene_label, label_reason=label_reason, map_reason_to_en=map_reason_to_en,
                                 first_sentence=first_sentence)


def xmi_to_window_samples(xmi_path: Path, context_size: int, tokenizer: PreTrainedTokenizer, stride: Stride,
                          coarse: bool, drop_noninformative: bool, labels: bool = True) -> List:
    doc = UIMADocument.from_xmi(xmi_path)

    return doc_to_window_samples(doc, context_size=context_size, tokenizer=tokenizer, stride=stride, coarse=coarse,
                                 drop_noninformative=drop_noninformative, labels=labels)


def xmi_to_llama_samples(xmi_path: Path, context_size: int, tokenizer: PreTrainedTokenizer, train: bool,
                         cot_config: CoTConfig) -> Dict:
    doc = UIMADocument.from_xmi(xmi_path)

    return doc_to_llama_samples(doc, context_size=context_size, tokenizer=tokenizer, labels=train,
                                cot_config=cot_config)


class SSCDataset:
    num_borders: int
    num_nonborders: int

    def __init__(self, files_or_docs: Union[List[Path], List[UIMADocument]], context_size: int, tokenizer,
                 stride: Stride, coarse: bool, drop_noninformative: bool = False, labels: bool = True):
        self.samples = []

        for file in files_or_docs:
            try:
                if isinstance(file, Path):
                    self.samples.extend(xmi_to_window_samples(file, context_size, tokenizer, stride, coarse=coarse,
                                                              drop_noninformative=drop_noninformative, labels=labels))
                else:
                    self.samples.extend(doc_to_window_samples(file, context_size, tokenizer, stride, coarse=coarse,
                                                              drop_noninformative=drop_noninformative, labels=labels))
            except RuntimeError as e:
                logger.error(f"Error extracting samples from {file}: {e}")

        self.num_borders = sum(
            [len([label for label in sample["labels"] if label != Label.NOBORDER]) for sample in self.samples])
        self.num_nonborders = sum(
            [len([label for label in sample["labels"] if label == Label.NOBORDER]) for sample in self.samples])

    def get_samples(self):
        return self.samples

    def yield_samples(self):
        for sample in self.get_samples():
            yield sample
