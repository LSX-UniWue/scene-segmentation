import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Pattern
from loguru import logger


def get_non_cot_target(label: str, label_reason: frozenset, map_reason_to_en: dict, **kwargs):
    if label == "True":
        target = label + ", because there is a greater change in " + " and ".join(
            [map_reason_to_en.get(dim.strip(","), dim.strip(",")) for dim in label_reason]) + "."
    elif label == "False":
        target = label + ", because there is no significant change in narrative action, location, time or characters."
    else:
        raise RuntimeError(f"Invalid scene label {label}")

    return target


def get_short_list_cot_target(label: str, label_reason: frozenset, first_sentence: bool, **kwargs):
    if label == "False":
        sentence_output = "a) No, " \
                          "b) no, " \
                          "c) no, " \
                          "d) no, " \
                          "e) therefore the sentence does not start a new scene."
    else:
        if first_sentence:
            sentence_output = "a) Yes, " \
                              "b) yes, " \
                              "c) yes, " \
                              "d) yes, " \
                              "e) therefore the sentence starts a new scene."
        elif not label_reason:
            logger.warning(f"No reason annotation found for scene label True. Using default sentence output.")
            sentence_output = "a) Maybe, " \
                              "b) maybe, " \
                              "c) maybe, " \
                              "d) maybe, " \
                              "e) therefore the sentence starts a new scene."
        else:
            sentence_output = f"a) {'Yes' if 'Handlung' in label_reason else 'no'}, " \
                              f"b) {'yes' if 'Raum' in label_reason else 'no'}, " \
                              f"c) {'yes' if 'Zeit' in label_reason else 'no'}, " \
                              f"d) {'yes' if 'Figuren' in label_reason else 'no'}, " \
                              f"e) therefore the sentence starts a new scene."

    return sentence_output


def get_exhaustive_cot_target(label: str, label_reason: frozenset, first_sentence: bool = False, **kwargs):
    if label == "False":
        sentence_output = "a) There is no significant change in narrative action, " \
                          "b) there is no significant change in location, " \
                          "c) there is no significant change in time, " \
                          "d) there is no significant change in characters, " \
                          "e) therefore the sentence does not start a new scene."
    else:
        if first_sentence:
            sentence_output = "a) There is a significant change in narrative action, " \
                              "b) there is a significant change in location, " \
                              "c) there is a significant change in time, " \
                              "d) there is a significant change in characters, " \
                              "e) therefore the sentence starts a new scene."
        elif not label_reason:
            logger.warning(f"No reason annotation found for scene label True. Using default sentence output.")
            sentence_output = "a) There may be a significant change in narrative action, " \
                              "b) there may be a significant change in location, " \
                              "c) there may be a significant change in time, " \
                              "d) there may be a significant change in characters, " \
                              "e) therefore the sentence starts a new scene."
        else:
            sentence_output = f"a) {'There is a' if 'Handlung' in label_reason else 'there is no'} significant change in narrative action, " \
                              f"b) {'there is a' if 'Raum' in label_reason else 'there is no'} significant change in location, " \
                              f"c) {'there is a' if 'Zeit' in label_reason else 'there is no'} significant change in time, " \
                              f"d) {'there is a' if 'Figuren' in label_reason else 'there is no'} significant change in characters, " \
                              f"e) therefore the sentence starts a new scene."
    return sentence_output


def get_short_cot_target(label: str, label_reason: frozenset, map_reason_to_en: dict, first_sentence: bool, **kwargs):
    if first_sentence:
        sentence_output = "This is the first sentence of the document. Therefore the sentence starts a new scene."
    elif label == "True":
        sentence_output = "There is a significant change in " + " and ".join(
            [map_reason_to_en.get(dim.strip(","), dim.strip(",")) for dim in
             label_reason]) + ". Therefore the sentence starts a new scene."
    elif label == "False":
        sentence_output = "There is no significant change in narrative action, location, time or characters. Therefore the sentence does not start a new scene."
    else:
        raise RuntimeError(f"Invalid scene label {label}")

    return sentence_output


@dataclass
class CoTConfig:
    prompt: str
    get_target: Callable
    regex: Pattern


class CoTConfigs(Enum):
    no_cot = CoTConfig(
        prompt=("Does the sentence in <sentence>...</sentence> introduce the beginning of a new scene and a "
                "significant break in time, location or characters? Answer 'True' or 'False' and provide a reason "
                "for your decision. A scene is defined as a segment of text with a coherent structure across the "
                "dimensions 'characters' (which characters are present in the narration), 'location' (where does the"
                " narration take place), and 'time' (continuous time in the narration). A significant break in any"
                " of these dimensions corresponds to a scenes change. "),
        get_target=get_non_cot_target,
        regex=None,
    )

    exhaustive_list = CoTConfig(
        prompt=("A scene is defined as a segment of text with a coherent structure across the "
                "dimensions 'characters' (which characters are present in the narration), 'location' (where does the"
                " narration take place), and 'time' (continuous time in the narration). A significant break in any"
                " of these dimensions corresponds to a scenes change. Does the sentence in <sentence>...</sentence>"
                " introduce the beginning of a new scene? Think step by step: a) Does the sentence introduce a "
                "significant change in narrative action? b) Does the sentence introduce a significant change in "
                "location? c) Does the sentence introduce a significant change in time? d) Does the sentence "
                "introduce a significant change in characters? e) Does the sentence therefore start a new scene?"),
        get_target=get_exhaustive_cot_target,
        regex=re.compile(
            r"a\) (?P<action>.*)"
            r"b\) (?P<location>.*)"
            r"c\) (?P<time>.*)"
            r"d\) (?P<characters>.*)"
            r"e\) (?P<border>.*)"
        )
    )

    short_list = CoTConfig(
        prompt=("A scene is defined as a segment of text with a coherent structure across the "
                "dimensions 'characters' (which characters are present in the narration), 'location' (where does the"
                " narration take place), and 'time' (continuous time in the narration). A significant break in any"
                " of these dimensions corresponds to a scenes change. Does the sentence in <sentence>...</sentence>"
                " introduce the beginning of a new scene? Think step by step: a) Does the sentence introduce a "
                "significant change in narrative action? b) Does the sentence introduce a significant change in "
                "location? c) Does the sentence introduce a significant change in time? d) Does the sentence "
                "introduce a significant change in characters? e) Does the sentence therefore start a new scene?"),
        get_target=get_short_list_cot_target,
        regex=re.compile(
            r"a\) (?P<action>.*)"
            r"b\) (?P<location>.*)"
            r"c\) (?P<time>.*)"
            r"d\) (?P<characters>.*)"
            r"e\) (?P<border>.*)"
        )
    )

    short = CoTConfig(
        prompt=("A scene is defined as a segment of text with a coherent structure across the "
                "dimensions 'characters' (which characters are present in the narration), 'location' (where does the"
                " narration take place), and 'time' (continuous time in the narration). A significant break in any"
                " of these dimensions corresponds to a scenes change. Does the sentence in <sentence>...</sentence>"
                " introduce the beginning of a new scene? Think step by step: Which, if any, of the important dimensions"
                " change? Does the sentence therefore start a new scene?"),
        get_target=get_short_cot_target,
        regex=re.compile(r"(?P<reason>.*). (?P<border>Therefore the sentence .*)")
    )
