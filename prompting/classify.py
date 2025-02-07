import json
import os
from dataclasses import dataclass

from pathlib import Path

from typing import Type, Literal, Optional

from langchain.adapters import openai
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, BaseMessage, ChatMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from tqdm import tqdm
from wuenlp.impl.UIMANLPStructs import UIMADocument, UIMAScene, UIMASentence

from loguru import logger
import tiktoken

from utils.constants import datasets_folder

INPUT_PERCENTAGE = 0.8

OLLAMA_API_BASE = "OLLAMA API Base URL"

api_key = "Your OpenAI API Key"
openrouter_api_key = "Your OpenRouter API Key"

os.environ["OPENAI_API_KEY"] = api_key

prompt_classify = ("Does the sentence in <sentence>...</sentence> introduce the beginning of a new scene and a "
                   "significant break in time, location or characters? Answer 'True' or 'False' and provide a reason "
                   "for your decision. A scene is defined as a segment of text with a coherent structure across the "
                   "dimensions 'characters' (which characters are present in the narration), 'location' (where does the"
                   " narration take place), and 'time' (continuous time in the narration). A significant break in any"
                   " of these dimensions corresponds to a scenes change. ")

seed = 1337


@dataclass
class Model:
    name: str
    context_size: int
    json: bool
    openai: bool = True
    openrouter: bool = False


GPT35 = Model("gpt-3.5-turbo", int(4096 * INPUT_PERCENTAGE), json=False, openai=True)
GPT35JSON = Model("gpt-3.5-turbo-1106", int(4096 * INPUT_PERCENTAGE), json=True, openai=True)
GPT35JSON512 = Model("gpt-3.5-turbo-1106", int(512), json=True, openai=True)
GPT35512 = Model("gpt-3.5-turbo-1106", int(512), json=False, openai=True)
GPT4 = Model("gpt-4-1106-preview", int(128000 * INPUT_PERCENTAGE), json=True, openai=True)
GPT4o = Model("gpt-4o", int(512 * INPUT_PERCENTAGE), json=False, openai=True)
GPTo1mini = Model("o1-mini", int(512 * INPUT_PERCENTAGE), json=False, openai=True)
GPTo1 = Model("o1", int(512 * INPUT_PERCENTAGE), json=False, openai=True)
GPT4omini = Model("gpt-4o-mini", int(512 * INPUT_PERCENTAGE), json=False, openai=True)
llama2 = Model("llama2", int(4096 * INPUT_PERCENTAGE), json=True, openai=False)
llama3 = Model("llama3", int(4096 * INPUT_PERCENTAGE), json=True, openai=False)
llama38b = Model("llama3:8b", int(512 * INPUT_PERCENTAGE), json=True, openai=False)
llama370b = Model("llama3:70b", int(512 * INPUT_PERCENTAGE), json=True, openai=False)
llama3405b = Model("llama3.1:405b", int(512 * INPUT_PERCENTAGE), json=True, openai=False)
commandrplus = Model("command-r-plus:latest", int(512 * INPUT_PERCENTAGE), json=True, openai=False)
llama3instruct = Model("llama3:instruct", int(512 * INPUT_PERCENTAGE), json=True, openai=False)
mistral = Model("mistral", int(4096 * INPUT_PERCENTAGE), json=True, openai=False)
r1 = Model("deepseek/deepseek-r1", int(512 * INPUT_PERCENTAGE), json=True, openai=False, openrouter=True)


def build_sentence_sample(sentence: UIMASentence, model: Model) -> str:
    s = sentence.text

    encoding = tiktoken.get_encoding("cl100k_base")

    context = ["<sentence>" + sentence.text + "</sentence>"]
    prev_sentence = sentence.previous
    next_sentence = sentence.next
    while len(encoding.encode(" ".join(context + [prev_sentence.text if prev_sentence is not None else "",
                                                  next_sentence.text if next_sentence is not None else ""]))) < model.context_size:
        if prev_sentence is not None:
            context.insert(0, prev_sentence.text)
            prev_sentence = prev_sentence.previous
        if next_sentence is not None:
            context.append(next_sentence.text)
            next_sentence = next_sentence.next

    context = " ".join(context)
    return context


def classify_sentences(doc: UIMADocument, model: Model, cache_dir: Path, out_dir: Path, ):
    logger.info("Classifying sentences in document {} with {} characters", doc.path, len(doc.text))
    text = doc.text

    cache_file = (cache_dir / str(model) / doc.path.name).with_suffix(".json")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if (cache_file).is_file():
        logger.info(f"Loading cached classification from {cache_file}")
        with open(cache_file, "r") as f:
            predicted_labels = json.load(f)
    else:
        predicted_labels = {"labels": [], "reasons": []}
    if len(predicted_labels["labels"]) != len(doc.sentences):
        logger.info("Cached classification does not match document, resuming classification")

        prompt_text = prompt_classify
        print(prompt_text)
        chain = get_chain(model, prompt_text)

        for i, sentence in enumerate(tqdm(doc.sentences, total=len(doc.sentences))):
            if i < len(predicted_labels["labels"]):
                continue
            sample = build_sentence_sample(sentence, model)

            scene_change = None
            retry_count = 0

            while scene_change is None and retry_count < 10:
                while (response := get_response_safe(chain, sample)) is None:
                    pass
                logger.info(sentence)
                logger.info(response["text"])
                chain.memory.chat_memory.messages = chain.memory.chat_memory.messages[:-2]
                lines = response["text"].split("\n")
                if "True" in lines[0]:
                    scene_change = True
                    reason = response["text"][4:]
                elif "False" in lines[0]:
                    scene_change = False
                    reason = response["text"][5:]
                else:
                    retry_count += 1
                    logger.warning(f"Invalid response {response['text']}, retrying")

            if scene_change is None:
                raise RuntimeError("Failed to classify sentence")

            predicted_labels["labels"].append(scene_change)
            predicted_labels["reasons"].append(reason)

            with open(cache_file, "w") as f:
                json.dump(predicted_labels, f)

    prev_end = 0
    prev_reason = None

    for sentence, label, reason in list(zip(doc.sentences, predicted_labels["labels"], predicted_labels["reasons"]))[
                                   1:]:
        if label:
            ss = doc.create_system_scene(prev_end, sentence.previous.end, scene_type="Scene", add_to_document=True)
            ss.additional_features["reason"] = prev_reason
            prev_reason = reason
            prev_end = sentence.previous.end

    ss = doc.create_system_scene(prev_end, len(text), scene_type="Scene", add_to_document=True)
    ss.additional_features["reason"] = prev_reason

    model_out_dir = out_dir / str(model)
    model_out_dir.mkdir(exist_ok=True, parents=True)
    doc.serialize(model_out_dir / doc.path.name)


def get_response_safe(chain, sample):
    try:
        response = chain(sample)
    except TypeError:
        response = None
    return response


class DeveloperMessage(ChatMessage):
    type: Literal["developer"] = "developer"
    role = "developer"


class DeveloperMessagePromptTemplate(SystemMessagePromptTemplate):
    """System message prompt template.
    This is a message that is not sent to the user.
    """

    _msg_class: Type[BaseMessage] = DeveloperMessage


class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(self,
                 model_name: str,
                 openai_api_key: Optional[str] = None,
                 openai_api_base: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        openai_api_key = openai_api_key or openrouter_api_key
        super().__init__(openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key,
                         model_name=model_name, **kwargs)


def get_chain(model, prompt_text) -> Chain:
    if model.openai:
        chat_model = ChatOpenAI(model=model.name)
        if model.json:
            chat_model = chat_model.bind(
                response_format={"type": "json_object"})
    elif model.openrouter:
        chat_model = ChatOpenRouter(model_name=model.name)
    else:
        chat_model = ChatOllama(model=model.name, base_url=OLLAMA_API_BASE)
    memory = ConversationBufferMemory(return_messages=True, llm=chat_model, max_token_limit=model.context_size)
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(prompt_text) if model.name not in (
                "o1", "o1-mini") else HumanMessagePromptTemplate.from_template(prompt_text),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    chain: LLMChain = LLMChain(llm=chat_model, memory=memory, prompt=prompt)

    if model.name in ("o1", "o1-mini"):
        chat_model.temperature = 1

    return chain


def get_chunks(doc, model):
    text = doc.text

    encoding = tiktoken.get_encoding("cl100k_base")
    sentences = doc.sentences
    encoded_sentences = encoding.encode_batch(sentences.text)
    chunk_ends = [0]
    current_chunk = []

    for c in doc.chunks:
        doc.remove_annotation(c)

    for sentence, encoded_sentence in zip(sentences, encoded_sentences):
        if len(current_chunk) + len(encoded_sentence) > model.context_size:
            chunk_ends.append(sentence.end)
            current_chunk = []
            doc.create_chunk(chunk_ends[-2], chunk_ends[-1], chunk_type="GPTChunk", add_to_document=True)
        current_chunk.extend(encoded_sentence)
    chunk_ends.append(len(text))
    doc.create_chunk(chunk_ends[-2], len(text), chunk_type="GPTChunk", add_to_document=True)
    return doc.chunks


def notify(x):
    logger.info(x)


def classify():
    in_dir = datasets_folder / "test_full"
    classify_cache_dir = Path("data/cache_classify")
    classify_cache_dir.mkdir(exist_ok=True, parents=True)
    classify_out_dir = Path("data/output_classify/")
    classify_out_dir.mkdir(exist_ok=True, parents=True)

    # reference_doc = UIMADocument.from_xmi(in_dir / "Hänsel und Gretel.xmi.zip")
    # for file in in_dir.iterdir():
    for file in [in_dir / "Harry Potter und der Halbblutprinz - Kapitel Der Slug Club.xmi.zip"]:
        notify(f"Classifying {file.name}")
        if file.suffix in (".xmi", ".zip"):
            print(file)
            doc = UIMADocument.from_xmi(file)
            classify_sentences(doc, model=GPTo1, cache_dir=classify_cache_dir, out_dir=classify_out_dir)
            notify(f"Finished classifying {file.name}")


if __name__ == '__main__':
    classify()
