from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import torch
from datasets import ClassLabel
from loguru import logger
from torch import nn
from torch.nn import LSTM
from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoTokenizer, AutoConfig, \
    BitsAndBytesConfig, QuantoConfig, XLNetModel

from abc import ABC

from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import ModelOutput

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

import ssc.json_encoder
from utils.constants import Label

import sys

debugger_is_active = hasattr(sys, 'gettrace') and sys.gettrace() is not None
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.device("mps") else "cpu"


class EmbeddingModelType(str, Enum):
    BertStyleModel = "bert-style-model"
    XLNet = "xlnet"
    XXLNet = "xxlnet"
    LLM = "llm"


class LabelSet(Enum):
    Coarse = ClassLabel(2, names=[Label.BORDER, Label.NOBORDER])
    Fine = ClassLabel(4, names=[Label.S2S, Label.S2NS, Label.NS2S, Label.NOBORDER])


class Stride(str, Enum):
    Half = "half"
    Full = "full"


class TaskHead(nn.Module, ABC):
    weight: float

    def __init__(self, weight: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

    def forward(self, x: torch.Tensor, sep_indices: torch.Tensor, labels=None):
        raise NotImplementedError()


class BiLSTMEncoder(nn.Module):
    num_layers: int
    layer_size: int

    def __init__(self, input_size: int, num_layers: int, layer_size: int):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = LSTM(input_size=input_size, hidden_size=layer_size, num_layers=num_layers, bidirectional=True,
                         batch_first=True)

    def forward(self, inputs: torch.Tensor):
        return self.lstm(inputs)


def get_model_type(embedding_model_name: str):
    _config = AutoConfig.from_pretrained(embedding_model_name)
    if _config.model_type in ("bert", "modernbert"):
        embedding_model_type = EmbeddingModelType.BertStyleModel
    elif _config.model_type == "xlnet":
        embedding_model_type = EmbeddingModelType.XLNet
    elif _config.model_type == "llama":
        embedding_model_type = EmbeddingModelType.LLM
    else:
        raise ValueError(f"Unsupported model type {_config.model_type}")

    return embedding_model_type


class MLPSceneHead(nn.Module):
    """Scene Border Classification head"""

    def __init__(self, dropout_rate: float, input_size: int, num_classes: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_1_size = 2048
        self.hidden_2_size = 256
        self.num_classes = num_classes

        self.fc1 = nn.Linear(self.input_size, self.hidden_1_size)
        self.gelu1 = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_1_size, self.hidden_2_size)
        self.gelu2 = nn.GELU()
        self.fc3 = nn.Linear(self.hidden_2_size, self.num_classes)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.dropout3 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.dropout1(x)
        a = self.gelu1(self.fc1(x))
        a = self.dropout2(a)
        b = self.gelu2(self.fc2(a))
        b = self.dropout3(b)
        c = self.fc3(b)
        return c


class LinearSceneHead(nn.Module):
    """Scene Border Classification head"""

    def __init__(self, dropout_rate: float, input_size: int, num_classes: int):
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        self.fc1 = nn.Linear(self.input_size, self.num_classes)

        nn.init.kaiming_normal_(self.fc1.weight)

        self.dropout1 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.dropout1(x)
        a = self.fc1(x)
        return a


class SSCModelConfig(PretrainedConfig):
    embedding_model_name: str
    embedding_model_type: EmbeddingModelType
    has_mems: bool
    tokenizer_name: str
    label_set: ClassLabel
    stride: Stride
    additional_heads: List[TaskHead]
    only_last_layer: bool
    dropout: float
    context_size: int
    train_embedding_model: bool
    sep_token: str
    lstm_num_layers: int
    lstm_layer_size: int
    class_weights: List[float] = None
    only_sep_mems: bool = False

    def __init__(self, embedding_model_name: str = None, label_set: ClassLabel = LabelSet.Coarse,
                 stride: Stride = Stride.Full,
                 additional_heads: List[TaskHead] = (), only_last_layer: bool = False, dropout: float = 0,
                 context_size: int = 512, train_embedding_model: bool = True, lstm_num_layers: int = 0,
                 lstm_layer_size: int = 0, tokenizer_name: str = None, has_mems: bool = False,
                 class_weights: List[float] = None, only_sep_mems: bool = False, no_mlp: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_model_name = embedding_model_name
        self.label_set = label_set
        self.stride = stride
        self.additional_heads = additional_heads
        self.only_last_layer = only_last_layer
        self.dropout = dropout
        self.context_size = context_size
        self.train_embedding_model = train_embedding_model
        self.has_mems = has_mems
        self.no_mlp = no_mlp
        if tokenizer_name is None:
            tokenizer_name = embedding_model_name
        self.tokenizer_name = tokenizer_name
        if embedding_model_name is None:
            self.sep_token = None
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.sep_token = tokenizer.sep_token
            if self.sep_token is None:
                sep_token = "<|reserved_special_token_0|>"
                logger.info(f"Setting the sep token to {sep_token}")
                assert len(
                    tokenizer(sep_token)) == 2, f"sep token {sep_token} tokenizes to more than one token!"
                self.sep_token = sep_token
        self.lstm_num_layers = lstm_num_layers
        self.lstm_layer_size = lstm_layer_size
        if embedding_model_name is not None:
            self.embedding_model_type = get_model_type(embedding_model_name)
        else:
            self.embedding_model_type = None
        self.class_weights = class_weights
        self.only_sep_mems = only_sep_mems

    @property
    def class_weights(self):
        return self._class_weights

    @class_weights.setter
    def class_weights(self, value):
        self._class_weights = list(value) if value is not None else None


@dataclass
class SSCModelOutput(TokenClassifierOutput):
    mems: Optional[torch.FloatTensor] = None


class SSCModel(PreTrainedModel):
    config_class = SSCModelConfig
    config: SSCModelConfig

    def __init__(self, config: SSCModelConfig):
        super().__init__(config)

        if torch.cuda.is_available():
            model_type = get_model_type(config.embedding_model_name)
            if model_type == EmbeddingModelType.LLM:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_quant_type="nf8",
                    bnb_8bit_compute_dtype=torch.bfloat16,
                )
            else:
                quantization_config = None
            self.embedding_model: PreTrainedModel = AutoModel.from_pretrained(config.embedding_model_name,
                                                                              device_map="auto" if model_type == EmbeddingModelType.LLM else "cuda",
                                                                              quantization_config=quantization_config)
        else:
            # this is only for development on the laptop
            quantization_config = QuantoConfig(weights="int4")
            self.embedding_model: PreTrainedModel = AutoModel.from_pretrained(config.embedding_model_name,
                                                                              quantization_config=None)

        # peft for embedding model
        if config.embedding_model_type == EmbeddingModelType.LLM:
            peft_config = LoraConfig(
                r=32,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.embedding_model = prepare_model_for_kbit_training(self.embedding_model)
            self.embedding_model = get_peft_model(self.embedding_model, peft_config)
            self.embedding_model.print_trainable_parameters()

        if config.train_embedding_model is False:
            logger.info("Freezing embedding model")
            for param in self.embedding_model.parameters():
                param.requires_grad = False
        self.config = config

        if config.lstm_num_layers > 0:
            self.lstm = BiLSTMEncoder(input_size=self.embedding_model.config.hidden_size,
                                      num_layers=config.lstm_num_layers, layer_size=config.lstm_layer_size)
            if self.embedding_model.dtype == torch.float16:
                self.lstm.half()
            self.feed_forward = MLPSceneHead(self.config.dropout, config.lstm_layer_size * 2,
                                             self.config.label_set.num_classes)
        else:
            if config.no_mlp:
                self.feed_forward = LinearSceneHead(self.config.dropout, self.embedding_model.config.hidden_size,
                                                    self.config.label_set.num_classes)
            else:
                self.feed_forward = MLPSceneHead(self.config.dropout, self.embedding_model.config.hidden_size,
                                                 self.config.label_set.num_classes)
        if self.embedding_model.dtype == torch.float16:
            self.feed_forward.half()

        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(config.class_weights).to(DEVICE) if config.class_weights is not None else None)
        if self.embedding_model.dtype == torch.float16:
            self.criterion.half()

        if debugger_is_active:
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def forward(self, input_ids: torch.Tensor, labels=None, additional_heads_labels: List = None,
                mems: torch.Tensor = None, sep_token_indices: list = None, attention_mask: torch.Tensor = None,
                sentence_indices: torch.Tensor = None, **kwargs):
        # logger.info(f"Received unexpected arguments {kwargs}")
        # print(input_ids)
        if debugger_is_active:
            readable_samples = [(b.item(), a) for a, b in
                                zip(self.tokenizer.decode(input_ids[0]).split("[SEP]"), labels[0])]
        with torch.autograd.set_detect_anomaly(False):
            if self.config.has_mems:
                transformer_output = self.embedding_model(input_ids, output_hidden_states=True,
                                                          attention_mask=attention_mask, mems=mems, use_mems=True)
            else:
                transformer_output = self.embedding_model(input_ids, output_hidden_states=True,
                                                          attention_mask=attention_mask)
            hidden_states = transformer_output.hidden_states

            # print(hidden_states)

            assert len(sep_token_indices) == len(
                labels), f"Number of SEP indices ({len(sep_token_indices)}) does not match number of labels ({len(labels)})!"

            if self.config.has_mems:
                mems = self.update_mems(transformer_output)

            if self.config.only_last_layer:
                embeddings = hidden_states[-1].squeeze()
            else:
                layers = [-4, -3, -2, -1]
                embeddings = torch.stack([hidden_states[i] for i in layers]).sum(0).squeeze()

            if self.config.model_type == EmbeddingModelType.LLM:
                sep_embeddings = []
                prev_index = 0
                for sep_token_index in sep_token_indices[1:]:
                    sep_embeddings.append(embeddings[prev_index:sep_token_index, :].mean(0))
                    prev_index = sep_token_index
                sep_embeddings.append(embeddings[prev_index:, :].mean(0))
                sep_embeddings = torch.stack(sep_embeddings)
            else:
                sep_embeddings = embeddings[sep_token_indices, :]

            # print(sep_embeddings)

            ## main head

            if self.config.lstm_num_layers > 0:
                lstm_output, _ = self.lstm(sep_embeddings)
                logits = self.feed_forward(lstm_output)
            else:
                logits = self.feed_forward(sep_embeddings)

            ## additional tasks/heads
            head_outputs = []

            additional_heads = self.config.additional_heads
            for i, additional_head in enumerate(additional_heads):
                head_output = additional_head(transformer_output, sep_token_indices, labels, additional_heads_labels[i])

                head_outputs.append(head_output)

        ret = {
            "logits": logits,
            "additional_head_logits": [ho["logits"] for ho in head_outputs],
            "sentence_indices": sentence_indices
        }

        if self.config.has_mems:
            ret["mems"] = mems

        if labels is not None:
            logits_squeeze = logits.squeeze()
            labels_squeeze = labels.squeeze()
            # print(logits_squeeze)
            # print(labels_squeeze)
            loss = self.criterion(logits_squeeze, labels_squeeze)
            # print(loss)

            loss += sum([head_outputs[i]["loss"] * additional_heads[i].weight for i in
                         range(len(additional_heads))])

            ret["loss"] = loss
            ret["labels"] = labels
        output = ModelOutput(**ret)
        if self.config.has_mems:
            assert output["mems"] == output[3]
        return output

    def update_mems(self, transformer_output):
        if self.config.only_sep_mems:
            # TODO: May need to adapt positional encoding (or just don't care)
            raise NotImplementedError()
        else:
            mems = [mem[-512:, :, :] for mem in transformer_output.mems]

        return mems
