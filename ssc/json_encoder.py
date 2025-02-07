import dataclasses
import json
from datetime import datetime

from datasets import ClassLabel


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            result = dataclasses.asdict(obj)
            result['_dataclass'] = obj.__class__.__name__
            return result
        # if isinstance(obj, PreTrainedModel) or isinstance(obj, PreTrainedTokenizerBase):
        #    return {"_type": obj.__class__.__name__, "name_or_path": obj.name_or_path}
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def custom_decoder(dct):
    if '_dataclass' in dct:
        cls_name = dct.pop('_dataclass')
        if cls_name == 'ClassLabel':
            dct.pop("_type")
            return ClassLabel(**dct)
    return dct


_original_dumps = json.dumps
_original_dump = json.dump
_original_loads = json.loads
_original_load = json.load


def _patched_dumps(*args, **kwargs):
    kwargs['cls'] = CustomJSONEncoder
    return _original_dumps(*args, **kwargs)


def _patched_dump(*args, **kwargs):
    kwargs['cls'] = CustomJSONEncoder
    return _original_dump(*args, **kwargs)


def _patched_loads(*args, **kwargs):
    kwargs['object_hook'] = custom_decoder
    return _original_loads(*args, **kwargs)


def _patched_load(*args, **kwargs):
    kwargs['object_hook'] = custom_decoder
    return _original_load(*args, **kwargs)


json.dumps = _patched_dumps
json.dump = _patched_dump
json.loads = _patched_loads
json.load = _patched_load
