from enum import Enum
from pathlib import Path

SCENE_TYPES = {
    "scene", "nonscene", "szene", "nichtszene", "nicht-szene", "szene ebene 1", "szene ebene 2",
    "szene ebene 3"}
NONSCENE_TYPES = {"nonscene", "nichtszene", "nicht-szene"}


class StringEnum(str, Enum):
    pass


class Label(StringEnum):
    BORDER = "BORDER"
    NOBORDER = "NOBORDER"
    S2S = "Scene-To-Scene"
    S2NS = "Scene-To-Nonscene"
    NS2S = "Nonscene-To-Scene"


scene_type_mapper = {
    "Szene Ebene 1": "Scene",
    "Szene": "Scene",
    "Szene Ebene 2": "Scene",
    "Szene Ebene 3": "Scene",
    "Nicht-Szene": "NonScene",
    "Scene": "Scene",
    "NonScene": "NonScene",
    "Nonscene": "NonScene",
}

root_dir = Path(__file__).parent.parent
datasets_folder = root_dir / "data" / "full"
