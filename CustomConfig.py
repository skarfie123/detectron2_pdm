# pyright: reportMissingImports=false
import os
import pprint
from dataclasses import dataclass
from enum import Enum, auto
from typing import List

import jsonpickle

from detectron2_pdm.MustBeSet import MustBeSet


class Category(Enum):
    MERGED = auto()
    GROUND = auto()
    VERTICAL = auto()


@dataclass
class ConfigSet:
    category: Category
    imageset: str
    dataset: str
    numClasses: int
    pdmClasses: List[int]
    folder: str


# TODO make pdmClasses List[str] to make it easier to config
# TODO check pdm classes function shows included and excluded


class CustomConfig(metaclass=MustBeSet):
    model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    modelWeights = ""
    saveInterval: int = 1000
    trainingConfig: ConfigSet = None  # type: ignore
    testingConfigs: List[ConfigSet] = None  # type: ignore

    driveOutputs: str = None  # type: ignore # path to all output folders
    driveDatasets: str = None  # type: ignore # path to all datasets

    @classmethod
    def set(
        cls,
        trainingConfig: ConfigSet,
        testingConfigs: List[ConfigSet],
        driveOutputs: str,
        driveDatasets: str,
        model: str = None,
        modelWeights: str = None,
        saveInterval: int = None,
    ):
        cls.trainingConfig = trainingConfig
        cls.testingConfigs = testingConfigs

        cls.set_drive(driveOutputs, driveDatasets)

        if model is not None:
            cls.model = model
        if modelWeights is not None:
            cls.modelWeights = modelWeights
        if saveInterval is not None:
            cls.saveInterval = saveInterval

    @classmethod
    def set_drive(
        cls,
        driveOutputs: str,
        driveDatasets: str,
    ):
        cls.driveOutputs = driveOutputs
        cls.driveDatasets = driveDatasets

    @classmethod
    def save(cls, config_dir="/content", config_file="config.json"):
        with open(os.path.join(config_dir, config_file), "w") as outfile:
            encoded = jsonpickle.encode(
                {
                    "trainingConfig": cls.trainingConfig,
                    "testingConfigs": cls.testingConfigs,
                    "model": cls.model,
                    "modelWeights": cls.modelWeights,
                    "saveInterval": cls.saveInterval,
                },
                indent=4,
            )
            outfile.write(encoded)

    @classmethod
    def load(cls, config_dir="/content", config_file="config.json") -> bool:
        if not os.path.exists(os.path.join(config_dir, config_file)):
            return False
        with open(os.path.join(config_dir, config_file), "r") as infile:
            decoded = jsonpickle.decode(infile.read())
            cls.trainingConfig = decoded["trainingConfig"]
            cls.testingConfigs = decoded["testingConfigs"]
            cls.model = decoded["model"]
            cls.modelWeights = decoded["modelWeights"]
            cls.saveInterval = decoded["saveInterval"]
        return True

    @classmethod
    def pretty(cls):
        pprint.pprint(
            {
                "trainingConfig": cls.trainingConfig,
                "testingConfigs": cls.testingConfigs,
                "driveOutputs": cls.driveOutputs,
                "driveDatasets": cls.driveDatasets,
                "model": cls.model,
                "modelWeights": cls.modelWeights,
                "saveInterval": cls.saveInterval,
            }
        )


IMAGESET_MASK_VERTICAL = "mask_vertical"
IMAGESET_MASK_GROUND = "mask_ground"
IMAGESET_ORIGINAL_IMAGES = "original_images"
