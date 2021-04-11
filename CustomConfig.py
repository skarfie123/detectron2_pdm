import os
from enum import Enum, auto
from dataclasses import dataclass
from typing import List
from detectron2_pdm.MustBeSet import MustBeSet
import jsonpickle
import pprint


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


class CustomConfig(metaclass=MustBeSet):
    trainingConfig: ConfigSet = None
    testingConfigs: List[ConfigSet] = None

    driveOutputs: str = None  # path to all output folders
    driveDatasets: str = None  # path to all datasets

    @classmethod
    def set(
        cls,
        trainingConfig: ConfigSet,
        testingConfigs: List[ConfigSet],
        driveOutputs: str,
        driveDatasets: str,
    ):
        cls.trainingConfig = trainingConfig
        cls.testingConfigs = testingConfigs

        cls.set_drive(driveOutputs, driveDatasets)

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
                },
                indent=4,
            )
            outfile.write(encoded)

    @classmethod
    def load(cls, config_dir="/content", config_file="config.json"):
        if not os.path.exists(os.path.join(config_dir, config_file)):
            return
        with open(os.path.join(config_dir, config_file), "r") as infile:
            decoded = jsonpickle.decode(infile.read())
            cls.trainingConfig = decoded["trainingConfig"]
            cls.testingConfigs = decoded["testingConfigs"]

    @classmethod
    def pretty(cls):
        return pprint.pformat(
            {
                "trainingConfig": cls.trainingConfig,
                "testingConfigs": cls.testingConfigs,
                "driveOutputs": cls.driveOutputs,
                "driveDatasets": cls.driveDatasets,
            }
        )


IMAGESET_MASK_VERTICAL = "mask_vertical"
IMAGESET_MASK_GROUND = "mask_ground"
IMAGESET_ORIGINAL_IMAGES = "original_images"