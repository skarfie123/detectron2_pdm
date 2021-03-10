from detectron2_pdm import Datasets


class MustSet(type):
    def __getattribute__(cls, key):
        if super().__getattribute__(key) is None:
            raise Exception("Config has not been set, use CustomConfig.set()")
        return super().__getattribute__(key)


import os
import json


class CustomConfig(metaclass=MustSet):
    category = None
    imageset = None
    dataset = None
    numClasses = None
    pdmClasses = None

    @classmethod
    def set(
        cls,
        category: str,
        imageset: str,
        dataset: str,
        numClasses: int,
        pdmClasses: list,
    ):
        cls.category = category
        cls.imageset = imageset
        cls.dataset = dataset
        cls.numClasses = numClasses
        cls.pdmClasses = pdmClasses

        Datasets.register(imageset, dataset)

    @classmethod
    def save(cls, output_dir="/content"):
        with open(os.path.join(output_dir, "config.json"), "w") as outfile:
            json.dump(
                {
                    "category": cls.category,
                    "imageset": cls.imageset,
                    "dataset": cls.dataset,
                    "numClasses": cls.numClasses,
                    "pdmClasses": cls.pdmClasses,
                },
                outfile,
            )
