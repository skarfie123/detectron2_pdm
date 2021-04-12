# pyright: reportMissingImports=false
import os
import subprocess

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from detectron2_pdm.CustomConfig import CustomConfig


def download(imageset, dataset):
    if not os.path.exists(f"/content/{imageset}/"):
        subprocess.call(
            f"cp {CustomConfig.driveDatasets}/{imageset}.zip /content/",
            shell=True,
        )
        subprocess.call(f"unzip {imageset}.zip > /dev/null", shell=True)
        subprocess.call(f"mkdir {imageset}", shell=True)
        subprocess.call(f"mv img* {imageset}/", shell=True)
    if not (
        os.path.exists(f"/content/{dataset}_train.json")
        and os.path.exists(f"/content/{dataset}_val.json")
        and os.path.exists(f"/content/{dataset}_test.json")
    ):
        subprocess.call(
            f"cp {CustomConfig.driveDatasets}/{dataset}.zip /content/",
            shell=True,
        )
        subprocess.call(f"unzip {dataset}.zip > /dev/null", shell=True)


def register(imageset, dataset):
    download(imageset, dataset)
    try:
        register_coco_instances(
            f"{dataset}_train",
            {},
            f"/content/{dataset}/{dataset}_train.json",
            f"/content/{imageset}/",
        )
        register_coco_instances(
            f"{dataset}_val",
            {},
            f"/content/{dataset}/{dataset}_val.json",
            f"/content/{imageset}/",
        )
        register_coco_instances(
            f"{dataset}_test",
            {},
            f"/content/{dataset}/{dataset}_test.json",
            f"/content/{imageset}/",
        )
    except AssertionError:
        print("Dataset already registered")


def clear():
    DatasetCatalog.clear()
    MetadataCatalog.clear()
