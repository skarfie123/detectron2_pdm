import os
import subprocess

from detectron2.data.datasets import register_coco_instances


def download(imageset, dataset):
    if not os.path.exists(f"/content/{imageset}/"):
        subprocess.call(
            f"cp /content/gdrive/MyDrive/Share/4YPDatasets/{imageset}.zip /content/",
            shell=True,
        )
        subprocess.call(f"unzip {imageset}.zip > /dev/null", shell=True)
        subprocess.call(f"mkdir {imageset}", shell=True)
        subprocess.call(f"mv img* {imageset}/", shell=True)
    if not os.path.exists(f"/content/{dataset}_train.json"):
        subprocess.call(
            f"cp /content/gdrive/MyDrive/Share/4YPDatasets/{dataset}.zip /content/",
            shell=True,
        )
        subprocess.call(f"unzip {dataset}.zip > /dev/null", shell=True)


def register(imageset, dataset):
    download(imageset, dataset)
    register_coco_instances(
        f"{dataset}_train",
        {},
        f"/content/{dataset}_train.json",
        f"/content/{imageset}/",
    )
    register_coco_instances(
        f"{dataset}_val",
        {},
        f"/content/{dataset}_val.json",
        f"/content/{imageset}/",
    )
    register_coco_instances(
        f"{dataset}_test",
        {},
        f"/content/{dataset}_test.json",
        f"/content/{imageset}/",
    )
