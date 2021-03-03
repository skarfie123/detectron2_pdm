import os
import subprocess

from detectron2.data.datasets import register_coco_instances


def downloadGround():
    subprocess.call(
        "cp /content/gdrive/MyDrive/Share/4YPDatasets/mask_ground.zip /content/",
        shell=True,
    )
    subprocess.call(
        "cp /content/gdrive/MyDrive/Share/4YPDatasets/ground_200_annotations.zip /content/",
        shell=True,
    )
    subprocess.call("unzip mask_ground.zip > /dev/null", shell=True)
    subprocess.call("mkdir mask_ground", shell=True)
    subprocess.call("mv img* mask_ground/", shell=True)
    subprocess.call("unzip ground_200_annotations.zip > /dev/null", shell=True)


def downloadVertical():
    subprocess.call(
        "cp /content/gdrive/MyDrive/Share/4YPDatasets/mask_vertical.zip /content/",
        shell=True,
    )
    subprocess.call(
        "cp /content/gdrive/MyDrive/Share/4YPDatasets/vertical_300_annotations.zip /content/",
        shell=True,
    )
    subprocess.call("unzip mask_vertical.zip > /dev/null", shell=True)
    subprocess.call("mkdir mask_vertical", shell=True)
    subprocess.call("mv img* mask_vertical/", shell=True)
    subprocess.call("unzip vertical_300_annotations.zip > /dev/null", shell=True)


def registerGround():
    if not os.path.exists("/content/mask_ground/"):
        downloadGround()
    register_coco_instances(
        "ground_200_train",
        {},
        "/content/ground_200_train.json",
        "/content/mask_ground/",
    )
    register_coco_instances(
        "ground_200_val", {}, "/content/ground_200_val.json", "/content/mask_ground/"
    )
    register_coco_instances(
        "ground_200_test", {}, "/content/ground_200_test.json", "/content/mask_ground/"
    )


def registerVertical():
    if not os.path.exists("/content/mask_vertical/"):
        downloadVertical()
    register_coco_instances(
        "vertical_300_train",
        {},
        "/content/vertical_300_train.json",
        "/content/mask_vertical/",
    )
    register_coco_instances(
        "vertical_300_val",
        {},
        "/content/vertical_300_val.json",
        "/content/mask_vertical/",
    )
    register_coco_instances(
        "vertical_300_test",
        {},
        "/content/vertical_300_test.json",
        "/content/mask_vertical/",
    )
