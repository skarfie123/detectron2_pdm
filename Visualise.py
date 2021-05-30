# pyright: reportMissingImports=false
import os
import random as rnd

import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from google.colab.patches import cv2_imshow

from detectron2_pdm import Datasets
from detectron2_pdm.CustomConfig import CustomConfig
from detectron2_pdm.Main import get_cfg


def compare(
    cfg=None,
    subset: str = "_test",
    filterAnnotation: int = None,
    filterOutput: int = None,
    random: int = None,
    scale: float = 0.5,
    threshold: float = 0.7,
    original: bool = False,
    testIndex: int = None,
    weights_file="model_final.pth",
    save=False,
):
    if cfg is None:
        cfg = get_cfg(weights_file=weights_file)
    elif weights_file is not None:
        cfg.MODEL.WEIGHTS = f"{cfg.OUTPUT_DIR}/{weights_file}"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    predictor = DefaultPredictor(cfg)

    if testIndex is None:
        tests = list(range(len(CustomConfig.testingConfigs)))
    else:
        tests = [testIndex]

    randomSelection = None

    for i in tests:
        print(f"Test: {i} - {CustomConfig.testingConfigs[i].folder}")
        dataset = CustomConfig.testingConfigs[i].dataset
        Datasets.register(
            CustomConfig.testingConfigs[i].imageset,
            dataset,
        )

        dataset_dicts = DatasetCatalog.get(dataset + subset)
        # TODO: apply filters before random
        if random is not None:
            if randomSelection is None:
                dataset_dicts = rnd.sample(dataset_dicts, random)
                randomSelection = {
                    os.path.basename(d["file_name"]) for d in dataset_dicts
                }
            else:
                dataset_dicts = [
                    d
                    for d in dataset_dicts
                    if os.path.basename(d["file_name"]) in randomSelection
                ]

        count = 0
        errors = 0
        for d in dataset_dicts:
            base = os.path.basename(d["file_name"])
            print(base)
            try:
                if (
                    filterAnnotation is not None
                    and len(
                        list(
                            filter(
                                lambda a: a["category_id"] == filterAnnotation,
                                d["annotations"],
                            )
                        )
                    )
                    == 0
                ):
                    continue

                im = cv2.imread(d["file_name"])
                outputs = predictor(im)
                outputs["instances"] = outputs["instances"][
                    [
                        (
                            outputs["instances"].pred_classes[j]
                            in CustomConfig.testingConfigs[i].pdmClasses
                        )
                        for j in range(len(outputs["instances"].pred_classes))
                    ]
                ]
                v = Visualizer(
                    im[:, :, ::-1],
                    metadata=MetadataCatalog.get(dataset + subset),
                    scale=scale,
                    instance_mode=ColorMode.IMAGE_BW,
                )
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

                if (
                    filterOutput is not None
                    and len(
                        outputs["instances"][
                            outputs["instances"].pred_classes == filterOutput
                        ]
                    )
                    == 0
                ):
                    continue

                visualizer = Visualizer(
                    im[:, :, ::-1],
                    metadata=MetadataCatalog.get(dataset + subset),
                    scale=scale,
                )
                out2 = visualizer.draw_dataset_dict(
                    {
                        **d,
                        "annotations": [
                            j
                            for j in d["annotations"]
                            if j["category_id"]
                            in CustomConfig.testingConfigs[i].pdmClasses
                        ],
                    }
                )
                if original:
                    img = np.concatenate(
                        (
                            cv2.resize(im, (0, 0), fx=0.5, fy=0.5),
                            out.get_image()[:, :, ::-1],
                            out2.get_image()[:, :, ::-1],
                        ),
                        axis=1,
                    )
                else:
                    img = np.concatenate(
                        (
                            out.get_image()[:, :, ::-1],
                            out2.get_image()[:, :, ::-1],
                        ),
                        axis=1,
                    )
                if save:
                    cv2.imwrite(
                        f"/content/outputs/{CustomConfig.testingConfigs[i].folder}/{base}",
                        img,
                    )
                else:
                    cv2_imshow(img)
                count += 1
            except Exception:
                print("Error:", d["file_name"])
                errors += 1
        print(f"Count: {count}, Errors{errors}")
