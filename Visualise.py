import os
import random as rnd

import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from google.colab.patches import cv2_imshow

from detectron2_pdm.CustomTrainer import CustomTrainer
from detectron2_pdm.Main import find_outputn, get_cfg


def compare(
    cfg=None,
    set="_test",
    filterAnnotation=None,
    filterOutput=None,
    random=None,
    scale=0.5,
    threshold=0.7,
    outputn=None,
):
    if outputn is None:
        outputn = find_outputn()
    if cfg is None:
        cfg = get_cfg(outputn)
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    dataset = CustomTrainer.dataset()
    predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get(dataset + set)
    if random is not None:
        dataset_dicts = rnd.sample(dataset_dicts, random)
    count = 0
    errors = 0
    for d in dataset_dicts:
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
            v = Visualizer(
                im[:, :, ::-1],
                metadata=MetadataCatalog.get(dataset + set),
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
                metadata=MetadataCatalog.get("vertical_200_train"),
                scale=scale,
            )
            out2 = visualizer.draw_dataset_dict(d)
            cv2_imshow(
                np.concatenate(
                    (out.get_image()[:, :, ::-1], out2.get_image()[:, :, ::-1]), axis=1
                )
            )
            count += 1
        except:
            print("Error:", d["file_name"])
            errors += 1
    print(f"Count: {count}, Errors{errors}")
