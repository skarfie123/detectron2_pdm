from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from google.colab.patches import cv2_imshow
import cv2
from detectron2_pdm.Main import get_cfg, find_outputn
from detectron2_pdm.CustomTrainer import CustomTrainer
import random as rnd
import numpy as np


def compare(
    cfg, set="_test", filterAnnotation=None, filterOutput=None, random=None, scale=0.5
):
    if cfg is None:
        cfg = get_cfg(find_outputn())
    dataset = "vertical_200" if CustomTrainer.vNotG else "ground_200"
    predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get(dataset + set)
    if random is not None:
        dataset_dicts = rnd.sample(dataset_dicts, random)
    count = 0
    for d in dataset_dicts:
        if (
            filterAnnotation is not None
            and len(
                list(
                    filter(
                        lambda a: a["category_id"] == filterAnnotation, d["annotations"]
                    )
                )
            )
            == 0
        ):
            continue

        count += 1
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
                outputs["instances"][outputs["instances"].pred_classes == filterOutput]
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
    print(count)