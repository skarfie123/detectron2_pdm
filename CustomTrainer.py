from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2_pdm.PDM_Evaluator import PDM_Evaluator
import os


class CustomTrainer(DefaultTrainer):
    vNotG = True
    vClasses = [1, 2, 3, 4]
    gClasses = [0, 1, 2, 3, 4, 5, 6]  # TODO find classes for ground

    @classmethod
    def classes(cls):
        return cls.vClasses if cls.vNotG else cls.gClasses

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        classes = cls.classes()
        return DatasetEvaluators(
            [
                COCOEvaluator(dataset_name, ("bbox", "segm"), True, output_folder),
                PDM_Evaluator(dataset_name, classes),
            ]
        )
