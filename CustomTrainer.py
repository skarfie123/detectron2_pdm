import os

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators

from detectron2_pdm.PDM_Evaluator import PDM_Evaluator
from detectron2_pdm.CustomConfig import CustomConfig


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        pdmClasses = CustomConfig.pdmClasses
        return DatasetEvaluators(
            [
                COCOEvaluator(dataset_name, ("bbox", "segm"), True, output_folder),
                PDM_Evaluator(dataset_name, pdmClasses),
            ]
        )
