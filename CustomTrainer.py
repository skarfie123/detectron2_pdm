from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from PDM_BBOX_Evaluator import PDM_BBOX
import os


class CustomTrainer(DefaultTrainer):
    vNotG = True
    # def __init__(self, cfg, vNotG):
    #     super().__init__(cfg)
    #     self.vNotG = vNotG
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluators(
            [
                COCOEvaluator(dataset_name, ("bbox", "segm"), True, output_folder),
                PDM_BBOX(
                    dataset_name,
                    [1, 2, 3, 4] if cls.vNotG else [0, 1, 2, 3, 4, 5, 6],  # TODO: find
                ),
            ]
        )
