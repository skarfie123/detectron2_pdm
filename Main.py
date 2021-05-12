# pyright: reportMissingImports=false
import os

from detectron2 import model_zoo as mz
from detectron2.config import get_cfg as get_default
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset

from detectron2_pdm import Datasets
from detectron2_pdm.CustomConfig import CustomConfig
from detectron2_pdm.CustomTrainer import CustomTrainer
from detectron2_pdm.PDM_Evaluator import PDM_Evaluator

# TODO: make it not assume /content/outputs, and use os.path.join


def get_cfg(
    iterations: int = 1000,
    weights_file: str = None,
):
    cfg = get_default()
    cfg.TEST.EVAL_PERIOD = 100
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = iterations
    cfg.SOLVER.CHECKPOINT_PERIOD = CustomConfig.saveInterval
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

    cfg.OUTPUT_DIR = f"/content/outputs/{CustomConfig.trainingConfig.folder}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(f"Output Dir: {cfg.OUTPUT_DIR}")
    if CustomConfig.load(cfg.OUTPUT_DIR):
        print("Note: config existed in training folder so loaded it")

    cfg.DATASETS.TRAIN = (CustomConfig.trainingConfig.dataset + "_train",)
    cfg.DATASETS.TEST = (CustomConfig.trainingConfig.dataset + "_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = CustomConfig.trainingConfig.numClasses

    if CustomConfig.modelWeights == "":
        cfg.merge_from_file(mz.get_config_file(CustomConfig.model))
        cfg.MODEL.WEIGHTS = mz.get_checkpoint_url(CustomConfig.model)
    else:
        cfg.merge_from_file(CustomConfig.model)
        cfg.MODEL.WEIGHTS = CustomConfig.modelWeights
    if weights_file is not None:
        cfg.MODEL.WEIGHTS = f"{cfg.OUTPUT_DIR}/{weights_file}"

    Datasets.register(
        CustomConfig.trainingConfig.imageset,
        CustomConfig.trainingConfig.dataset,
    )

    return cfg


def train(
    iterations=None,
    evaluation=False,
    resume=True,
    save=False,
):
    cfg = get_cfg(iterations=iterations)

    CustomConfig.save(cfg.OUTPUT_DIR)

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()

    if evaluation:
        evaluate()

    if save:
        if resume or not os.exists(
            f"{CustomConfig.driveOutputs}/{CustomConfig.trainingConfig.folder}/"
        ):
            os.system(
                f"cp -rf /content/outputs/{CustomConfig.trainingConfig.folder} {CustomConfig.driveOutputs}"
            )
        else:
            print(
                "Warning: folder exists, did not save to Drive (did you forget to set resume=True?)"
            )
            while True:
                inp = input("Save and overwrite? (y/n): ").lower().strip()
                if inp == "y":
                    os.system(
                        f"cp -rf /content/outputs/{CustomConfig.trainingConfig.folder} {CustomConfig.driveOutputs}"
                    )
                    break
                elif inp == "n":
                    break
                print("Invalid answer")


def evaluate(
    subset: str = "_test",
    threshold: float = 0.7,
    weights_file="model_final.pth",
    testIndex: int = None,
):
    cfg = get_cfg(weights_file=weights_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)

    if testIndex is None:
        tests = list(range(len(CustomConfig.testingConfigs)))
    else:
        tests = [testIndex]

    results = {}
    for i in tests:
        print(f"Test: {i}")
        dataset = CustomConfig.testingConfigs[i].dataset
        pdmClasses = CustomConfig.testingConfigs[i].pdmClasses

        Datasets.register(
            CustomConfig.testingConfigs[i].imageset,
            dataset,
        )

        evaluator = DatasetEvaluators(
            [
                COCOEvaluator(
                    dataset + subset,
                    ("bbox", "segm"),
                    False,
                    output_dir=f"/content/outputs/{CustomConfig.testingConfigs[i].folder}/coco_eval_test",
                ),
                PDM_Evaluator(dataset + subset, pdmClasses),
            ]
        )
        test_loader = build_detection_test_loader(cfg, dataset + subset)
        result = inference_on_dataset(trainer.model, test_loader, evaluator)
        print(f"{CustomConfig.testingConfigs[i].folder} = {result}")
        results[CustomConfig.testingConfigs[i].folder] = result
    return results


def combine(v, g):
    """ Combine PDM results from vertical and ground into one so that it can be compared with merged """
    c = {i: v[i] for i in ["PDM: Presence", "PDM: Detection", "PDM: Measurement"]}
    for i in c:
        for j in g[i]:
            if j in c[i]:
                c[i][j] = (4 * c[i][j] + 5 * g[i][j]) / 9
            else:
                c[i][j] = g[i][j]
    return c


""" def evaluate_all_checkpoints(outputn):
    import logging

    log = logging.getLogger("detectron2")
    ll = log.getEffectiveLevel()
    log.setLevel(logging.WARNING)
    results = {}
    folder = f"/content/outputs/{CustomConfig.category}{outputn}"
    for file in sorted(os.listdir(folder)):
        if file.endswith(".pth") and not file.endswith("_final.pth"):
            print(">>>", file)
            results[file] = evaluate(model_file=file)
    log.setLevel(ll)
    return results """
