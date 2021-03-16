import os

from detectron2 import model_zoo as mz
from detectron2.config import get_cfg as get_default
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset

from detectron2_pdm import Datasets
from detectron2_pdm.CustomConfig import CustomConfig
from detectron2_pdm.CustomTrainer import CustomTrainer
from detectron2_pdm.PDM_Evaluator import PDM_Evaluator


def get_cfg(
    outputn,
    model="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    iterations=1000,
    model_zoo=True,
    weights_file=None,
):
    dataset = CustomConfig.dataset
    cfg = get_default()
    if model_zoo:
        cfg.merge_from_file(mz.get_config_file(model))
        cfg.MODEL.WEIGHTS = mz.get_checkpoint_url(model)
    else:
        cfg.merge_from_file(model)
        if weights_file is None:
            raise Exception("Please provide path the weights file")
        cfg.MODEL.WEIGHTS = weights_file
    cfg.DATASETS.TRAIN = (dataset + "_train",)
    cfg.DATASETS.TEST = (dataset + "_val",)
    cfg.TEST.EVAL_PERIOD = 100
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = iterations
    cfg.SOLVER.CHECKPOINT_PERIOD = 3000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = CustomConfig.numClasses
    if isinstance(outputn, str):
        cfg.OUTPUT_DIR = outputn
    else:
        cfg.OUTPUT_DIR = f"./outputs/{CustomConfig.category}{outputn}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(f"Output Dir: {cfg.OUTPUT_DIR}")
    CustomConfig.load(cfg.OUTPUT_DIR)
    Datasets.register(CustomConfig.imageset, CustomConfig.dataset)
    return cfg


def find_outputn():
    """Gives latest outputn"""
    outputn = 0
    while os.path.exists(f"/content/outputs/{CustomConfig.category}{outputn}"):
        outputn += 1
    return outputn - 1


def train(
    iterations=None,
    cfg=None,
    evaluation=True,
    classes=None,
    resume=True,
    save=False,
):
    if cfg is None:
        cfg = get_cfg(find_outputn() + 1)
    if not iterations is None:
        cfg.SOLVER.MAX_ITER = iterations

    CustomConfig.save(cfg.OUTPUT_DIR)

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()

    if evaluation:
        evaluate(cfg, trainer, classes)

    if save and (
        resume
        or not os.exists(
            f"{CustomConfig.driveOutputs}/{cfg.OUTPUT_DIR.split('/')[-1]}/"
        )
    ):
        # TODO: this doesnt work when a OUTPUT_DIR is in another dir
        # it assumes /content/outputs/folder
        # but doesnt work for /content/outputs/another/folder
        # maybe change OUTPUT_DIR from relative to absolute and just use as a whole here?
        os.system(
            f"cp -rf /content/outputs/{cfg.OUTPUT_DIR.split('/')[-1]} {CustomConfig.driveOutputs}"
        )
    elif save:
        print(
            "Warning: folder exists, did not save to Drive (did you forget to set resume=True?)"
        )


def evaluate(
    cfg=None,
    trainer=None,
    pdmClasses=None,
    set="_test",
    threshold=0.7,
    model_file="model_final.pth",
):
    if cfg is None:
        cfg = get_cfg(find_outputn())
    dataset = CustomConfig.dataset
    if pdmClasses is None:
        pdmClasses = CustomConfig.pdmClasses
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    if trainer is None:
        trainer = CustomTrainer(cfg)
        trainer.resume_or_load(resume=False)

    evaluator = DatasetEvaluators(
        [
            COCOEvaluator(
                dataset + set,
                ("bbox", "segm"),
                False,
                output_dir=cfg.OUTPUT_DIR + "/coco_eval_test",
            ),
            PDM_Evaluator(dataset + set, pdmClasses),
        ]
    )
    test_loader = build_detection_test_loader(cfg, dataset + set)
    result = inference_on_dataset(trainer.model, test_loader, evaluator)
    print(f"{cfg.OUTPUT_DIR.split('/')[-1]}_{cfg.SOLVER.MAX_ITER} = {result}")
    return result


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


def evaluate_all_checkpoints(outputn):
    import logging

    log = logging.getLogger("detectron2")
    ll = log.getEffectiveLevel()
    log.setLevel(logging.WARNING)
    results = {}
    folder = f"/content/outputs/{CustomConfig.category}{outputn}"
    for file in sorted(os.listdir(folder)):
        if file.endswith(".pth") and not file.endswith("_final.pth"):
            print(">>>", file)
            results[file] = evaluate(cfg=get_cfg(outputn), model_file=file)
    log.setLevel(ll)
    return results