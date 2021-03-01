import os
from detectron2 import model_zoo
from detectron2.config import get_cfg as get_default
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset
from detectron2_pdm.CustomTrainer import CustomTrainer
from detectron2_pdm.PDM_Evaluator import PDM_Evaluator
from detectron2.data import build_detection_test_loader


def get_cfg(
    outputn,
    model="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    iterations=1000,
    output_dir=None,
):
    dataset = "vertical_200" if CustomTrainer.vNotG else "ground_200"
    cfg = get_default()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.DATASETS.TRAIN = (dataset + "_train",)
    cfg.DATASETS.TEST = (dataset + "_val",)
    cfg.TEST.EVAL_PERIOD = 100
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6 if CustomTrainer.vNotG else 7
    if isinstance(outputn, str):
        cfg.OUTPUT_DIR = outputn
    else:
        cfg.OUTPUT_DIR = (
            "./outputs/" + ("vertical" if CustomTrainer.vNotG else "ground") + str(outputn)
        )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(f"Output Dir: {cfg.OUTPUT_DIR}")
    return cfg


outputk = 0


def find_outputk():
    while os.path.exists(
        "/content/outputs/"
        + ("vertical" if CustomTrainer.vNotG else "ground")
        + str(outputk)
    ):
        outputk += 1


def train(
    iterations=None,
    cfg=None,
    evaluation=True,
    classes=None,
    resume=True,
    save=False,
    overwrite=False,
):
    if cfg is None:
        find_outputk()
        cfg = get_cfg(outputk)
    if not iterations is None:
        cfg.MAX_ITER = iterations
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()

    if evaluation:
        evaluate(cfg, trainer, classes)

    if save and (
        overwrite
        or not os.exists(
            f"/content/gdrive/My\\ Drive/4YP\\ Output/detectron/{cfg.OUTPUT_DIR.split('/')[-1]}/"
        )
    ):
        os.system(
            f"cp -r /content/outputs/{cfg.OUTPUT_DIR.split('/')[-1]}/* /content/gdrive/My\\ Drive/4YP\\ Output/detectron/{cfg.OUTPUT_DIR.split('/')[-1]}/"
        )


def evaluate(cfg=None, trainer=None, classes=None):
    if cfg is None:
        find_outputk()
        cfg = get_cfg(outputk - 1)
    dataset = "vertical_200" if CustomTrainer.vNotG else "ground_200"
    if classes is None:
        classes = CustomTrainer.classes()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    if trainer is None:
        trainer = CustomTrainer(cfg)
        trainer.resume_or_load(resume=False)

    evaluator = DatasetEvaluators(
        [
            COCOEvaluator(
                dataset + "_test",
                ("bbox", "segm"),
                False,
                output_dir=cfg.OUTPUT_DIR + "/coco_eval_test",
            ),
            PDM_Evaluator(dataset + "_test", classes),
        ]
    )
    test_loader = build_detection_test_loader(cfg, dataset + "_test")
    print(
        f"{cfg.OUTPUT_DIR.split('/')[-1]}={inference_on_dataset(trainer.model, test_loader, evaluator)}"
    )
