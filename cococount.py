import json
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Counts COCO annotations per category")
parser.add_argument(
    "annotations",
    metavar="coco_annotations",
    type=str,
    nargs="+",
    help="Path to COCO annotations file.",
)

args = parser.parse_args()


def main(args):
    if len(args.annotations) == 1:
        args.annotations = glob.glob(args.annotations[0])
    outputs = {}
    for ann in args.annotations:
        with open(ann, "rt", encoding="UTF-8") as annotations:
            coco = json.load(annotations)
            info = coco["info"]
            licenses = coco["licenses"]
            images = coco["images"]
            annotations = coco["annotations"]
            categories = coco["categories"]

            counts = {}
            labels = {}

            for c in categories:
                labels[c["id"]] = c["name"]
                counts[c["name"]] = 0

            for a in annotations:
                counts[labels[a["category_id"]]] += 1

            print(ann, sum(counts.values()), counts, len(images), sep="\t")

            outputs[ann.split(".")[0].split("\\")[1]] = counts
    metrics = set()
    for output in outputs.values():
        metrics = metrics.union(set(output.keys()))
    for metric in metrics:
        for output in outputs.values():
            if metric not in output:
                output[metric] = 0
    x = np.arange(0, len(metrics) * (len(outputs.keys()) + 1), len(outputs.keys()) + 1)

    for i, o in enumerate(outputs.keys()):
        plt.bar(
            x + i + 0.5, [outputs[o][metric] for metric in metrics], width=1, label=o
        )
    plt.xticks(x, [m.split(":")[0] for m in metrics], rotation=90)
    # plt.hlines(1, x[0], x[-1])
    plt.legend()
    plt.title("Counts")
    plt.show()


if __name__ == "__main__":
    main(args)