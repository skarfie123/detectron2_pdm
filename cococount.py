import json
import argparse
import glob

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

            print(ann, sum(counts.values()), counts, sep="\t")


if __name__ == "__main__":
    main(args)