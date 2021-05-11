import argparse
import glob
import json


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

            labels = {}
            generality = {}

            for c in categories:
                labels[c["id"]] = c["name"]
                generality[c["name"]] = {i["id"]: 0 for i in images}

            for a in annotations:
                generality[labels[a["category_id"]]][a["image_id"]] += 1

            for k in generality:
                generality[k] = round(
                    len([i for i in generality[k].values() if i > 0])
                    / len(generality[k].values()),
                    2,
                )

            generality["Average"] = round(
                sum(generality.values()) / len(generality.values()), 2
            )

            print(ann, generality, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Counts COCO annotations per category")
    parser.add_argument(
        "annotations",
        metavar="coco_annotations",
        type=str,
        nargs="+",
        help="Path to COCO annotations file.",
    )

    args = parser.parse_args()
    main(args)
