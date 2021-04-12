# modified from https://github.com/akarazniewicz/cocosplit.git

import argparse
import glob
import json

parser = argparse.ArgumentParser(
    description="Splits COCO annotations file into training and test sets."
)
parser.add_argument(
    "annotations",
    metavar="coco_annotations",
    type=str,
    nargs="+",
    help="Path to COCO annotations file.",
)

args = parser.parse_args()


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, "wt", encoding="UTF-8") as coco:
        json.dump(
            {
                "info": info,
                "licenses": licenses,
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
            coco,
            indent=2,
            sort_keys=True,
        )


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

            mapIm = {}
            for i in range(len(images)):
                mapIm[images[i]["id"]] = i + 1
                images[i]["id"] = i + 1
            cat_id = 1
            map = {}
            for c in categories:
                map[c["id"]] = cat_id
                c["id"] = cat_id
                cat_id += 1
            ann_id = 1
            for a in annotations:
                a["category_id"] = map[a["category_id"]]
                a["image_id"] = mapIm[a["image_id"]]
                a["id"] = ann_id
                ann_id += 1

            save_coco(
                ann,
                info,
                licenses,
                images,
                annotations,
                categories,
            )

            # print("Saved {} entries in {} and {} in {}".format(len(x), args.train, len(y), args.test))


if __name__ == "__main__":
    main(args)
