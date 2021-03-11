# modified from https://github.com/akarazniewicz/cocosplit.git

import json
import argparse
import funcy
from sklearn.model_selection import train_test_split
import numpy as np
import glob

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
parser.add_argument(
    "--having-annotations",
    dest="having_annotations",
    action="store_true",
    help="Ignore all images without annotations. Keep only these with at least one annotation",
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


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i["id"]), images)
    return funcy.lfilter(lambda a: int(a["image_id"]) in image_ids, annotations)


def main(args):
    if len(args.annotations) == 1:
        args.annotations = glob.glob(args.annotations[0])
    print(*args.annotations, sep="\n")
    for ann in args.annotations:
        with open(ann, "rt", encoding="UTF-8") as annotations:
            coco = json.load(annotations)
            info = coco["info"]
            licenses = coco["licenses"]
            images = coco["images"]
            annotations = coco["annotations"]
            categories = coco["categories"]

            print(coco.keys())

            print("Original", len(images))

            def nothing():
                pass

            funcy.lmap(
                lambda a: print(a, next(i for i in images if i["id"] == a["image_id"]))
                if a["segmentation"] == []
                else nothing(),
                annotations,
            )

            """ print("Annotations", len(annotations))
            a2 = []
            for i in range(len(annotations)):
                if (
                    max(annotations[i]["bbox"][2], annotations[i]["bbox"][3]) < 50
                    or min(annotations[i]["bbox"][2], annotations[i]["bbox"][3]) < 30
                ):
                    pass
                else:
                    a2.append(annotations[i])
            annotations = a2
            print("Annotations filtered by size", len(annotations))

            c2 = []
            ch = []
            for c in categories:
                if c["name"] == "human" or c["name"] == "car":
                    ch.append(c["id"])
                else:
                    c2.append(c)
            print(len(c2), len(categories))
            categories = c2

            a2 = []
            for i in range(len(annotations)):
                if annotations[i]["category_id"] in ch:
                    pass
                else:
                    a2.append(annotations[i])
            annotations = a2
            print("Annotations filtered cars and humans", len(annotations)) """

            images_with_annotations = funcy.lmap(
                lambda a: int(a["image_id"]), annotations
            )

            images = funcy.lremove(
                lambda i: i["id"] not in images_with_annotations, images
            )

            print("Removed empty images", len(images))

            images = funcy.lremove(lambda i: "copy" in i["file_name"].lower(), images)

            print("Removed copy", len(images))

            def f(e):
                return e["file_name"]

            # print(len(images), end=" - >")
            images.sort(key=f)
            images = images[-300:]
            # funcy.lmap(lambda i : print(i['file_name'][9:12], end="\t"), images)

            print(len(images))

            no_segm = funcy.lfilter(lambda a: len(a["segmentation"]) == 0, annotations)
            print(len(no_segm), len(annotations))
            image_ids = funcy.lmap(lambda i: i["image_id"], no_segm)
            funcy.lmap(
                lambda i: print("! no segm annot in #" + i["file_name"])
                if i["id"] in image_ids
                else nothing(),
                images,
            )

            """ save_coco(
                ann,
                info,
                licenses,
                images,
                filter_annotations(annotations, images),
                categories,
            ) """

            # print("Saved {} entries in {} and {} in {}".format(len(x), args.train, len(y), args.test))


if __name__ == "__main__":
    main(args)