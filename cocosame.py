import argparse
import json

import funcy

parser = argparse.ArgumentParser(
    description="Splits COCO annotations file into training and test sets."
)
parser.add_argument(
    "original",
    type=str,
    help="Path to COCO annotations file.",
)
parser.add_argument("target", type=str, help="Where to store COCO training annotations")

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
    print(args)
    with open(args.original, "rt", encoding="UTF-8") as annotations:
        coco = json.load(annotations)
        images = coco["images"]

        print("original", sorted(int(i["file_name"][9:12]) for i in images))
        ims = set(i["file_name"] for i in images)
    with open(args.target, "rt", encoding="UTF-8") as annotations:
        coco = json.load(annotations)
        info = coco["info"]
        licenses = coco["licenses"]
        images = coco["images"]
        anns = coco["annotations"]
        categories = coco["categories"]

        print("target before", sorted(int(i["file_name"][9:12]) for i in images))

        images = [i for i in images if i["file_name"] in ims]
        print("target after", sorted(int(i["file_name"][9:12]) for i in images))
        save_coco(
            args.target.replace(".json", "_same.json"),
            info,
            licenses,
            images,
            filter_annotations(anns, images),
            categories,
        )


if __name__ == "__main__":
    main(args)
