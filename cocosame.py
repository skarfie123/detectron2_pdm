import argparse
import json

import funcy


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
        categories = coco["categories"]

        print("original", sorted(int(i["file_name"][9:12]) for i in images))
        ims = set(i["file_name"] for i in images)
        cats_orig = {c["name"]: c["id"] for c in categories}
    print(cats_orig)
    with open(args.target, "rt", encoding="UTF-8") as annotations:
        coco2 = json.load(annotations)
        info2 = coco2["info"]
        licenses2 = coco2["licenses"]
        images2 = coco2["images"]
        anns2 = coco2["annotations"]
        categories2 = coco2["categories"]

        print("target before", sorted(int(i["file_name"][9:12]) for i in images))
        print("")

        images2 = [i for i in images2 if i["file_name"] in ims]
        cats_target = {c["id"]: c["name"] for c in categories2}
        for a in anns2:
            a["category_id"] = cats_orig[cats_target[a["category_id"]]]
        print("target after", sorted(int(i["file_name"][9:12]) for i in images))
        print("")
        print(cats_target)
        print("")
        print(categories, categories2)
        save_coco(
            args.target,
            info2,
            licenses2,
            images2,
            filter_annotations(anns2, images2),
            categories,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splits COCO annotations file into training and test sets."
    )
    parser.add_argument(
        "original",
        type=str,
        help="Path to COCO annotations file.",
    )
    parser.add_argument(
        "target", type=str, help="Where to store COCO training annotations"
    )

    args = parser.parse_args()
    main(args)
