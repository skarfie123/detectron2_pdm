# modified from https://github.com/akarazniewicz/cocosplit.git

import argparse
import glob
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
            print(list((i["name"], i["id"]) for i in categories))

            print("Original", len(images))
            print("Annotations", len(annotations))

            # filter by size
            if args.filter_size:
                a2 = []
                for i in range(len(annotations)):
                    if (
                        max(annotations[i]["bbox"][2], annotations[i]["bbox"][3]) < 50
                        or min(annotations[i]["bbox"][2], annotations[i]["bbox"][3])
                        < 30
                    ):
                        pass
                    else:
                        a2.append(annotations[i])
                annotations = a2
                print("Annotations filtered by size", len(annotations))

            # remove images with no annotations
            images_with_annotations = funcy.lmap(
                lambda a: int(a["image_id"]), annotations
            )
            images = funcy.lremove(
                lambda i: i["id"] not in images_with_annotations, images
            )
            print("Removed empty images", len(images))

            # remove images copy images
            images = funcy.lremove(lambda i: "copy" in i["file_name"].lower(), images)
            print("Removed copy", len(images))

            # warn about empty annotations
            def nothing():
                pass

            no_segm = funcy.lfilter(lambda a: len(a["segmentation"]) == 0, annotations)
            if len(no_segm) > 0:
                print(f"{len(no_segm) = }, {len(annotations) = }")
            image_ids = funcy.lmap(lambda a: a["image_id"], no_segm)
            funcy.lmap(
                lambda i: print("! no segm annotation in #" + i["file_name"])
                if i["id"] in image_ids
                else nothing(),
                images,
            )

            save_coco(
                ann,
                info,
                licenses,
                images,
                filter_annotations(annotations, images),
                categories,
            )

            # print("Saved {} entries in {} and {} in {}".format(len(x), args.train, len(y), args.test))


if __name__ == "__main__":
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
        "--filter-size",
        dest="filter_size",
        action="store_true",
        help="filter annotations that are too small",
    )

    args = parser.parse_args()
    main(args)
