# modified from https://github.com/akarazniewicz/cocosplit.git

import argparse
import json




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
    with open(args.annotations[0], "rt", encoding="UTF-8") as annotations1, open(
        args.annotations[1], "rt", encoding="UTF-8"
    ) as annotations2:
        coco1 = json.load(annotations1)
        info1 = coco1["info"]
        licenses1 = coco1["licenses"]
        images1 = coco1["images"]
        annotations1 = coco1["annotations"]
        categories1 = coco1["categories"]
        coco2 = json.load(annotations2)
        images2 = coco2["images"]
        annotations2 = coco2["annotations"]
        categories2 = coco2["categories"]
        print(annotations1[0])
        print(
            len(set(im["id"] for im in annotations1)),
            len(set(im["id"] for im in annotations2)),
            len(set(im["id"] for im in annotations1 + annotations2)),
        )
        print(
            len(set(im["id"] for im in categories1)),
            len(set(im["id"] for im in categories2)),
            len(set(im["id"] for im in categories1 + categories2)),
        )

        mapIm1 = {}
        for i in range(len(images1)):
            mapIm1[images1[i]["id"]] = int(images1[i]["file_name"][9:12])
            images1[i]["id"] = int(images1[i]["file_name"][9:12])
        mapIm2 = {}
        for i in range(len(images2)):
            mapIm2[images2[i]["id"]] = int(images2[i]["file_name"][9:12])
            # note relies on both having same images, but not necessarily same image_ids
        cat_id = 1
        ann_id = 1
        map1 = {}
        map2 = {}
        for c in categories1:
            map1[c["id"]] = cat_id
            c["id"] = cat_id
            cat_id += 1
        for c in categories2:
            map2[c["id"]] = cat_id
            c["id"] = cat_id
            cat_id += 1
        for a in annotations1:
            a["category_id"] = map1[a["category_id"]]
            a["image_id"] = mapIm1[a["image_id"]]
            a["id"] = ann_id
            ann_id += 1
        for a in annotations2:
            a["category_id"] = map2[a["category_id"]]
            a["image_id"] = mapIm2[a["image_id"]]
            a["id"] = ann_id
            ann_id += 1
        print(
            len(set(im["id"] for im in annotations1)),
            len(set(im["id"] for im in annotations2)),
            len(set(im["id"] for im in annotations1 + annotations2)),
        )
        print(
            len(set(im["id"] for im in categories1)),
            len(set(im["id"] for im in categories2)),
            len(set(im["id"] for im in categories1 + categories2)),
        )

        save_coco(
            args.annotations[2],
            info1,
            licenses1,
            images1,
            annotations1 + annotations2,
            categories1 + categories2,
        )

        # print("Saved {} entries in {} and {} in {}".format(len(x), args.train, len(y), args.test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Splits COCO annotations file into training and test sets."
    )
    parser.add_argument(
        "annotations",
        nargs=3,
        metavar="coco_annotations",
        type=str,
        help="Path to COCO annotations file.",
    )

    args = parser.parse_args()
    main(args)
