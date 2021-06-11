import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as mplPath
from tqdm import tqdm
from pathlib import Path


def convert_polygon(polygon):
    """ converts [x1,y2,x2,y2,...] to boolean mask """
    width = 1920
    height = 1080
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T
    if isinstance(polygon, list):
        polygon = np.array(polygon)
    path = mplPath(polygon.reshape(-1, 2))
    grid = path.contains_points(points)
    grid = grid.reshape((height, width))

    return grid


def main(args):
    with open(args.annotations, "rt", encoding="UTF-8") as annotations:
        coco = json.load(annotations)
        annotations = coco["annotations"]
        categories = coco["categories"]

        cats = {}
        for c in categories:
            cats[c["id"]] = c["name"]

        heatmaps = {}
        try:
            for a in tqdm(annotations):
                a_mask = convert_polygon(a["segmentation"][0]).astype(int)
                cat = cats[a["category_id"]]
                if cat in heatmaps:
                    heatmaps[cat] += a_mask
                else:
                    heatmaps[cat] = a_mask
        except KeyboardInterrupt:
            pass

        Path(args.annotations).parent.joinpath("heatmaps").mkdir(exist_ok=True)
        plt.inferno()

        for cat in heatmaps:
            plt.imshow(heatmaps[cat])
            plt.title(cat)
            plt.savefig(
                Path(args.annotations).parent.joinpath(
                    "heatmaps",
                    f"{Path(args.annotations).stem}_{cat.replace(' ', '-')}.png",
                )
            )
            plt.figure()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parses COCO annotations and plots heatmaps."
    )
    parser.add_argument(
        "annotations",
        metavar="coco_annotations",
        type=str,
        help="Path to COCO annotations file.",
    )

    args = parser.parse_args()
    main(args)
