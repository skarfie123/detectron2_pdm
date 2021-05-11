import itertools
from collections import OrderedDict

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from matplotlib.path import Path
from typing import List, Dict


DETECTION_DISTANCE_THRESHOLD = 150  # XXX: parameter
MEASUREMENT_DISTANCE_THRESHOLD = 50  # XXX: parameter

METRIC_PRECISION = "Precision"
METRIC_RECALL = "Recall"
METRIC_F1 = "f1"
METRIC_GENERALITY = "Generality"
METRIC_SPATIAL_ERROR = "Spatial Error"
METRIC_IOU = "IoU"


class PresenceData:
    "Holds Presence related values for a class"

    def __init__(self) -> None:
        self.tp: int = 0
        self.fp: int = 0
        self.fn: int = 0
        self.tn: int = 0

    def precision(self) -> float:
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return 1

    def recall(self) -> float:
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return 1

    def f1(self) -> float:
        try:
            return (
                2
                * self.precision()
                * self.recall()
                / (self.precision() + self.recall())
            )
        except ZeroDivisionError:
            return 0

    def generality(self) -> float:
        "images with this class annotated / total images"
        return (self.tp + self.fn) / (self.tp + self.fn + self.tn + self.fp)


class DetectionData:
    "Holds Detection related values for a class"

    def __init__(self) -> None:
        self.tp: int = 0
        self.fp: int = 0
        self.fn: int = 0
        self.se_sum: float = 0  # sum of spactial errors for each tp

    def precision(self) -> float:
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return 1

    def recall(self) -> float:
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return 1

    def f1(self) -> float:
        try:
            return (
                2
                * self.precision()
                * self.recall()
                / (self.precision() + self.recall())
            )
        except ZeroDivisionError:
            return 0

    def generality(self, total_annotations: int) -> float:
        "annotations of this class / total annotations"
        return (self.tp + self.fn) / total_annotations

    def spatial_error(self) -> float:
        try:
            return self.se_sum / self.tp
        except ZeroDivisionError:
            return 0


class MeasurementData:
    "Holds Measurement related values for a class"

    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.count = 0
        self.annotatedPixels = 0

    def precision(self) -> float:
        try:
            return self.sums[METRIC_PRECISION] / self.count
        except ZeroDivisionError:
            return 1

    def recall(self) -> float:
        try:
            return self.sums[METRIC_RECALL] / self.count
        except ZeroDivisionError:
            return 1

    def f1(self) -> float:
        try:
            return self.sums[METRIC_F1] / self.count
        except ZeroDivisionError:
            return 0

    def generality(self, total_pixels: int) -> float:
        "total annotated pixels of this class / total pixels"
        return self.annotatedPixels / total_pixels

    def spatial_error(self) -> float:
        try:
            return self.sums[METRIC_SPATIAL_ERROR] / self.count
        except ZeroDivisionError:
            return 0

    def iou(self) -> float:
        try:
            return self.sums[METRIC_IOU] / self.count
        except ZeroDivisionError:
            return 0


class PDM_Evaluator(DatasetEvaluator):
    def __init__(self, datasetName: str, classes: List[int]):
        self.classes: List[int] = classes
        self.datasetName: str = datasetName
        self.dataset: List[Dict] = DatasetCatalog.get(datasetName)
        self.totalAnnotations = sum(len(image["annotations"]) for image in self.dataset)
        self.totalPixels = sum(
            image["height"] * image["width"] for image in self.dataset
        )
        self.reset()

    def reset(self):
        self.totalAnnotations = 0
        self.presences: Dict[int, PresenceData] = {}
        self.detections: Dict[int, DetectionData] = {}
        self.measurements: Dict[int, MeasurementData] = {}
        for c in self.classes:
            self.presences[c] = PresenceData()
            self.detections[c] = DetectionData()
            self.measurements[c] = MeasurementData()

    def process(self, inputs, outputs):
        instances = outputs[0]["instances"]
        filename = inputs[0]["file_name"]
        all_annotations = next(
            image for image in self.dataset if image["file_name"] == filename
        )["annotations"]
        for c in self.classes:
            predictions = instances[instances.pred_classes == c]
            annotations = [
                image for image in all_annotations if image["category_id"] == c
            ]

            # Presence
            self.process_presence(c, annotations, predictions)

            # Detection
            triplets = self.process_detection(c, annotations, predictions)
            # NOTE: since we already matched pairs and calculated distances in Detection, we can pass them on as triplets to Measurement

            # Measurement
            self.process_measurement(c, annotations, predictions, triplets)

        if not len(outputs + inputs) == 2:
            print(
                f"WARNING: more than one input ({len(inputs)}) and output ({len(outputs)}) given to evaluator"
            )

    def process_presence(self, c, annotations, predictions):
        if len(annotations) == 0 and len(predictions) == 0:
            self.presences[c].tn += 1
        elif len(annotations) > 0 and len(predictions) > 0:
            self.presences[c].tp += 1
        elif len(annotations) == 0 and len(predictions) > 0:
            self.presences[c].fp += 1
        elif len(annotations) > 0 and len(predictions) == 0:
            self.presences[c].fn += 1

    def process_detection(self, c, annotations, predictions):
        pairs = self.match_pairs(annotations, predictions)
        if len(pairs) == 0:
            return []  # don't score detection if Na == Np == 0
        tp_triplets = []
        # NOTE: each pair is (a_index, p_index)
        for pair in pairs:
            if pair[0] == -1:
                self.detections[c].fp += 1
                continue
            if pair[1] == -1:
                self.detections[c].fn += 1
                continue
            distance = self.detection_distance(
                annotations[pair[0]],
                predictions.pred_boxes[pair[1]].tensor[0],
            )
            if distance < DETECTION_DISTANCE_THRESHOLD:
                self.detections[c].tp += 1
                self.detections[c].se_sum += distance / min(
                    annotations[pair[0]]["bbox"][2:]
                )
                tp_triplets.append((pair[0], pair[1], distance))
            else:
                self.detections[c].fp += 1
                self.detections[c].fn += 1
        return tp_triplets

    def process_measurement(self, c, annotations, predictions, triplets):
        if len(triplets) == 0:
            return  # don't score measurement if none were TP
        # note pair is actually a triplet now, including the distances we calculated during the Detection stage
        for triplet in triplets:
            if triplet[2] < MEASUREMENT_DISTANCE_THRESHOLD:
                a_mask = self.convert_polygon(
                    annotations[triplet[0]]["segmentation"][0]
                )
                p_mask = predictions.pred_masks[triplet[1]].cpu().numpy()

                intersection = np.sum(np.logical_and(a_mask, p_mask))
                union = np.sum(np.logical_or(a_mask, p_mask))

                precision = intersection / np.sum(p_mask)
                recall = intersection / np.sum(a_mask)
                try:
                    f1 = 2 * precision * recall / (precision + recall)
                except ZeroDivisionError:
                    f1 = 0
                se = triplet[2] / min(annotations[triplet[0]]["bbox"][2:])
                iou = intersection / union
                self.measurements[c].sums[METRIC_PRECISION] += precision
                self.measurements[c].sums[METRIC_RECALL] += recall
                self.measurements[c].sums[METRIC_F1] += f1
                self.measurements[c].sums[METRIC_SPATIAL_ERROR] += se
                self.measurements[c].sums[METRIC_IOU] += iou
                self.measurements[c].count += 1
                self.measurements[c].annotatedPixels += sum(a_mask)

    def evaluate(self):
        Presence = {}
        Detection = {}
        Measurement = {}
        classNames = MetadataCatalog.get(self.datasetName).thing_classes
        for c in self.classes:
            # Presence
            Presence[f"{classNames[c]}: {METRIC_PRECISION}"] = self.presences[
                c
            ].precision()
            Presence[f"{classNames[c]}: {METRIC_RECALL}"] = self.presences[c].recall()
            Presence[f"{classNames[c]}: {METRIC_F1}"] = self.presences[c].f1()
            Presence[f"{classNames[c]}: {METRIC_GENERALITY}"] = self.presences[
                c
            ].generality()

            # Detection
            Detection[f"{classNames[c]}: {METRIC_PRECISION}"] = self.detections[
                c
            ].precision()
            Detection[f"{classNames[c]}: {METRIC_RECALL}"] = self.detections[c].recall()
            Detection[f"{classNames[c]}: {METRIC_F1}"] = self.detections[c].f1()
            Detection[f"{classNames[c]}: {METRIC_GENERALITY}"] = self.detections[
                c
            ].generality(self.totalAnnotations)
            Detection[f"{classNames[c]}: {METRIC_SPATIAL_ERROR}"] = self.detections[
                c
            ].spatial_error()

            # Measurement
            Measurement[f"{classNames[c]}: {METRIC_PRECISION}"] = self.measurements[
                c
            ].precision()
            Measurement[f"{classNames[c]}: {METRIC_RECALL}"] = self.measurements[
                c
            ].recall()
            Measurement[f"{classNames[c]}: {METRIC_F1}"] = self.measurements[c].f1()
            Measurement[f"{classNames[c]}: {METRIC_GENERALITY}"] = self.measurements[
                c
            ].generality(self.totalAnnotations)
            Measurement[f"{classNames[c]}: {METRIC_SPATIAL_ERROR}"] = self.measurements[
                c
            ].spatial_error()
            Measurement[f"{classNames[c]}: {METRIC_IOU}"] = self.measurements[c].iou(
                self.totalPixels
            )

        for metric in [METRIC_PRECISION, METRIC_RECALL, METRIC_F1, METRIC_GENERALITY]:
            Presence[f"Average: {metric}"] = np.mean(
                [Presence[f"{classNames[c]}: {metric}"] for c in self.classes]
            )
        for metric in [
            METRIC_PRECISION,
            METRIC_RECALL,
            METRIC_F1,
            METRIC_GENERALITY,
            METRIC_SPATIAL_ERROR,
        ]:
            Detection[f"Average: {metric}"] = np.mean(
                [Detection[f"{classNames[c]}: {metric}"] for c in self.classes]
            )
        for metric in [
            METRIC_PRECISION,
            METRIC_RECALL,
            METRIC_F1,
            METRIC_GENERALITY,
            METRIC_SPATIAL_ERROR,
            METRIC_IOU,
        ]:
            Measurement[f"Average: {metric}"] = np.mean(
                [Measurement[f"{classNames[c]}: {metric}"] for c in self.classes]
            )
        result = {
            "PDM: Presence": Presence,
            "PDM: Detection": Detection,
            "PDM: Measurement": Measurement,
        }
        return OrderedDict(result)

    def print_result(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            print(func.__name__, result)
            return result

        return wrapper

    @staticmethod
    def convert_polygon(polygon):
        """ converts [x1,y2,x2,y2,...] to boolean mask """
        width = 1920
        height = 1080
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x, y = x.flatten(), y.flatten()

        points = np.vstack((x, y)).T
        if isinstance(polygon, list):
            polygon = np.array(polygon)
        path = Path(polygon.reshape(-1, 2))
        grid = path.contains_points(points)
        grid = grid.reshape((height, width))

        return grid

    @staticmethod
    def distance(a, p):
        return np.sqrt((a[0] - p[0]) ** 2 + (a[1] - p[1]) ** 2)

    @classmethod
    # @print_result
    def detection_distance(cls, a, pb):
        a_centre = (
            a["bbox"][0] + a["bbox"][2] / 2,
            a["bbox"][1] + a["bbox"][3] / 2,
        )
        p_centre = (
            (pb[0].item() + pb[2].item()) / 2,
            (pb[1].item() + pb[3].item()) / 2,
        )
        return cls.distance(a_centre, p_centre)

    @classmethod
    # @print_result
    def match_pairs(cls, annotations, predictions):
        """Match pairs by allocating closest prediction to each annotation"""
        Na = len(annotations)
        Np = len(predictions)
        if Na == Np == 0:
            return []
        if Na == Np == 1:
            return [(0, 0)]  # only one permutation
        a_centres = []
        p_centres = []
        # Annotations are [x1,y1,w,h]
        for a in annotations:
            a_centres.append(
                (
                    a["bbox"][0] + a["bbox"][2] / 2,
                    a["bbox"][1] + a["bbox"][3] / 2,
                )
            )
        # Predictions are [x1,y1,x2,y2]
        for pb in predictions.pred_boxes:
            p_centres.append(
                (
                    (pb[0].item() + pb[2].item()) / 2,
                    (pb[1].item() + pb[3].item()) / 2,
                )
            )
        # infinite points for use with the extra items
        a_centres.append((np.inf, np.inf))
        p_centres.append((np.inf, np.inf))
        # pad extra items pointing to the infinite point
        pairs = list(itertools.zip_longest(range(Na), range(Np), fillvalue=-1))
        # print(a_centres, p_centres, pairs, sep="\n")
        swaps = 0
        """
        pairs[i] pairs[j] - the indices of the pairs to compare
        pairs[i][0] pairs[i][1] - a and p indicies of the ith pair
        a_centres[pairs[i][0]], p_centres[pairs[i][1]] - the a and p centres of the first pair
        a_centres[pairs[j][0]], p_centres[pairs[j][1]] - the a and p centres of the second pair
        """
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                # Original distances
                AiPi = cls.distance(a_centres[pairs[i][0]], p_centres[pairs[i][1]])
                AjPj = cls.distance(a_centres[pairs[j][0]], p_centres[pairs[j][1]])

                # Swapped distances
                AiPj = cls.distance(a_centres[pairs[i][0]], p_centres[pairs[j][1]])
                AjPi = cls.distance(a_centres[pairs[j][0]], p_centres[pairs[i][1]])

                # improve minimum
                if min(AiPj, AjPi) < min(AiPi, AjPj):
                    pairs[i], pairs[j] = (pairs[i][0], pairs[j][1]), (
                        pairs[j][0],
                        pairs[i][1],
                    )
                    swaps += 1
                elif min(AiPj, AjPi) == min(AiPi, AjPj):
                    # improve maximum if minimum conserved
                    if max(AiPj, AjPi) < max(AiPi, AjPj):
                        pairs[i], pairs[j] = (pairs[i][0], pairs[j][1]), (
                            pairs[j][0],
                            pairs[i][1],
                        )
                        swaps += 1
        # print(pairs)
        # print(swaps)
        return pairs
