import itertools
from collections import OrderedDict

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from matplotlib.path import Path


class PDM_Evaluator(DatasetEvaluator):
    def __init__(self, datasetName, classes):
        self.classes = classes
        self.datasetName = datasetName
        self.dataset = DatasetCatalog.get(datasetName)
        self.reset()

    def reset(self):
        self.presences = {}
        self.detections = {}
        self.measurements = {}
        for c in self.classes:
            self.presences[c] = []
            self.detections[c] = []
            self.measurements[c] = []

    def process(self, inputs, outputs):
        instances = outputs[0]["instances"]
        filename = inputs[0]["file_name"]
        all_annotations = next(
            item for item in self.dataset if item["file_name"] == filename
        )["annotations"]
        for c in self.classes:
            predictions = instances[instances.pred_classes == c]
            annotations = [item for item in all_annotations if item["category_id"] == c]
            # Presence
            if (len(annotations) == 0 and len(predictions) == 0) or (
                len(annotations) > 0 and len(predictions) > 0
            ):
                self.presences[c].append(1)
            else:
                self.presences[c].append(0)
            # Detection
            pairs = self.match_pairs(annotations, predictions)
            if len(pairs) == 0:
                continue  # don't score detection if Na == Np == 0
            pairs2 = []
            tp = 0
            fp = 0
            fn = 0
            distances = []
            DETECTION_DISTANCE_THRESHOLD = 150  # XXX: parameter
            """ each pair is (a_index, p_index) """
            for pair in pairs:
                if pair[0] == -1:
                    fp += 1
                    continue
                if pair[1] == -1:
                    fn += 1
                    continue
                distance = self.detection_distance(
                    annotations[pair[0]], predictions.pred_boxes[pair[1]].tensor[0]
                )
                if distance < DETECTION_DISTANCE_THRESHOLD:
                    tp += 1
                    distances.append(distance)  # TODO append for tp or all?
                    pairs2.append(
                        (pair[0], pair[1], distance)
                    )  ## use only these for measurement
                else:
                    fp += 1
                    fn += 1
            try:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * precision * recall / (precision + recall)
                se = sum(distances) / len(distances)
            except ZeroDivisionError:
                precision = 0
                recall = 0
                f1 = 0
                se = 0
            self.detections[c].append(
                {
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "Spatial Error": se,
                }
            )
            # Measurement
            if len(pairs2) == 0:
                continue  # don't score measurement if none were TP
            tp = 0  # now count TP with stricter requirements
            distances = []
            IoUs = []
            MEASUREMENT_DISTANCE_THRESHOLD = 50  # XXX: parameter
            for pair in pairs2:
                if pair[2] < MEASUREMENT_DISTANCE_THRESHOLD:
                    tp += 1
                    distances.append(pair[2])  # TODO append for tp or all?
                    a_mask = self.convert_polygon(
                        annotations[pair[0]]["segmentation"][0]
                    )
                    p_mask = predictions.pred_masks[pair[1]]
                    intersection = np.sum(np.logical_and(a_mask, p_mask))
                    union = np.sum(np.logical_or(a_mask, p_mask))
                    IoUs.append(intersection / union)
                else:
                    fp += 1
                    fn += 1
            try:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * precision * recall / (precision + recall)
                se = sum(distances) / len(distances)
                iou = sum(IoUs) / len(IoUs)
            except ZeroDivisionError:
                precision = 0
                recall = 0
                f1 = 0
                se = 0
                iou = 0
            self.measurements[c].append(
                {
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "Spatial Error": se,
                    "IoU": iou,
                }
            )
        if not len(outputs + inputs) == 2:
            print(
                f"WARNING: more than one input ({len(inputs)}) and output ({len(outputs)}) given to evaluator"
            )

    def evaluate(self):
        Presence = {}
        Detection = {}
        Measurement = {}
        sum_presences = 0
        sum_detections = {}
        sum_measurements = {}
        classNames = MetadataCatalog.get(self.datasetName).thing_classes
        for c in self.classes:
            try:
                Presence[classNames[c]] = sum(self.presences[c]) / len(
                    self.presences[c]
                )
                sum_presences += Presence[classNames[c]]
            except ZeroDivisionError:
                Presence[classNames[c]] = 0
            for metric in self.detections[c][0]:
                try:
                    Detection[f"{classNames[c]}: {metric}"] = sum(
                        d[metric] for d in self.detections[c]
                    ) / len(self.detections[c])
                    sum_detections[metric] += Detection[f"{classNames[c]}: {metric}"]
                except ZeroDivisionError:
                    Detection[f"{classNames[c]}: {metric}"] = 0
            for metric in self.measurements[c][0]:
                try:
                    Measurement[f"{classNames[c]}: {metric}"] = sum(
                        m[metric] for m in self.measurements[c]
                    ) / len(self.measurements[c])
                    sum_measurements[metric] += Measurement[
                        f"{classNames[c]}: {metric}"
                    ]
                except ZeroDivisionError:
                    Measurement[f"{classNames[c]}: {metric}"] = 0
        Presence["Average"] = sum_presences / len(self.classes)
        for metric in self.detections[c][0]:
            Detection[f"Average: {metric}"] = sum_detections[metric] / len(self.classes)
        for metric in self.measurements[c][0]:
            Measurement[f"Average: {metric}"] = sum_measurements[metric] / len(
                self.classes
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
        a_centre = (a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2)
        p_centre = (
            (pb[0].item() + pb[2].item()) / 2,
            (pb[1].item() + pb[3].item()) / 2,
        )
        return cls.distance(a_centre, p_centre)

    @classmethod
    @print_result
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
        for a in annotations:
            a_centres.append(
                (a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2)
            )
        for pb in predictions.pred_boxes:
            p_centres.append(
                ((pb[0].item() + pb[2].item()) / 2, (pb[1].item() + pb[3].item()) / 2)
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
                    pairs[i][1], pairs[j][1] = pairs[j][1], pairs[i][1]
                    swaps += 1
                elif min(AiPj, AjPi) == min(AiPi, AjPj):
                    # improve maximum if minimum conserved
                    if max(AiPj, AjPi) < max(AiPi, AjPj):
                        pairs[i][1], pairs[j][1] = pairs[j][1], pairs[i][1]
                        swaps += 1
        # print(pairs)
        # print(swaps)
        return pairs


############################################################################################################


class PDM_Evaluator_Old(DatasetEvaluator):
    def __init__(self, datasetName, classes):
        self.classes = classes
        self.datasetName = datasetName
        self.dataset = DatasetCatalog.get(datasetName)
        self.reset()

    def reset(self):
        self.presences = {}
        self.detections = {}
        self.measurements = {}
        for c in self.classes:
            self.presences[c] = []
            self.detections[c] = []
            self.measurements[c] = []

    def process(self, inputs, outputs):
        instances = outputs[0]["instances"]
        filename = inputs[0]["file_name"]
        all_annotations = next(
            item for item in self.dataset if item["file_name"] == filename
        )["annotations"]
        for c in self.classes:
            predictions = instances[instances.pred_classes == c]
            annotations = [item for item in all_annotations if item["category_id"] == c]
            if (len(annotations) == 0 and len(predictions) == 0) or (
                len(annotations) > 0 and len(predictions) > 0
            ):
                self.presences[c].append(1)
            else:
                self.presences[c].append(0)
            pairs = self.match_pairs(
                annotations, predictions
            )  # allocate optimal pairs if same number else None
            if pairs:
                self.detections[c].append(
                    np.mean(
                        [
                            self.detection_score(
                                annotations[pair[0]],
                                predictions.pred_boxes[pair[1]].tensor[0],
                            )
                            for pair in pairs
                        ]
                    )
                )
                self.measurements[c].append(
                    np.mean(
                        [
                            self.measurement_score(
                                annotations[pair[0]],
                                predictions.pred_boxes[pair[1]].tensor[0],
                            )
                            for pair in pairs
                        ]
                    )
                )
            elif (
                len(annotations) > 0 and len(predictions) > 0
            ):  # note choosing to penalise detection only for those which were present and non zero
                self.detections[c].append(
                    0
                )  # note to self: detection scores are pretty good when they pair up, but average tends to 0. either most do not pair up, and/or most not present
                # self.measurements[c].append(0) # note choosing to excluding this so that measurement metric only covers correctly detected images
        if not len(outputs + inputs) == 2:
            print(
                f"WARNING: more than one input ({len(inputs)}) and output ({len(outputs)})"
            )

    def evaluate(self):
        result = {}
        average_presences = 0
        average_detections = 0
        average_measurements = 0
        classNames = MetadataCatalog.get(self.datasetName).thing_classes
        for c in self.classes:
            try:
                result[f"P: {classNames[c]}"] = sum(self.presences[c]) / len(
                    self.presences[c]
                )
                average_presences += result[f"P: {classNames[c]}"]
            except ZeroDivisionError:
                result[f"P: {classNames[c]}"] = -1
            try:
                result[f"D: {classNames[c]}"] = sum(self.detections[c]) / len(
                    self.detections[c]
                )
                average_detections += result[f"D: {classNames[c]}"]
            except ZeroDivisionError:
                result[f"D: {classNames[c]}"] = -1
            try:
                result[f"M: {classNames[c]}"] = sum(self.measurements[c]) / len(
                    self.measurements[c]
                )
                average_measurements += result[f"M: {classNames[c]}"]
            except ZeroDivisionError:
                result[f"M: {classNames[c]}"] = -1
        average_presences = average_presences / len(self.classes)
        average_detections = average_detections / len(self.classes)
        average_measurements = average_measurements / len(self.classes)
        result["P"] = average_presences
        result["D"] = average_detections
        result["M"] = average_measurements
        return OrderedDict({"PDM_Old": result})

    def print_result(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            print(func.__name__, result)
            return result

        return wrapper

    @staticmethod
    def distance(a, p):
        return np.sqrt((a[0] - p[0]) ** 2 + (a[1] - p[1]) ** 2)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def distance_score(cls, a_value, p_value, width):
        assert a_value >= 0 and p_value >= 0 and width > 0
        # return np.exp(-((a_value-p_value)/width)**2)
        return 2 * cls.sigmoid(-((4 * (a_value - p_value) / width) ** 2))

    @classmethod
    def size_score(cls, a_size, p_size):
        assert a_size > 0, "zero size annotation"
        if p_size == 0:
            return 0
        # return max(1-abs(a_size-p_size)/a_size, 0)
        return cls.distance_score(a_size, p_size, 2 * a_size)

    @classmethod
    def match_pairs(cls, annotations, predictions):
        n = len(annotations)
        if not len(predictions) == n:
            return  # cannot pair if different
        if n == 0:
            return  # note choose not to score when both zero
        if n == 1:
            return [(0, 0)]  # only one permutation
        a_centres = []
        for a in annotations:
            a_centres.append(
                (a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2)
            )
        p_centres = []
        for pb in predictions.pred_boxes:
            p_centres.append(
                ((pb[0].item() + pb[2].item()) / 2, (pb[1].item() + pb[3].item()) / 2)
            )
        pairs = []
        for i in range(n):
            pairs.append([i, i])
        # print(a_centres, p_centres, pairs, sep="\n")
        swaps = -1
        tswaps = 0
        iter = 0
        while not swaps == 0:
            # print(iter)
            iter += 1
            swaps = 0
            for i in range(n):
                for j in range(n - 1):
                    # pairs[i] pairs[j] - the indices of the pairs to compare
                    # a_centres[pairs[i][0]], p_centres[pairs[i][1]] - the centres of the first pair
                    # a_centres[pairs[j][0]], p_centres[pairs[j][1]] - the centres of the second pair
                    if max(
                        cls.distance(a_centres[pairs[i][0]], p_centres[pairs[j][1]]),
                        cls.distance(a_centres[pairs[j][0]], p_centres[pairs[i][1]]),
                    ) < max(
                        cls.distance(a_centres[pairs[i][0]], p_centres[pairs[i][1]]),
                        cls.distance(a_centres[pairs[j][0]], p_centres[pairs[j][1]]),
                    ):
                        pairs[i][1], pairs[j][1] = (
                            pairs[j][1],
                            pairs[i][1],
                        )  # swap if the max distance can be improved
                        swaps += 1
            if swaps == 0:
                break  # stop if no swaps were made
            tswaps += swaps
        # print(pairs)
        # print(iter, tswaps)
        return pairs

    @classmethod
    # @print_result
    def detection_score(cls, a, pb):
        WIDTH = 150
        a_centre = (a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2)
        p_centre = (
            (pb[0].item() + pb[2].item()) / 2,
            (pb[1].item() + pb[3].item()) / 2,
        )
        return cls.distance_score(a_centre[0], p_centre[0], WIDTH) * cls.distance_score(
            a_centre[1], p_centre[1], WIDTH
        )

    @classmethod
    # @print_result
    def measurement_score(cls, a, pb):
        WIDTH = 25
        a_centre = (a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2)
        p_centre = (
            (pb[0].item() + pb[2].item()) / 2,
            (pb[1].item() + pb[3].item()) / 2,
        )
        return (
            cls.distance_score(a_centre[0], p_centre[0], WIDTH)
            * cls.distance_score(a_centre[1], p_centre[1], WIDTH)
            * cls.size_score(a["bbox"][2], pb[2].item() - pb[0].item())
            * cls.size_score(a["bbox"][3], pb[3].item() - pb[1].item())
        )
