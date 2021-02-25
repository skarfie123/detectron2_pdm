from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import DatasetEvaluator
import numpy as np
from collections import OrderedDict


class PDM_BBOX(DatasetEvaluator):
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
            print("WARNING", len(inputs), len(outputs))

    def evaluate(self):
        result = {}
        average_presences = 0
        average_detections = 0
        average_measurements = 0
        classNames = MetadataCatalog.get(self.datasetName).thing_classes
        for c in self.classes:
            try:
                result[f"AP: {classNames[c]}"] = sum(self.presences[c]) / len(
                    self.presences[c]
                )
                average_presences += result[f"AP: {classNames[c]}"]
            except ZeroDivisionError:
                result[f"AP: {classNames[c]}"] = "N/A"
            try:
                result[f"AD: {classNames[c]}"] = sum(self.detections[c]) / len(
                    self.detections[c]
                )
                average_detections += result[f"AD: {classNames[c]}"]
            except ZeroDivisionError:
                result[f"AD: {classNames[c]}"] = "N/A"
            try:
                result[f"AM: {classNames[c]}"] = sum(self.measurements[c]) / len(
                    self.measurements[c]
                )
                average_measurements += result[f"AM: {classNames[c]}"]
            except ZeroDivisionError:
                result[f"AM: {classNames[c]}"] = "N/A"
        average_presences = average_presences / len(self.classes)
        average_detections = average_detections / len(self.classes)
        average_measurements = average_measurements / len(self.classes)
        result["AP"] = average_presences
        result["AD"] = average_detections
        result["AM"] = average_measurements
        return OrderedDict({"PDM_BBOX": result})

    def print_result(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            print(func.__name__, result)
            return result

        return wrapper

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

    @staticmethod
    def distance(a, p):
        return np.sqrt((a[0] - p[0]) ** 2 + (a[1] - p[1]) ** 2)

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

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
