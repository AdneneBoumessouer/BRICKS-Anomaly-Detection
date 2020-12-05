import numpy as np
from processing.utils import get_indices
from processing.resmaps import ResmapCalculator
from processing.anomaly import AnomalyMap
from processing.utils import printProgressBar
from skimage import measure


class AnomalyDetector:
    pass


class HighContrastAnomalyDetector:
    def __init__(self, vmin=0.18, vmax=1.0, vstep=2e-3):
        self.vmin = vmin
        self.vmax = vmax
        self.vstep = vstep

    def fit(self, resmaps_val, min_area, verbose=1):
        self.min_area = min_area
        ths = np.arange(self.vmin, self.vmax, self.vstep, dtype="float")
        printProgressBar(0, len(ths), length=80, verbose=verbose)
        for i, th in enumerate(ths):
            imgs_binary = resmaps_val > th
            imgs_labeled = np.array([measure.label(binary) for binary in imgs_binary])
            areas = np.array(
                [
                    regionprop.area
                    for labeled in imgs_labeled
                    for regionprop in measure.regionprops(labeled)
                ]
            )
            largest_area = np.amax(areas)
            if largest_area < min_area:
                self.threshold = th
                break
            printProgressBar(i + 1, len(ths), length=80, verbose=verbose)

        printProgressBar(len(ths), len(ths), length=80, verbose=verbose)
        return self.threshold

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_min_area(self, min_area):
        self.min_area = min_area

    def predict(self, resmaps_test):
        anomaly_maps = []
        predictions = []
        # threshold resmaps
        imgs_binary = resmaps_test > self.threshold
        # label resmaps (extract connected componenets)
        imgs_labeled = [measure.label(binary) for binary in imgs_binary]
        # loop over labeled images
        for labeled in imgs_labeled:
            # initialize to defect-free
            is_defective = 0
            props = []
            # loop over labeled regions
            for regionprop in measure.regionprops(labeled):
                if regionprop.area > self.min_area:
                    is_defective = 1
                    props.append(regionprop)
            # append prediction
            predictions.append(is_defective)
            # append defective labeled image and its properties
            if is_defective:
                anomaly_map = AnomalyMap(labeled, regionprops=props)
                anomaly_map.remove_unsued_labels_from_labeled()
                anomaly_maps.append(anomaly_map)
            else:
                anomaly_maps.append(None)
        return predictions, anomaly_maps


class LowContrastAnomalyDetector:
    def __init__(self, vmin=0.05, vmax=0.18, vstep=2e-3):
        self.vmin = vmin
        self.vmax = vmax
        self.vstep = vstep

    def fit(self, resmaps_val):
        # threshold resmaps
        imgs_binary = (self.vmin < resmaps_val) & (resmaps_val < self.vmax)
        # label resmaps (extract connected componenets)
        imgs_labeled = [measure.label(binary) for binary in imgs_binary]
        largest_areas = [
            np.amax(measure.regionprops_table(labeled, properties=["area"])["area"])
            for labeled in imgs_labeled
        ]
        # largest_areas = compute_largest_areas(binary)
        self.min_area = int(round(np.percentile(largest_areas, 95)))
        return self.min_area

    def set_min_area(self, min_area):
        self.min_area = min_area

    def predict(self, resmaps_test):
        anomaly_maps = []
        predictions = []
        # threshold resmaps
        imgs_binary = (self.vmin < resmaps_test) & (resmaps_test < self.vmax)
        # label resmaps (extract connected componenets)
        imgs_labeled = [measure.label(binary) for binary in imgs_binary]
        # loop over labeled images
        for labeled in imgs_labeled:
            # initialize to defect-free
            is_defective = 0
            props = []
            # loop over labeled regions
            for regionprop in measure.regionprops(labeled):
                if regionprop.area > self.min_area:
                    is_defective = 1
                    props.append(regionprop)
            # append prediction
            predictions.append(is_defective)
            # append defective labeled image and its properties
            if is_defective:
                anomaly_map = AnomalyMap(labeled, regionprops=props)
                anomaly_map.remove_unsued_labels_from_labeled()
                anomaly_maps.append(anomaly_map)
            else:
                anomaly_maps.append(None)
        return predictions, anomaly_maps

