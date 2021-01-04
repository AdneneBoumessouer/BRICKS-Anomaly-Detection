import numpy as np
from processing.anomaly import AnomalyMap
from processing.utils import printProgressBar
from skimage import measure, morphology


class HighContrastAnomalyDetector:
    def __init__(self):
        pass

    def estimate_threshold(self, resmaps_val, min_area, vstep=1e-3, verbose=1):
        self.min_area = min_area

        # initialize thresholds
        vmin = np.amin(resmaps_val) + vstep
        vmax = np.amax(resmaps_val) + vstep
        ths = np.arange(vmin, vmax, vstep, dtype="float")

        # loop over thresholds
        printProgressBar(0, len(ths), length=80, verbose=verbose)
        for i, th in enumerate(ths):
            imgs_binary = resmaps_val > th
            imgs_binary = np.array(
                [
                    morphology.binary_opening(binary, selem=morphology.square(3))
                    for binary in imgs_binary
                ]
            )
            imgs_labeled = np.array(
                [measure.label(binary, connectivity=1) for binary in imgs_binary]
            )
            areas = np.array(
                [
                    regionprop.area
                    for labeled in imgs_labeled
                    for regionprop in measure.regionprops(labeled)
                ]
            )
            if areas.size > 0:
                largest_area = np.amax(areas)
                if largest_area < min_area:
                    self.threshold = th
                    break
            else:
                self.threshold = th
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
        imgs_binary = np.array(
            [
                morphology.binary_opening(binary, selem=morphology.square(3))
                for binary in imgs_binary
            ]
        )
        # label resmaps (extract connected componenets)
        imgs_labeled = [measure.label(binary, connectivity=1) for binary in imgs_binary]
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
                anomaly_maps.append(anomaly_map)
            else:
                anomaly_maps.append(None)
        return predictions, anomaly_maps


class LowContrastAnomalyDetector:
    def __init__(self, vmin=0.05):
        self.vmin = vmin

    def estimate_area(self, resmaps_val, threshold):
        self.threshold = threshold
        # threshold resmaps
        imgs_binary = (self.vmin <= resmaps_val) & (resmaps_val <= self.threshold)
        imgs_binary = np.array(
            [
                morphology.binary_opening(binary, selem=morphology.square(5))  # TODO 3
                for binary in imgs_binary
            ]
        )
        # label resmaps (extract connected componenets)
        imgs_labeled = [measure.label(binary, connectivity=1) for binary in imgs_binary]
        largest_areas = [
            np.amax(measure.regionprops_table(labeled, properties=["area"])["area"])
            for labeled in imgs_labeled
            if measure.regionprops(labeled)
        ]
        # largest_areas = compute_largest_areas(binary)
        self.min_area = int(1.15 * round(np.percentile(largest_areas, 95)))
        return self.min_area

    def set_min_area(self, min_area):
        self.min_area = min_area

    def set_threshold(self, threshold):
        self.threshold = threshold

    def predict(self, resmaps_test):
        anomaly_maps = []
        predictions = []
        # threshold resmaps
        imgs_binary = (self.vmin < resmaps_test) & (resmaps_test < self.threshold)
        imgs_binary = np.array(
            [
                morphology.binary_opening(binary, selem=morphology.square(5))
                for binary in imgs_binary
            ]
        )
        # label resmaps (extract connected componenets)
        imgs_labeled = [measure.label(binary, connectivity=1) for binary in imgs_binary]
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
                anomaly_maps.append(anomaly_map)
            else:
                anomaly_maps.append(None)
        return predictions, anomaly_maps

