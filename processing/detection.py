import numpy as np
from processing.utils import get_indices
from processing.resmaps import ResmapCalculator
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
                self.th = th
                break
            printProgressBar(i + 1, len(ths), length=80, verbose=verbose)

        printProgressBar(len(ths), len(ths), length=80, verbose=verbose)
        return self.th

    def predict(self, resmaps_test):
        defects = []
        predictions = []
        # threshold resmaps
        imgs_binary = resmaps_test > self.th
        # label resmaps (extract connected componenets)
        imgs_labeled = [measure.label(binary) for binary in imgs_binary]  # np.array()
        # loop over labeled images
        for labeled in imgs_labeled:
            # initialize to defect-free
            is_defective = 0
            props = []
            # loop over labeled regions
            for regionprop in measure.regionprops(labeled):
                if regionprop.area > self.min_area:
                    is_defective = 1
                    props.append((regionprop.area, regionprop.bbox))
            # append prediction
            predictions.append(is_defective)
            # append defective labeled image and its properties
            if is_defective:
                defects.append((labeled, props))
            else:
                defects.append(None)
        return predictions, defects


class LowContrastAnomalyDetector:
    def __init__(self, vmin=0.05, vmax=0.18, vstep=2e-3):
        self.vmin = vmin
        self.vmax = vmax
        self.vstep = vstep

    def fit(self, resmaps_val):
        binary = (self.vmin < resmaps_val) & (resmaps_val < self.vmax)
        largest_areas = compute_largest_areas(binary)
        self.th_area = int(round(np.percentile(largest_areas, 95)))
        return self.th_area

    def predict(self, resmaps_test, th_area):
        binary = (self.vmin < resmaps_test) & (resmaps_test < self.vmax)
        largest_areas = compute_largest_areas(binary)


def compute_largest_areas(imgs_binary):
    imgs_labeled = np.array([measure.label(binary) for binary in imgs_binary])
    largest_areas = [
        np.amax(measure.regionprops_table(labeled, properties=["area"])["area"])
        for labeled in imgs_labeled
    ]
    return np.array(largest_areas)


# def fetch_resmaps(generator, model, method, index_arr):
#     imgs_input = generator._get_batches_of_transformed_samples(index_arr)[0]

#     # Get reconstructions
#     imgs_pred = model.predict(imgs_input)

#     filenames = [generator.filenames[i] for i in index_arr]

#     # Get resmaps
#     RC = ResmapCalculator(
#         imgs_input=imgs_input,
#         imgs_pred=imgs_pred,
#         color_out="grayscale",
#         method=method,
#         filenames=filenames,
#         vmin=vmin,
#         vmax=vmax,
#         dtype="float64",
#     )
#     resmaps = RC.get_resmaps()
#     return resmaps

