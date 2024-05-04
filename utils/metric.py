import numpy as np


def calc_i_and_u(pred, label, n_classes):
    """
    This function takes the prediction and label of a single image,
    returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
        (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pred = (1 + pred) * (label >= 0)
    label = 1 + label

    # Compute area intersection:
    intersection = pred * (pred == label)
    area_intersection, _ = np.histogram(intersection, bins=n_classes, range=(1, n_classes + 1))

    # Compute area union:
    area_pred, _ = np.histogram(pred, bins=n_classes, range=(1, n_classes + 1))
    area_lab, _ = np.histogram(label, bins=n_classes, range=(1, n_classes + 1))
    area_union = area_pred + area_lab - area_intersection
    return area_intersection, area_union
