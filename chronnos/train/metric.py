import numpy as np
from astropy.nddata import block_reduce


def iou(outputs, labels):
    filter = np.zeros((1, 512, 512))
    filter[:, :, 160:351] = 1
    f = block_reduce(filter, (1, filter.shape[1] // labels.shape[2], filter.shape[2] // labels.shape[3]),
                     np.mean) >= 0.5
    f = np.repeat(np.expand_dims(f, 0), labels.shape[0], 0)
    f = labels.new(f)  # use same device

    outputs = outputs.int()
    labels = labels.int()

    intersection = (outputs & labels).float() * f
    intersection = intersection.sum((1, 2, 3))
    union = (outputs | labels).float() * f
    union = union.sum((1, 2, 3))

    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou


def accuracy(outputs, labels):
    filter = np.zeros((1, 512, 512))
    filter[:, :, 160:351] = 1
    f = block_reduce(filter, (1, filter.shape[1] // labels.shape[2], filter.shape[2] // labels.shape[3]),
                     np.mean) >= 0.5
    f = np.repeat(np.expand_dims(f, 0), labels.shape[0], 0)
    f = labels.new(f)  # use same device

    correct = (labels.eq(outputs) * f).float()
    return correct.sum((1, 2, 3)) / f.sum((1, 2, 3))

