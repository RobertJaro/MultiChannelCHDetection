import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from chronnos.data.convert import sdo_cmaps
from chronnos.train.metric import iou, accuracy


class PlotCallback():

    def __init__(self, data, model, path, prefix, cmaps=None):
        super().__init__()
        self.data = data
        self.path = path
        self.model = model
        self.prefix = prefix
        self.cmaps = sdo_cmaps if cmaps is None else cmaps

    def call(self, epoch, alpha=None):
        maps, y, y_pred = self._loadData(alpha)

        img_matrix = [[("", o[j], self.cmaps[j]) for j in range(o.shape[0])] +
                      [("GT", s[i], "gray") for i in range(s.shape[0])] +
                      [("Prediction", p[j], "gray") for j in range(p.shape[0])]
                      for o, s, p in zip(maps, y, y_pred)]

        n_rows = len(img_matrix)
        n_columns = len(img_matrix[0])
        f, axarr = plt.subplots(n_rows, n_columns, figsize=(3 * n_columns, 3 * n_rows))
        for i in range(n_rows):
            for j in range(n_columns):
                ax = axarr[i, j]
                ax.axis("off")
                ax.set_title(img_matrix[i][j][0])
                image = img_matrix[i][j][1]
                image = image if len(image.shape) == 2 or image.shape[-1] == 3 else image[..., 0]
                cmap = img_matrix[i][j][2]
                ax.imshow(image, cmap=cmap, vmin=0, vmax=1)

        plt.tight_layout()
        name = "%s_iteration%03d" % (self.prefix, epoch + 1)
        path = os.path.join(self.path, "%s.jpg" % name)
        plt.savefig(path, dpi=100)
        plt.close("all")

    def _loadData(self, alpha):
        x, y = zip(*self.data)
        if self.model.fade:
            y_pred = self.model.forwardFadeIn(torch.tensor(x).cuda(), alpha)
        else:
            y_pred = self.model.forward(torch.tensor(x).cuda())
        y_pred = y_pred.detach().cpu().numpy()
        x = (np.array(x) + 1) / 2
        return x, y, y_pred

class ValidationCallback():

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.history = {'iou': [], 'acc': []}

    def call(self, model, valid_loader_map, valid_loader_mask, alpha=None):
        iou_res = []
        accuracy_res = []
        for x, y, in zip(valid_loader_map, valid_loader_mask):
            if model.fade:
                y_pred = model.forwardFadeIn(x.cuda(), alpha)
            else:
                y_pred = model.forward(x.cuda())
            y_pred = y_pred.cpu()
            y_pred, y = y_pred >= 0.5, y >= 0.5

            iou_res.extend(iou(y_pred, y))
            accuracy_res.extend(accuracy(y_pred, y))
        self.history['iou'].append(np.mean(iou_res))
        self.history['acc'].append(np.mean(accuracy_res))

        self._plot()
        return np.mean(iou_res), np.mean(accuracy_res)

    def _plot(self):
        plt.figure(figsize=(16, 10))
        plt.subplot(211)
        plt.ylabel('IoU')
        plt.xlabel('Epoch')
        epochs = range(len(self.history['iou']))
        plt.plot(epochs, self.history['iou'])
        plt.subplot(212)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.plot(epochs, self.history['acc'])
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, 'validation_history.jpg'), dpi=100)
        plt.close()

