import numpy as np
import sklearn.metrics

class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    # def _fast_hist(self, label_true, label_pred, n_class):
    #     mask = (label_true >= 0) & (label_true < n_class)
    #     hist = np.bincount(
    #         n_class * label_true[mask].astype(int) +
    #         label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    #     return hist

    def update(self, label_trues, label_preds):
        # for lt, lp in zip(label_trues, label_preds):
            # self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
        self.confusion_matrix += sklearn.metrics.confusion_matrix(label_trues.flatten(),
                                                                  label_preds.flatten(),
                                                                  labels=list(range(self.n_classes)))


    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwIU
        """
        hist = self.confusion_matrix
        pa = np.diag(hist).sum() / hist.sum()
        ca = np.diag(hist) / hist.sum(axis=1)
        mca = np.nanmean(ca)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mciu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum() # fraction of the pixels that come from each class
        fwiu = (freq[freq > 0] * iu[freq > 0]).sum()

        return {'Pixel Acc: ': pa,
                'Class Accuracy: ': ca,
                'Mean Class Acc: ': mca,
                'Mean IoU: ': mciu,
                'Freq Weighted IoU: ': fwiu,
                'confusion_matrix': self.confusion_matrix}, iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
