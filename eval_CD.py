import numpy as np
from tabulate import tabulate
import os
import cv2
from PIL import Image
from skimage import io, measure
###################       metrics      ###################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""
    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        """获得当前混淆矩阵，并计算当前F1得分，并更新混淆矩阵"""
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict



def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x+1e-6)**-1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1


def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    #
    cls_iou = dict(zip(['iou_'+str(i) for i in range(n_class)], iu))

    cls_precision = dict(zip(['precision_'+str(i) for i in range(n_class)], precision))

    cls_recall = dict(zip(['recall_'+str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_'+str(i) for i in range(n_class)], F1))

    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1':mean_F1}

    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """计算一组预测的混淆矩阵"""
    def __fast_hist(label_gt, label_pred):
        """
                Collect values for Confusion Matrix
                For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
                :param label_gt: <np.array> ground-truth
                :param label_pred: <np.array> prediction
                :return: <np.ndarray> values for confusion matrix
                """
        mask = (label_gt >= 0) & (label_gt < num_classes) # 获取有效标签
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes) # 使用np.bincount函数计算在所有可能的类别对之间真实标签和预测标签的匹配次数。
        return hist
    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix

def print_table(metrics, headers):
    table = tabulate(metrics, headers, tablefmt='grid')  # 使用'grid'格式，带外边框
    print(table)

def get_mIoU(num_classes, label_gts, label_preds,pth):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    a=100*score_dict['precision_0']
    print('000')
    metrics = [
        ['unchange',100*score_dict['precision_0'], 100*score_dict['recall_0'], 100*score_dict['iou_0'],100*score_dict['F1_0'],100*score_dict['acc']],
        ['change',100*score_dict['precision_1'], 100*score_dict['recall_1'], 100*score_dict['iou_1'],100*score_dict['F1_1'],100*score_dict['acc']],
        ['average',(score_dict['precision_0']+score_dict['precision_1'])*0.5*100,(score_dict['recall_0']+score_dict['recall_1'])*0.5*100, score_dict['miou']*100,score_dict['mf1']*100,score_dict['acc']*100]
    ]
    headers = ['Class', 'Precision', 'Recall', 'IoU', 'F1 Score','Acc']
    # 打印表格
    print_table(metrics, headers)
    save_to_txt(metrics, headers, pth)
    print('已保存至',pth)
    return metrics


# 使用tabulate库保存表格到txt文件（带外边框）
def save_to_txt(metrics, headers, filename='metrics_output.txt'):
    table = tabulate(metrics, headers, tablefmt='grid')  # 使用'grid'格式，带外边框
    with open(filename, 'w') as f:
        f.write(table)

if __name__ == '__main__':

    GT_dir = ('D:\lm\\data\WHU\\train\label')
    pred_dir = 'C:\\Users\lm\Desktop'
    clolr_save_dir = 'C:\\Users\lm\Desktop\save'
    info_txt_path = os.path.join(pred_dir, 'info.txt')
    f = open(info_txt_path, 'w+')
    data_list = os.listdir(pred_dir)
    num_valid = 0
    preds = []
    GTs = []
    for idx, it in enumerate(data_list):
        if it[-4:] == '.png':
            pred_path = os.path.join(pred_dir, it)
            pred = io.imread(pred_path)
            # pred = cv2.flip(pred, 1)
            pred = cv2.resize(pred, (256, 256), interpolation=cv2.INTER_LINEAR)
            h, w = pred.shape[:2]
            GT_path = os.path.join(GT_dir, it[:-4] + '.png')
            GT = io.imread(GT_path)
            # pred=pred[:,:,0]
            pred[pred > 0] = 255
            if GT.max() != 255:
                GT = GT.astype(np.uint8)
                GT[GT > 0] = 255
            pred = pred // 255
            GT = GT // 255
            preds.append(pred)
            GTs.append(GT)
            """红色漏判 蓝色错判"""
            I = np.stack((255 * GT, 255 * pred, 255 * pred), 2)
            """红色错判 蓝色漏判"""
            # I = np.stack((255 * pred, 255 * GT, 255 * GT), 2)
            if not os.path.exists(clolr_save_dir):
                os.makedirs(clolr_save_dir)
            I_path = os.path.join(clolr_save_dir, it)
            io.imsave(I_path, I.astype(np.uint8))
    mIOU = get_mIoU(2, GTs, preds, info_txt_path)

