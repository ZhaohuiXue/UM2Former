import torch
import torch.distributed as dist


# 返回混淆矩阵
def evaluate(model, data_loader, device, num_classes, criterion, val_index, val_loss_sum, conf_mat=True):
    model.eval()
    if conf_mat:
        conf_mat = ConfusionMatrix(num_classes)
        with torch.no_grad():
            for image, target, _ in data_loader:
                image, target = image.to(device), target.to(device)
                predict = model(image)
                val_loss = criterion(predict, target)
                val_loss_sum = val_loss_sum + val_loss
                val_index = val_index + 1

                conf_mat.update(target.flatten(), predict.argmax(1).flatten())
            conf_mat.reduce_from_all_process()
        return conf_mat, val_index, val_loss_sum
    else:
        with torch.no_grad():
            for image, target, _ in data_loader:
                image, target = image.to(device), target.to(device)
                predict = model(image)
                val_loss = criterion(predict, target)
                val_loss_sum = val_loss_sum + val_loss
                val_index = val_index + 1
        return val_index, val_loss_sum


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, target, predict):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)  # 产生musk，不包含255的值
            # 很难不惊讶，下面这两步居然可以生成混淆矩阵
            inds = n*target[k].to(torch.int64) + predict[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()

        a = torch.diag(h)
        b = h.sum(0)
        c = h.sum(1)

        eps = 1e-7

        pa = a / (c + eps)
        ua = a / (b + eps)
        f1 = 2 * pa * ua / (pa + ua + eps)
        mean_f1 = torch.nanmean(f1)
        oa = torch.sum(a) / torch.sum(h)
        pe = torch.sum(b * c) / (torch.sum(c) * torch.sum(c))
        Kappa = (oa - pe) / (1 - pe)

        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))

        return mean_f1, oa, Kappa, iou

    def reduce_from_all_process(self):
        '''
        torch.distributed.is_available()： 函数用于检测当前 PyTorch 是否支持分布式训练。
            如果返回 True，表示当前安装的 PyTorch 支持分布式训练；
            如果返回 False，表示当前安装的 PyTorch 不支持分布式训练或未安装分布式训练相关的扩展库。

        torch.distributed.is_initialized()： 函数用于检测当前进程是否已经初始化了分布式训练环境。
            如果返回True，表示当前进程已经完成分布式环境的初始化；
            如果返回False，表示当前进程还未完成分布式环境的初始化，或当前没有初始化分布式环境。
        '''
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        mf1, oa, kappa, iou= self.compute()
        return (
            'mF1: {:.1f}\n'
            'OA: {}\n'
            'Kappa: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                mf1 * 100,
                oa * 100,
                kappa * 100,
                ['{:.1f}'.format(i) for i in (iou * 100).tolist()],
                torch.nanmean(iou) * 100)