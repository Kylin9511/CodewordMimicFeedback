import torch

__all__ = ['AverageMeter', 'evaluator']


class AverageMeter(object):
    r"""Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"


def evaluator(sparse_pred, sparse_gt, raw_gt):
    r""" Evaluation of decoding implemented in PyTorch Tensor
         Computes normalized mean square error (NMSE) and rho.
    """

    with torch.no_grad():
        # Basic params
        nt = 32
        nc = 32
        nc_expand = 257

        # De-centralize
        sparse_gt = sparse_gt - 0.5
        sparse_pred = sparse_pred - 0.5

        # Calculate the NMSE
        power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
        difference = sparse_gt - sparse_pred
        mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
        nmse = (mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean()

        # Calculate the Rho
        n = sparse_pred.size(0)
        sparse_pred = torch.complex(sparse_pred[:, 0, ...], sparse_pred[:, 1, ...])
        complex_zeros = sparse_pred.new_zeros((n, nt, nc_expand - nc))
        sparse_pred = torch.cat((sparse_pred, complex_zeros), dim=2)
        raw_pred = torch.fft.fft(sparse_pred, dim=2)[:, :, :125]
        raw_gt = torch.complex(raw_gt[..., 0], raw_gt[..., 1])

        norm_pred = torch.sqrt(torch.sum(torch.conj(raw_pred) * raw_pred, dim=1))
        norm_pred = torch.abs(norm_pred)
        norm_gt = torch.sqrt(torch.sum(torch.conj(raw_gt) * raw_gt, dim=1))
        norm_gt = torch.abs(norm_gt)
        norm_cross = torch.abs(torch.sum(torch.conj(raw_gt) * raw_pred, dim=1))

        rho = torch.mean(norm_cross / (norm_pred * norm_gt), dim=1)
        rho = rho.mean()
        return rho, nmse
