import torch


def l1(output, target):
    return torch.mean(torch.abs(output - target))

def l1_wav(output_dict, target_dict):
	return l1(output_dict['segment'], target_dict['segment'])

def sisnr(output_dict, target_dict, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """
    x = output_dict['segment']; s = target_dict['segment']
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    sisnr = -20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))
    return sisnr.mean()

def get_loss_function(loss_type):
    if loss_type == "l1_wav":
        return l1_wav
    elif loss_type == "sisnr":
        return sisnr
    else:
        raise NotImplementedError("Error!")
