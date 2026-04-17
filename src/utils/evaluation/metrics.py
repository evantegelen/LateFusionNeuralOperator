import torch

def rmse(u_pred, u_true):
    err = (u_pred - u_true) ** 2
    return torch.sqrt(err.mean()) 

def nrmse(u_pred, u_true, eps=1e-8):
    err = (u_pred - u_true) ** 2
    rmse = torch.sqrt(err.mean())
    norm = torch.sqrt((u_true ** 2).mean()) + eps
    return (rmse / norm)

def max_error(u_pred, u_true):
    return torch.abs(u_pred - u_true).max()

def boundary_rmse(u_pred, u_true):
    left = (u_pred[:, :, 0, :] - u_true[:, :, 0, :]) ** 2
    right = (u_pred[:, :, -1, :] - u_true[:, :, -1, :]) ** 2
    err = (left + right) / 2.0
    return torch.sqrt(err).mean()

def conserved_error(u_pred, u_true):
    pred_sum = u_pred.sum(dim=2)
    true_sum = u_true.sum(dim=2)
    err = (pred_sum - true_sum) ** 2
    return torch.sqrt(err.mean())

def fourier_rmse(u_pred, u_true):
    # FFT over spatial dimension (1D case)
    u_pred_f = torch.fft.rfft(u_pred, dim=2)
    u_true_f = torch.fft.rfft(u_true, dim=2)

    err = torch.abs(u_pred_f - u_true_f) ** 2
    return torch.sqrt(err.mean())

def compute_all_metrics(u_pred, u_true):
    return {
        "RMSE": rmse(u_pred, u_true),
        "nRMSE": nrmse(u_pred, u_true),
        "Max": max_error(u_pred, u_true),
        "Boundary": boundary_rmse(u_pred, u_true),
        "Conserved": conserved_error(u_pred, u_true),
        "Fourier": fourier_rmse(u_pred, u_true),
    }