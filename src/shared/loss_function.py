import torch
from torch import nn
from torch.nn import functional as F


class CVAELoss(nn.Module):
    def __init__(self):
        super(CVAELoss, self).__init__()


    @staticmethod
    def renyi_divergence(recon, target, alpha=0.5):
        """
        Rényi divergence (alpha-divergence) between flattened distributions
        alpha=0.5 is Hellinger distance, alpha→1 is KL divergence

        : param recon:
        : param target:
        : param alpha:
        : return:
        """
        # Flatten and normalize to probability distributions
        recon_flat = recon.view(recon.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Add small epsilon for numerical stability
        eps = 1e-8
        p = F.softmax(target_flat, dim=1) + eps
        q = F.softmax(recon_flat, dim=1) + eps

        # Rényi divergence formula
        if alpha == 1.0:
            # KL divergence limit
            return torch.mean(torch.sum(p * torch.log(p / q), dim=1))
        else:
            log_term = torch.log(torch.sum(p**alpha * q**(1-alpha), dim=1))
            return torch.mean(log_term / (alpha - 1))


    @staticmethod
    def compute_gradient_loss(prediction, target, grid_shape):
        """
        Calculates the L1 difference in gradients between prediction and target.
        Forces the model to keep edges sharp and turbulent.
        """
        x_dim, y_dim, z_dim = grid_shape

        # Reshape to 5D: (Batch, Channel, D, H, W)
        prediction = prediction.view(-1, 1, x_dim, y_dim, z_dim)
        target = target.view(-1, 1, x_dim, y_dim, z_dim)

        # Compute gradients in X, Y, Z directions
        # We use simple finite difference
        pred_dx = torch.abs(prediction[:, :, 1:, :, :] - prediction[:, :, :-1, :, :])
        pred_dy = torch.abs(prediction[:, :, :, 1:, :] - prediction[:, :, :, :-1, :])
        pred_dz = torch.abs(prediction[:, :, :, :, 1:] - prediction[:, :, :, :, :-1])

        target_dx = torch.abs(target[:, :, 1:, :, :] - target[:, :, :-1, :, :])
        target_dy = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
        target_dz = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])

        loss_dx = F.l1_loss(pred_dx, target_dx)
        loss_dy = F.l1_loss(pred_dy, target_dy)
        loss_dz = F.l1_loss(pred_dz, target_dz)

        return loss_dx + loss_dy + loss_dz

    @staticmethod
    def compute_loss(recon_batch, data, mu, logvar, grid_shape, beta=1.0,
                     extreme_weight=1.0, renyi_weight=0.0, grad_weight = 1.0,
                     loss='l1'):
        """
        ELBO loss with reconstruction (L1 or MSE), KL divergence, and optional gradient loss and Renji loss
        :param recon_batch:
        :param data:
        :param mu:
        :param logvar:
        :param grid_shape:
        :param beta: optional weighting parameter for the KL Divergence. Default is 1.0.
        :param extreme_weight: Weighting applied to tail distributions. Default is 1.0. Make larger to force more extreme tail events
        :param renyi_weight: Weighting applied to renyi distributions. Default is 0.
        :param grad_weight: Weighting applied to gradient of the reconstruction loss. Default is 1.0.
        :param loss: Loss type. Options are L1 or None. L1 returns L1 loss, None returns MSE loss. Default is 'l1'.
        :return: total loss, reconstruction loss, KL loss
        """

        distance_from_mean = torch.abs(data - 0.5)
        is_extreme = (distance_from_mean > 0.3).float()
        weights = 1.0 + (is_extreme * extreme_weight)

        if loss == 'l1':
            recon_loss = F.l1_loss(recon_batch, data, reduction='none')
        else:
            recon_loss = F.mse_loss(recon_batch, data, reduction='none')

        recon_loss = (recon_loss * weights).mean()

        # Gradient Loss (Keeps edges sharp)
        grad_loss = CVAELoss.compute_gradient_loss(recon_batch, data, grid_shape)

        # Rényi divergence for distribution matching
        renyi_loss = CVAELoss.renyi_divergence(recon_batch, data, alpha=0.5)

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + (beta * kl_loss) + (grad_weight * grad_loss) + (renyi_weight * renyi_loss)

        return total_loss, recon_loss, kl_loss


# Module-level convenience function so train.py can call compute_loss() directly
def compute_loss(recon_batch, data, mu, logvar, grid_shape, beta=1.0,
                 extreme_weight=4.0, renyi_weight=0.0, grad_weight=1.0, loss='l1'):
    return CVAELoss.compute_loss(recon_batch, data, mu, logvar, grid_shape,
                                 beta=beta, extreme_weight=extreme_weight,
                                 renyi_weight=renyi_weight, grad_weight=grad_weight,
                                 loss=loss)