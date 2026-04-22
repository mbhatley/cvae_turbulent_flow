import numpy as np
import os

import torch
from torch.utils.data import DataLoader

from datetime import datetime

from src.shared.data_loader import CVAEDataset


def save_model(model, save_dir, config=None):
    """
    Saves the trained model, and generated emulated images

    :param model: trained model
    :param save_dir: directory to save model
    :param config: optional configuration dict to save with model
    """
    model_path = os.path.join(save_dir, 'model.pt')

    save_dict = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
        'grid_shape': model.grid_shape,
        'latent_size': model.latent_size,
        'n_knots': model.n_knots,
        'alpha': model.alpha
    }

    if config is not None:
        save_dict['config'] = config

    torch.save(save_dict, model_path)
    print(f"Model saved: {model_path}")


def export_reconstructions_npy(model, numpy_file, device,
                               output_file='reconstructions.npy', batch_size=8):
    """
    Run all images through the model IN ORDER and save as 4D numpy array.
    Shape: [n_images, z, y, x] — same as input, ready for inverse ECDF.
    """
    data_4d = np.load(numpy_file)
    n_images, z_dim, y_dim, x_dim = data_4d.shape
    grid_size = x_dim * y_dim * z_dim

    data_flat = np.zeros((n_images, grid_size))
    for i in range(n_images):
        volume = np.transpose(data_4d[i], (2, 1, 0))  # [z,y,x] -> [x,y,z]
        data_flat[i] = volume.flatten()

    dataset = CVAEDataset(data_flat)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_recons = []

    print("Running inference on all samples in order...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            data_batch, _, _ = [item.to(device) for item in batch_data]
            conditioning = torch.zeros(data_batch.size(0), 1, device=device)
            recon, _, _ = model(data_batch, conditioning)
            all_recons.append(recon.cpu().numpy())
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {min((batch_idx+1)*batch_size, n_images)}/{n_images}")

    all_recons = np.vstack(all_recons)

    recons_4d = np.zeros((n_images, z_dim, y_dim, x_dim))
    for i in range(n_images):
        volume = all_recons[i].reshape(x_dim, y_dim, z_dim)
        recons_4d[i] = np.transpose(volume, (2, 1, 0))  # [x,y,z] -> [z,y,x]

    np.save(output_file, recons_4d)
    print(f"Saved reconstructions: {output_file}, shape: {recons_4d.shape}")
    return recons_4d

def evaluate_model(model, test_loader, device, save_latents=False, save_dir='results'):
    model.eval()

    total_mse = 0
    total_samples = 0
    correlations = []

    all_mu = []
    all_logvar = []

    with torch.no_grad():
        for batch_data in test_loader:
            data, mask, labels = [item.to(device) for item in batch_data]
            conditioning = torch.zeros(data.size(0), 1, device=device)

            mu, logvar = model.encode(data, conditioning)
            recon, _, _ = model(data, conditioning)

            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())

            original = data.cpu().numpy()
            reconstructed = recon.cpu().numpy()

            total_mse += ((reconstructed - original) ** 2).sum()
            total_samples += len(data)

            for i in range(len(original)):
                corr = np.corrcoef(original[i].flatten(), reconstructed[i].flatten())[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

    mse = total_mse / total_samples
    correlation = np.mean(correlations) if correlations else 0

    print(f"MSE: {mse:.6f}")
    print(f"Correlation: {correlation:.4f}")

    if save_latents:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'mu.npy'),     np.concatenate(all_mu,     axis=0))
        np.save(os.path.join(save_dir, 'logvar.npy'), np.concatenate(all_logvar, axis=0))
        print(f"Latents saved to: {save_dir}")

    return mse, correlation
