"""
Standalone POD reconstruction benchmark for comparison against the CVAE.

pod.py builds the c-vector used to condition the CVAE. This script instead
asks: how well does POD alone reconstruct the field, at the same mode count
the CVAE is conditioned on? It uses the same 80/20 train/test split as
CVAESetup.load_data (random_state=42) and the same MSE/correlation
definitions as save_emulation.evaluate_model, so the printed numbers are
directly comparable to the CVAE's reported test metrics.
"""
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.pod.pod import load_and_flatten, compute_pod_basis, project_to_coeffs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'


def reconstruct_full_grid(coeffs, basis, n_modes, grid_size):
    """Embed a POD reconstruction (valid points only) back into the full,
    zero-filled grid -- matches how CVAEDataset zeroes out-of-domain NaNs,
    so this lines up with what evaluate_model sees as "original".
    """
    n_samples = coeffs.shape[0]
    valid_mask = basis['valid_mask']
    recon_valid = coeffs @ basis['modes'][:n_modes] + basis['mean_field']

    if valid_mask is not None:
        recon = np.zeros((n_samples, grid_size))
        recon[:, valid_mask] = recon_valid
    else:
        recon = recon_valid
    return recon


def evaluate_reconstruction(original_flat, recon_flat):
    """Same metric definitions as save_emulation.evaluate_model: per-sample
    squared error summed then averaged over samples (not elements), and
    mean per-sample Pearson correlation over the flattened grid.
    """
    original = np.nan_to_num(original_flat, nan=0.0)
    mse = ((recon_flat - original) ** 2).sum() / original.shape[0]

    correlations = []
    for i in range(original.shape[0]):
        corr = np.corrcoef(original[i], recon_flat[i])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    correlation = np.mean(correlations) if correlations else 0.0

    return mse, correlation


def unflatten(data_flat, grid_shape):
    """Inverse of load_and_flatten: [n, grid_size] (x,y,z order) -> [n, z, y, x]."""
    x_dim, y_dim, z_dim = grid_shape
    out = np.zeros((data_flat.shape[0], z_dim, y_dim, x_dim))
    for i in range(data_flat.shape[0]):
        volume = data_flat[i].reshape(x_dim, y_dim, z_dim)
        out[i] = np.transpose(volume, (2, 1, 0))
    return out


def run(numpy_file, pod_basis_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    data_flat, grid_shape, valid_mask = load_and_flatten(numpy_file)
    n_images, grid_size = data_flat.shape
    print(f"Loaded {n_images} snapshots, grid {grid_shape}, from {numpy_file}")

    n_modes = int(np.load(pod_basis_file)['n_modes'])
    print(f"Using n_modes={n_modes} (matched to {pod_basis_file})")

    # Same 80/20 split as CVAESetup.load_data, so train/test indices line up
    # with what the CVAE was fit and evaluated on.
    indices = np.arange(n_images)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # Fit POD on train only -- no leakage into the test metric below.
    basis = compute_pod_basis(data_flat[train_idx], valid_mask)

    # Reconstruct ALL 900 snapshots (in original order) through the train-fit
    # basis, matching how export_reconstructions_npy runs the trained CVAE
    # over every snapshot -- so the two .npy files are directly comparable.
    all_coeffs = project_to_coeffs(data_flat, basis, n_modes)
    recon_flat = reconstruct_full_grid(all_coeffs, basis, n_modes, grid_size)

    mse, correlation = evaluate_reconstruction(data_flat[test_idx], recon_flat[test_idx])
    print(f"POD ({n_modes} modes) test MSE: {mse:.6f}")
    print(f"POD ({n_modes} modes) test correlation: {correlation:.4f}")

    field = Path(numpy_file).stem
    recon_out = os.path.join(output_dir, f'pod_reconstructions_{field}.npy')
    np.save(recon_out, unflatten(recon_flat, grid_shape))
    print(f"Saved reconstructions (all {n_images} snapshots, original order): {recon_out}")

    return mse, correlation


if __name__ == '__main__':
    # Edit these directly and run in your IDE.
    NUMPY_FILE = str(DATA_DIR / 'skewnormal_gev_t.npy')
    POD_BASIS_FILE = str(DATA_DIR / 't_pod_basis.npz')  # matches n_modes to this c-vector basis
    OUTPUT_DIR = str(PROJECT_ROOT / 'src' / 'pod' / 'results')

    run(NUMPY_FILE, POD_BASIS_FILE, OUTPUT_DIR)