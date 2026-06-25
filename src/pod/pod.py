"""
Proper Orthogonal Decomposition (POD) for turbulent flow conditioning.
"""
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'


def load_and_flatten(numpy_file):
    """Flatten [n_images, z, y, x] -> [n_images, grid_size] using the same
    (x, y, z) ordering as src/shared/data_loader.py, so sample i here lines
    up with sample i in the PIT-transformed file.
    """
    data_4d = np.load(numpy_file)
    n_images, z_dim, y_dim, x_dim = data_4d.shape
    grid_shape = (x_dim, y_dim, z_dim)
    grid_size = x_dim * y_dim * z_dim

    data_flat = np.zeros((n_images, grid_size))
    for i in range(n_images):
        volume = np.transpose(data_4d[i], (2, 1, 0))
        data_flat[i] = volume.flatten()

    valid_mask = ~np.isnan(data_flat).all(axis=0)
    return data_flat, grid_shape, valid_mask


def _fill_nans(X, fill_values):
    if not np.isnan(X).any():
        return X
    X = X.copy()
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = fill_values[cols]
    return X


def compute_pod_basis(data_flat, valid_mask=None):
    """Economy-SVD POD. Returns mean field, modes (energy-ordered), singular
    values, and cumulative energy fraction. No truncation yet -- inspect the
    energy spectrum first, then call project_to_coeffs with your chosen r.

    Fit this on the TRAIN split only to avoid leakage into test metrics.
    """
    X = data_flat[:, valid_mask] if valid_mask is not None else data_flat
    X = _fill_nans(X, np.nanmean(X, axis=0))

    mean_field = X.mean(axis=0)
    Xc = X - mean_field

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # Vt rows = spatial modes

    energy = S ** 2
    energy_fraction = np.cumsum(energy) / energy.sum()

    return {
        'mean_field': mean_field,
        'modes': Vt,
        'singular_values': S,
        'energy_fraction': energy_fraction,
        'valid_mask': valid_mask,
    }


def modes_for_energy(basis, threshold=0.95):
    """How many leading modes are needed to reach `threshold` cumulative energy."""
    return int(np.searchsorted(basis['energy_fraction'], threshold) + 1)


def project_to_coeffs(data_flat, basis, n_modes):
    """Project snapshots onto the first n_modes POD modes -> (n_samples, n_modes)."""
    valid_mask = basis['valid_mask']
    X = data_flat[:, valid_mask] if valid_mask is not None else data_flat
    X = _fill_nans(X, basis['mean_field'])
    Xc = X - basis['mean_field']
    return Xc @ basis['modes'][:n_modes].T


def standardize(coeffs, mean=None, std=None):
    """Z-score each mode. Fit (mean=None) on the TRAIN split only, then reuse
    the returned mean/std for the test split and at inference/generation time.
    """
    if mean is None:
        mean = coeffs.mean(axis=0)
    if std is None:
        std = coeffs.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
    return (coeffs - mean) / std, mean, std


if __name__ == '__main__':
    # Edit these directly and run in your IDE.
    NUMPY_FILE = str(DATA_DIR / 't_original.npy')
    ENERGY_THRESHOLD = 0.95
    OUT_FILE = str(DATA_DIR / 't_pod_basis.npz')

    data_flat, grid_shape, valid_mask = load_and_flatten(NUMPY_FILE)
    print(f"Loaded {data_flat.shape[0]} snapshots, grid {grid_shape}, "
          f"{valid_mask.sum()}/{valid_mask.size} valid points")

    basis = compute_pod_basis(data_flat, valid_mask)
    r = modes_for_energy(basis, ENERGY_THRESHOLD)
    print(f"{r} modes needed for {ENERGY_THRESHOLD:.0%} cumulative energy "
          f"(max possible: {len(basis['singular_values'])})")

    # Print the spectrum so you can sanity-check the threshold / look for an elbow.
    for k in sorted(set([1, 2, 5, 10, 20, 50, r])):
        if k <= len(basis['energy_fraction']):
            print(f"  modes={k:4d}  energy={basis['energy_fraction'][k-1]:.4f}")

    coeffs = project_to_coeffs(data_flat, basis, r)
    coeffs_std, mean, std = standardize(coeffs)

    np.savez(OUT_FILE,
             mean_field=basis['mean_field'],
             modes=basis['modes'][:r],
             singular_values=basis['singular_values'],
             valid_mask=valid_mask if valid_mask is not None else np.array([]),
             coeff_mean=mean,
             coeff_std=std,
             coeffs_std=coeffs_std,
             n_modes=r)
    print(f"Saved POD basis ({r} modes), standardized coefficients "
          f"{coeffs_std.shape} to {OUT_FILE}")