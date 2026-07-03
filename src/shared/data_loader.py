import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split, TimeSeriesSplit


class CVAEDataset(Dataset):
    """
    Dataset for CVAE with NaN handling
    """
    def __init__(self, images, pod_coeffs=None):
        self.images = torch.FloatTensor(images)
        self.masks = (~torch.isnan(self.images)).float()
        self.images = torch.nan_to_num(self.images, nan=0.0)

        if pod_coeffs is not None:
            self.pod_coeffs = torch.FloatTensor(pod_coeffs)
        else:
            self.pod_coeffs = torch.zeros(len(images), 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx], self.pod_coeffs[idx]


class CVAESetup():
    """
    Load 4D numpy arrays and prepare data loaders for CVAE

    Expected input shape: [n_images, z, y, x]
    """

    def __init__(self, numpy_file, batch_size=32, model='3d', pod_file=None):
        self.numpy_file = numpy_file
        self.batch_size = batch_size
        self.pod_file = pod_file

    def setup_device(self):
        """
        Select best available device (CUDA > MPS > CPU)
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA")
        else:
            device = torch.device('cpu')
            print("Using CPU")

        return device

    def load_data(self):
        """
        Load 4D numpy array and create data loaders

        : return:   train_loader: DataLoader for training data,
                    test_loader: DataLoader for test data,
                    grid_shape: Tuple (x, y, z) dimensions,
                    grid_size: Total number of voxels,
        """
        print(f"Loading numpy file: {self.numpy_file}")

        # Load 4D array [n_images, z, y, x]
        data_4d = np.load(self.numpy_file)
        print(f"Loaded shape: {data_4d.shape}")

        # Parse dimensions
        n_images, z_dim, y_dim, x_dim = data_4d.shape
        grid_shape = (x_dim, y_dim, z_dim)  # (x, y, z) ordering
        grid_size = x_dim * y_dim * z_dim

        print(f"3D Grid: {x_dim}×{y_dim}×{z_dim} = {grid_size}")
        print(f"Number of images: {n_images}")

        # Reshape from [n_images, z, y, x] to [n_images, grid_size]
        data_flat = np.zeros((n_images, grid_size))

        for i in range(n_images):
            volume = np.transpose(data_4d[i], (2, 1, 0))
            data_flat[i] = volume.flatten()

        pod_coeffs = None
        if self.pod_file is not None:
            pod_data = np.load(self.pod_file)
            pod_coeffs = pod_data['coeffs_std']
            n_modes = int(pod_data['n_modes'])
            print(f"POD conditioning: {n_modes} modes, {pod_coeffs.shape[0]} samples")

        indices = np.arange(n_images)
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

        train_data = data_flat[train_idx]
        test_data = data_flat[test_idx]
        train_pod = pod_coeffs[train_idx] if pod_coeffs is not None else None
        test_pod = pod_coeffs[test_idx] if pod_coeffs is not None else None
        print(f"Train: {len(train_data)} samples, Test: {len(test_data)} samples")

        train_dataset = CVAEDataset(train_data, pod_coeffs=train_pod)
        test_dataset = CVAEDataset(test_data, pod_coeffs=test_pod)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        return train_loader, test_loader, grid_shape, grid_size

    def load_data_cv(self, n_splits=5):
        """
        Load data and yield train/val DataLoaders for each TimeSeriesSplit fold.
        Temporal order is preserved: each val fold is strictly after its train fold.

        Yields: fold_idx, train_loader, val_loader, grid_shape, grid_size
        """
        print(f"Loading numpy file: {self.numpy_file}")
        data_4d = np.load(self.numpy_file)
        print(f"Loaded shape: {data_4d.shape}")

        n_images, z_dim, y_dim, x_dim = data_4d.shape
        grid_shape = (x_dim, y_dim, z_dim)
        grid_size = x_dim * y_dim * z_dim

        print(f"3D Grid: {x_dim}×{y_dim}×{z_dim} = {grid_size}")
        print(f"Number of images: {n_images}")

        data_flat = np.zeros((n_images, grid_size))
        for i in range(n_images):
            volume = np.transpose(data_4d[i], (2, 1, 0))
            data_flat[i] = volume.flatten()

        pod_coeffs = None
        if self.pod_file is not None:
            pod_data = np.load(self.pod_file)
            pod_coeffs = pod_data['coeffs_std']
            n_modes = int(pod_data['n_modes'])
            print(f"POD conditioning: {n_modes} modes, {pod_coeffs.shape[0]} samples")

        tscv = TimeSeriesSplit(n_splits=n_splits)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(data_flat)):
            print(f"\nFold {fold + 1}/{n_splits}: "
                  f"train t={train_idx[0]}..{train_idx[-1]} ({len(train_idx)} samples), "
                  f"val t={val_idx[0]}..{val_idx[-1]} ({len(val_idx)} samples)")

            train_data = data_flat[train_idx]
            val_data   = data_flat[val_idx]
            train_pod  = pod_coeffs[train_idx] if pod_coeffs is not None else None
            val_pod    = pod_coeffs[val_idx]   if pod_coeffs is not None else None

            train_dataset = CVAEDataset(train_data, pod_coeffs=train_pod)
            val_dataset   = CVAEDataset(val_data,   pod_coeffs=val_pod)

            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
            )

            yield fold, train_loader, val_loader, grid_shape, grid_size