import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


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
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS")
        else:
            device = torch.device('cpu')
            print("Using CPU")

        return device

    def load_data(self):
        """
        Load 4D numpy array and create data loaders preserving 3D spatial geometry.

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

        # Transpose from [n_images, z, y, x] to [n_images, x, y, z] to match grid_shape ordering
        # and add a channel dimension to get [n_images, 1, x, y, z]
        data_spatial = np.transpose(data_4d, (0, 3, 2, 1))
        data_spatial = data_spatial[:, np.newaxis, :, :, :]

        pod_coeffs = None
        if self.pod_file is not None:
            pod_data = np.load(self.pod_file)
            pod_coeffs = pod_data['coeffs_std']
            n_modes = int(pod_data['n_modes'])
            print(f"POD conditioning: {n_modes} modes, {pod_coeffs.shape[0]} samples")

        indices = np.arange(n_images)
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

        # Slice the structured spatial tensors instead of flattened arrays
        train_data = data_spatial[train_idx]
        test_data = data_spatial[test_idx]

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


    # def load_data(self):
    #     """
    #     Load 4D numpy array and create data loaders
    #
    #     : return:   train_loader: DataLoader for training data,
    #                 test_loader: DataLoader for test data,
    #                 grid_shape: Tuple (x, y, z) dimensions,
    #                 grid_size: Total number of voxels,
    #     """
    #     print(f"Loading numpy file: {self.numpy_file}")
    #
    #     # Load 4D array [n_images, z, y, x]
    #     data_4d = np.load(self.numpy_file)
    #     print(f"Loaded shape: {data_4d.shape}")
    #
    #     # Parse dimensions
    #     n_images, z_dim, y_dim, x_dim = data_4d.shape
    #     grid_shape = (x_dim, y_dim, z_dim)  # (x, y, z) ordering
    #     grid_size = x_dim * y_dim * z_dim
    #
    #     print(f"3D Grid: {x_dim}×{y_dim}×{z_dim} = {grid_size}")
    #     print(f"Number of images: {n_images}")
    #
    #     # Reshape from [n_images, z, y, x] to [n_images, grid_size]
    #     data_flat = np.zeros((n_images, grid_size))
    #
    #     for i in range(n_images):
    #         volume = np.transpose(data_4d[i], (2, 1, 0))
    #         data_flat[i] = volume.flatten()
    #
    #     pod_coeffs = None
    #     if self.pod_file is not None:
    #         pod_data = np.load(self.pod_file)
    #         pod_coeffs = pod_data['coeffs_std']
    #         n_modes = int(pod_data['n_modes'])
    #         print(f"POD conditioning: {n_modes} modes, {pod_coeffs.shape[0]} samples")
    #
    #     indices = np.arange(n_images)
    #     train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    #
    #     train_data = data_flat[train_idx]
    #     test_data = data_flat[test_idx]
    #     train_pod = pod_coeffs[train_idx] if pod_coeffs is not None else None
    #     test_pod = pod_coeffs[test_idx] if pod_coeffs is not None else None
    #     print(f"Train: {len(train_data)} samples, Test: {len(test_data)} samples")
    #
    #     train_dataset = CVAEDataset(train_data, pod_coeffs=train_pod)
    #     test_dataset = CVAEDataset(test_data, pod_coeffs=test_pod)
    #
    #     train_loader = DataLoader(
    #         train_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         num_workers=0,
    #     )
    #
    #     test_loader = DataLoader(
    #         test_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=0,
    #     )
    #
    #     return train_loader, test_loader, grid_shape, grid_size
