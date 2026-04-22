import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split


class CVAEDataset(Dataset):
    """
    Dataset for CVAE with NaN handling
    """
    def __init__(self, images, labels=None):
        self.images = torch.FloatTensor(images)
        self.masks = (~torch.isnan(self.images)).float()
        self.images = torch.nan_to_num(self.images, nan=0.0)

        if labels is not None:
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = torch.zeros(len(images), dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx], self.labels[idx]


class CVAESetup():
    """
    Load 4D numpy arrays and prepare data loaders for CVAE

    Expected input shape: [n_images, z, y, x]
    """

    def __init__(self, numpy_file, batch_size=32, model = '3d'):
        """
        Initialize data setup

        :param numpy_file: 4D numpy array
        :param batch_size: batch size
        :param model: Defines if data is 2D or 3D. Default is 3D.
        """
        self.numpy_file = numpy_file
        self.batch_size = batch_size

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
            # Extract 3D volume [z, y, x]
            volume = data_4d[i]

            # Transpose to [x, y, z] to match grid_shape ordering
            volume = np.transpose(volume, (2, 1, 0))

            # Flatten to 1D
            data_flat[i] = volume.flatten()

        train_data, test_data = train_test_split(data_flat, test_size=0.2, random_state=42)
        print(f"Train: {len(train_data)} samples, Test: {len(test_data)} samples")

        # Create datasets
        train_dataset = CVAEDataset(train_data)
        test_dataset = CVAEDataset(test_data)

        # Create data loaders
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