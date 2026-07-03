##################
## Changes
## latent space 64 -> 128
## dropout rate 0.2 -> 0.1

import os
from pathlib import Path
from datetime import datetime

# Project root is three levels up from this file (src/conv3d/main.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'

import numpy as np
import torch

from src.shared.data_loader import CVAESetup
from src.shared.save_emulation import save_model, export_reconstructions_npy, evaluate_model
from src.conv3d.model import CVAE
from src.conv3d.train import train_model
from src.visualization.plot_3d import CVAEVisuals


def main():
    """
    Main training pipeline — single 80/20 chronological train/test split.
    """

    pod_file = str(DATA_DIR / 't_pod_basis.npz')
    pod_data = np.load(pod_file)
    n_modes = int(pod_data['n_modes'])

    setup = CVAESetup(
        numpy_file=str(DATA_DIR / 'skewnormal_gev_t.npy'),
        batch_size=8,
        pod_file=pod_file,
    )

    viz = CVAEVisuals()
    device = setup.setup_device()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    train_loader, test_loader, grid_shape, grid_size = setup.load_data()

    model = CVAE(
        grid_shape=grid_shape,
        latent_size=256,
        class_size=n_modes,
        dropout_rate=0.05,
    ).to(device)

    model, loss_dict = train_model(
        model, train_loader, test_loader, device, grid_shape, epochs=300
    )

    export_reconstructions_npy(
        model=model,
        numpy_file=str(DATA_DIR / 'skewnormal_gev_t.npy'),
        device=device,
        output_file=os.path.join(run_dir, 'reconstructions_t.npy'),
        batch_size=8,
        pod_file=pod_file,
    )

    viz.plot_training_curve(loss_dict['train_total'], loss_dict['test_total'], run_dir)
    viz.plot_component_losses(loss_dict, run_dir)
    viz.plot_learning_rate_curve(loss_dict['learning_rates'], run_dir)
    viz.plot_3d_reconstructions(model, test_loader, device, grid_shape, save_dir=run_dir)
    viz.plot_3d_latent_space(model, test_loader, device, save_dir=run_dir)
    viz.plot_detailed_3d_analysis(model, test_loader, device, grid_shape, save_dir=run_dir)

    print(f"\nAll outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()