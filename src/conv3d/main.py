import os
from pathlib import Path
from datetime import datetime

# Project root is three levels up from this file (src/conv3d/main.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'

import torch

from src.shared.data_loader import CVAESetup
from src.shared.save_emulation import save_model, export_reconstructions_npy, evaluate_model
from src.conv3d.model import CVAE
from src.conv3d.train import train_model
from src.visualization.plot_3d import CVAEVisuals


def main():
    """
    Main training pipeline using numpy array data
    """

    # Load data from numpy file
    setup = CVAESetup(
        numpy_file=str(DATA_DIR / 'transformed_data_u3.npy'),  # 4D array: [n_images, z, y, x]
        batch_size=8
    )

    viz = CVAEVisuals()

    device = setup.setup_device()

    train_loader, test_loader, grid_shape, grid_size = setup.load_data()

    # Initialize model
    model = CVAE(
        grid_shape=grid_shape,
        latent_size=128,
        class_size=1,
        dropout_rate=0.2,
    ).to(device)

    # Train model
    model, loss_dict = train_model(
        model, train_loader, test_loader, device, grid_shape, epochs=100
    )

    # # Evaluate model
    # mse, correlation = evaluate_model(
    #     model=model, test_loader=test_loader, device=device
    # )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    recons = export_reconstructions_npy(
        model=model,
        numpy_file=str(DATA_DIR / 'transformed_data_u3.npy'),
        device=device,
        output_file=os.path.join(save_dir, 'reconstructions_u3.npy'),
        batch_size=8
    )

    # Generate visualizations
    viz.plot_training_curve(loss_dict['train_total'], loss_dict['test_total'], save_dir)
    viz.plot_component_losses(loss_dict, save_dir)
    viz.plot_learning_rate_curve(loss_dict['learning_rates'], save_dir)

    viz.plot_3d_reconstructions(model, test_loader, device, grid_shape, save_dir=save_dir)
    viz.plot_3d_latent_space(model, test_loader, device, save_dir=save_dir)
    viz.plot_detailed_3d_analysis(model, test_loader, device, grid_shape, save_dir=save_dir)

    # # Save model configuration
    # model_config = {
    #     'grid_shape': grid_shape,
    #     'latent_size': model.latent_size,
    #     'class_size': model.class_size,
    #     'n_knots': model.n_knots,
    #     'alpha': model.alpha,
    #     'dropout_rate': model.dropout_rate,
    #     'knot_method': model.knot_method,
    #     'mse': float(mse),
    #     'correlation': float(correlation)
    # }

    # save_model(model, save_dir, config=model_config)

    print(f"All outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
