import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import seaborn as sns

class CVAEVisuals():

    def __init__(self, figsize_large=(18, 12), figsize_medium=(12, 6),
                 figsize_xlarge=(24, 16), dpi=150, cmap='RdYlBu_r'):
        """
        Initialize CVAEVisuals with common configuration parameters

        :param figsize_large: default figure size for large plots
        :param figsize_medium: default figure size for medium plots
        :param figsize_xlarge: default figure size for extra large plots
        :param dpi: resolution for saved figures
        :param cmap: default colormap for 3D visualizations
        """
        self.figsize_large = figsize_large
        self.figsize_medium = figsize_medium
        self.figsize_xlarge = figsize_xlarge
        self.dpi = dpi
        self.cmap = cmap

        # Set default plot style
        plt.style.use('default')
        sns.set_palette("husl")

    def _configure_axis(self, ax, title, xlabel=None, ylabel=None, grid=True):
        """
        Helper method to configure axis with common settings

        :param ax: matplotlib axis
        :param title: plot title
        :param xlabel: x-axis label
        :param ylabel: y-axis label
        :param grid: whether to show grid
        """
        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if grid:
            ax.grid(True, alpha=0.3)


    def plot_training_curve(self, train_losses, test_losses, save_dir):
        """Plot total loss curves"""
        fig, ax = plt.subplots(1, 1, figsize=self.figsize_medium)

        epochs = range(1, len(train_losses) + 1)

        # Loss curves
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, test_losses, 'r-', label='Validation Loss', linewidth=2)
        self._configure_axis(ax, 'Training Progress', 'Epoch', 'Loss')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curve.png'), dpi=self.dpi, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_component_losses(self, loss_dict, save_dir):
        """
        Plot KL and reconstruction losses separately

        :param loss_dict: Dictionary containing loss components from train_model
        :param save_dir: Directory to save plots
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize_large)

        epochs = range(1, len(loss_dict['train_recon']) + 1)

        # Reconstruction loss
        ax1.plot(epochs, loss_dict['train_recon'], 'b-', label='Training Recon Loss', linewidth=2)
        ax1.plot(epochs, loss_dict['test_recon'], 'r-', label='Validation Recon Loss', linewidth=2)
        self._configure_axis(ax1, 'Reconstruction Loss Progress', 'Epoch', 'Reconstruction Loss')
        ax1.legend()

        # KL loss
        ax2.plot(epochs, loss_dict['train_kl'], 'b-', label='Training KL Loss', linewidth=2)
        ax2.plot(epochs, loss_dict['test_kl'], 'r-', label='Validation KL Loss', linewidth=2)
        self._configure_axis(ax2, 'KL Divergence Progress', 'Epoch', 'KL Divergence Loss')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'component_losses.png'), dpi=self.dpi, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_learning_rate_curve(self, learning_rates, save_dir):
        """Plot learning rate schedule"""
        fig, ax = plt.subplots(1, 1, figsize=self.figsize_medium)

        epochs = range(1, len(learning_rates) + 1)

        ax.plot(epochs, learning_rates, 'g-', linewidth=2)
        self._configure_axis(ax, 'Learning Rate Schedule', 'Epoch', 'Learning Rate')
        # ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'learning_rate_curve.png'), dpi=self.dpi, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_3d_reconstructions(self, model, test_loader, device, grid_shape, n_samples=6, save_dir=None):
        """
        Plot 3D reconstructions with slice visualizations

        :param model: trained model
        :param test_loader: test data loader
        :param device: torch device
        :param grid_shape: tuple (x_dim, y_dim, z_dim) for proper reshaping
        :param n_samples: number of samples to visualize
        :param save_dir: directory to save plots
        """
        model.eval()

        with torch.no_grad():
            for batch in test_loader:
                images, masks, conditioning = batch
                images = images.to(device)
                conditioning = conditioning.to(device)

                recon, mu, logvar = model(images, conditioning)

                originals = images.cpu().numpy()
                reconstructions = recon.cpu().numpy()
                break

        n_samples = min(n_samples, len(originals))

        # Create 3D slice visualizations
        self._plot_3d_slice_reconstructions(
            originals, reconstructions, n_samples, grid_shape, save_dir
        )

        return {
            'originals': originals[:n_samples],
            'reconstructions': reconstructions[:n_samples]
        }

    def _plot_3d_slice_reconstructions(self, originals, reconstructions, n_samples, grid_shape, save_dir):
        """
        Produce ONE 3x3 grid per sample:
            Row 1: Original (X, Y, Z)
            Row 2: Reconstruction (X, Y, Z)
            Row 3: Difference (X, Y, Z)
            """

        x_dim, y_dim, z_dim = grid_shape

        slice_x = x_dim // 2
        slice_y = y_dim // 2
        slice_z = z_dim // 2

        for sample_idx in range(n_samples):

            orig_3d = originals[sample_idx].reshape(x_dim, y_dim, z_dim)
            recon_3d = reconstructions[sample_idx].reshape(x_dim, y_dim, z_dim)

            vmin = min(orig_3d.min(), recon_3d.min())
            vmax = max(orig_3d.max(), recon_3d.max())

            fig, axes = plt.subplots(3, 3, figsize=(18, 18))

            im1 = axes[0, 0].imshow(orig_3d[slice_x, :, :], cmap=self.cmap, vmin=vmin, vmax=vmax)
            axes[0, 0].set_title("Original X")
            plt.colorbar(im1, ax=axes[0, 0], shrink=0.5)

            im2 = axes[0, 1].imshow(orig_3d[:, slice_y, :], cmap=self.cmap, vmin=vmin, vmax=vmax)
            axes[0, 1].set_title("Original Y")
            plt.colorbar(im2, ax=axes[0, 1], shrink=0.5)


            im3 = axes[0, 2].imshow(orig_3d[:, :, slice_z], cmap=self.cmap, vmin=vmin, vmax=vmax)
            axes[0, 2].set_title("Original Z")
            plt.colorbar(im3, ax=axes[0, 2], shrink=0.5)


            im4 = axes[1, 0].imshow(recon_3d[slice_x, :, :], cmap=self.cmap, vmin=vmin, vmax=vmax)
            axes[1, 0].set_title("Reconstruction X")
            plt.colorbar(im4, ax=axes[1, 0], shrink=0.5)


            im5 = axes[1, 1].imshow(recon_3d[:, slice_y, :], cmap=self.cmap, vmin=vmin, vmax=vmax)
            axes[1, 1].set_title("Reconstruction Y")
            plt.colorbar(im5, ax=axes[1, 1], shrink=0.5)


            im6 = axes[1, 2].imshow(recon_3d[:, :, slice_z], cmap=self.cmap, vmin=vmin, vmax=vmax)
            axes[1, 2].set_title("Reconstruction Z")
            plt.colorbar(im6, ax=axes[1, 2], shrink=0.5)


            diff_x = recon_3d[slice_x, :, :] - orig_3d[slice_x, :, :]
            diff_y = recon_3d[:, slice_y, :] - orig_3d[:, slice_y, :]
            diff_z = recon_3d[:, :, slice_z] - orig_3d[:, :, slice_z]

            im7 = axes[2, 0].imshow(diff_x, cmap='RdBu_r', vmin=-abs(diff_x).max(), vmax=abs(diff_x).max())
            axes[2, 0].set_title("Difference X")
            plt.colorbar(im7, ax=axes[2, 0], shrink=0.5)


            im8 = axes[2, 1].imshow(diff_y, cmap='RdBu_r', vmin=-abs(diff_y).max(), vmax=abs(diff_y).max())
            axes[2, 1].set_title("Difference Y")
            plt.colorbar(im8, ax=axes[2, 1], shrink=0.5)


            im9 = axes[2, 2].imshow(diff_z, cmap='RdBu_r', vmin=-abs(diff_z).max(), vmax=abs(diff_z).max())
            axes[2, 2].set_title("Difference Z")
            plt.colorbar(im9, ax=axes[2, 2], shrink=0.5)


            for ax in axes.flatten():
                ax.set_xticks([])
                ax.set_yticks([])

            plt.tight_layout()

            fname = f"reconstruction_sample_{sample_idx+1}.png"
            self._save_and_show(fname, save_dir)


    def plot_detailed_3d_analysis(self, model, test_loader, device, grid_shape, save_dir=None, n_examples=3):
        """
        Create detailed analysis of 3D reconstructions

        :param model: trained model
        :param test_loader: test data loader
        :param device: torch device
        :param grid_shape: tuple (x_dim, y_dim, z_dim) for proper reshaping
        :param save_dir: directory to save plots
        :param n_examples: number of examples to analyze
        """

        model.eval()

        with torch.no_grad():
            for batch in test_loader:
                images, masks, conditioning = batch
                images = images.to(device)
                conditioning = conditioning.to(device)

                recon, mu, logvar = model(images, conditioning)

                originals = images.cpu().numpy()
                reconstructions = recon.cpu().numpy()
                break

        n_examples = min(n_examples, len(originals))

        # Create proper meshgrid for non-cubic grids
        x_dim, y_dim, z_dim = grid_shape
        x, y, z = np.meshgrid(range(x_dim), range(y_dim), range(z_dim), indexing='ij')
        coords = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

        for sample_idx in range(n_examples):
            fig = plt.figure(figsize=(20, 12))

            orig_data = originals[sample_idx].flatten()
            recon_data = reconstructions[sample_idx].flatten()

            # Get value ranges for consistent coloring
            vmin = min(orig_data.min(), recon_data.min())
            vmax = max(orig_data.max(), recon_data.max())

            # Original 3D plot
            ax1 = fig.add_subplot(3, 3, 1, projection='3d')
            scatter1 = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                                   c=orig_data, cmap=self.cmap, s=15, alpha=0.6,
                                   vmin=vmin, vmax=vmax)
            ax1.set_title(f'Original 3D Grid {sample_idx+1}', fontweight='bold')
            ax1.set_xlabel('X Index')
            ax1.set_ylabel('Y Index')
            ax1.set_zlabel('Z Index')
            plt.colorbar(scatter1, ax=ax1, shrink=0.5)

            # Reconstructed 3D plot
            ax2 = fig.add_subplot(3, 3, 2, projection='3d')
            scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                                   c=recon_data, cmap=self.cmap, s=15, alpha=0.6,
                                   vmin=vmin, vmax=vmax)
            ax2.set_title(f'Reconstructed 3D Grid {sample_idx+1}', fontweight='bold')
            ax2.set_xlabel('X Index')
            ax2.set_ylabel('Y Index')
            ax2.set_zlabel('Z Index')
            plt.colorbar(scatter2, ax=ax2, shrink=0.5)

            # Error plot
            ax3 = fig.add_subplot(3, 3, 3, projection='3d')
            error_data = recon_data - orig_data
            scatter3 = ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                                   c=error_data, cmap='RdBu_r', s=15, alpha=0.6)
            ax3.set_title(f'Error 3D Grid {sample_idx+1}', fontweight='bold')
            ax3.set_xlabel('X Index')
            ax3.set_ylabel('Y Index')
            ax3.set_zlabel('Z Index')
            plt.colorbar(scatter3, ax=ax3, shrink=0.5)

            # Error analysis
            error = recon_data - orig_data
            ax4 = fig.add_subplot(3, 3, 4)
            ax4.hist(error, bins=50, alpha=0.7, edgecolor='black')
            self._configure_axis(ax4, 'Reconstruction Error Distribution', 'Error', 'Frequency')

            # Correlation plot
            ax5 = fig.add_subplot(3, 3, 5)
            ax5.scatter(orig_data, recon_data, alpha=0.5, s=1)
            ax5.plot([orig_data.min(), orig_data.max()], [orig_data.min(), orig_data.max()], 'r--')
            self._configure_axis(ax5, 'Original vs Reconstructed', 'Original', 'Reconstructed')

            # Add correlation coefficient
            corr = np.corrcoef(orig_data, recon_data)[0, 1]
            ax5.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax5.transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Value distributions
            ax6 = fig.add_subplot(3, 3, 6)
            ax6.hist(orig_data, bins=30, alpha=0.7, label='Original', density=True)
            ax6.hist(recon_data, bins=30, alpha=0.7, label='Reconstructed', density=True)
            self._configure_axis(ax6, 'Value Distributions', 'Value', 'Density')
            ax6.legend()

            # Grid-based neighbor analysis (simplified spatial analysis)
            ax7 = fig.add_subplot(3, 3, 7)
            self._plot_grid_neighbor_analysis(
                orig_data, recon_data, grid_shape, ax7
            )

            # Residual analysis
            ax8 = fig.add_subplot(3, 3, 8)
            residuals = recon_data - orig_data
            ax8.scatter(range(len(residuals)), residuals, alpha=0.5, s=1)
            ax8.axhline(y=0, color='r', linestyle='--')
            self._configure_axis(ax8, 'Residuals vs Grid Index', 'Grid Point Index', 'Residual')

            plt.tight_layout()
            self._save_and_show(f'detailed_3d_analysis_example_{sample_idx + 1}.png', save_dir)

    def plot_3d_latent_space(self, model, test_loader, device, save_dir=None):
        """Visualize 3D latent space"""
        model.eval()

        latent_codes = []
        batch_indices = []  # Track batch indices instead of labels

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                images, masks, conditioning = batch
                images = images.to(device)
                conditioning = conditioning.to(device)

                mu, logvar = model.encode(images, conditioning)

                latent_codes.append(mu.cpu().numpy())
                # Use batch index as pseudo-label for coloring
                batch_indices.extend([batch_idx] * len(images))

        latent_codes = np.vstack(latent_codes)
        batch_indices_array = np.array(batch_indices)

        # Create multiple views of latent space
        fig, axes = plt.subplots(2, 3, figsize=self.figsize_large)

        # Projections of first few dimensions
        for i, (dim1, dim2) in enumerate([(0, 1), (0, 2), (1, 2)]):
            if i >= 3 or dim2 >= latent_codes.shape[1]:
                continue

            scatter = axes[0, i].scatter(latent_codes[:, dim1], latent_codes[:, dim2],
                                         c=batch_indices_array, cmap='tab10', alpha=0.6, s=5)
            self._configure_axis(axes[0, i], f'Latent Space (dims {dim1}-{dim2})',
                                 f'Latent Dimension {dim1}', f'Latent Dimension {dim2}')
            plt.colorbar(scatter, ax=axes[0, i], label='Batch')

        # Latent dimension analysis
        n_dims = min(5, latent_codes.shape[1])
        axes[1, 0].hist([latent_codes[:, i] for i in range(n_dims)],
                        bins=30, alpha=0.7, label=[f'Dim {i}' for i in range(n_dims)])
        axes[1, 0].set_title('Latent Dimension Distributions')
        axes[1, 0].legend()
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Frequency')

        # Latent space variance
        latent_vars = np.var(latent_codes, axis=0)
        axes[1, 1].bar(range(len(latent_vars)), latent_vars)
        self._configure_axis(axes[1, 1], 'Latent Dimension Variances', 'Dimension', 'Variance')

        # Correlation matrix of latent dimensions
        if latent_codes.shape[1] > 1:
            corr_matrix = np.corrcoef(latent_codes.T)
            im = axes[1, 2].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            self._configure_axis(axes[1, 2], 'Latent Dimension Correlations',
                                 'Dimension', 'Dimension', grid=False)
            plt.colorbar(im, ax=axes[1, 2])

        plt.tight_layout()
        self._save_and_show('3d_latent_space.png', save_dir)

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    def _save_and_show(self, filename, save_dir):
        """
        Helper method to save and display plots

        :param filename: name of file to save
        :param save_dir: directory to save to (if None, only shows plot)
        """
        if save_dir is not None:
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")

        plt.show()
        plt.close()

    def _plot_grid_neighbor_analysis(self, orig_data, recon_data, grid_shape, ax):
        """
        Analyze reconstruction quality based on grid neighbors

        :param orig_data: original flattened data
        :param recon_data: reconstructed flattened data
        :param grid_shape: tuple (x_dim, y_dim, z_dim)
        :param ax: matplotlib axis to plot on
        """
        x_dim, y_dim, z_dim = grid_shape

        # Reshape data
        orig_3d = orig_data.reshape(x_dim, y_dim, z_dim)
        recon_3d = recon_data.reshape(x_dim, y_dim, z_dim)

        # Calculate local gradients in each dimension
        orig_grad_x = np.abs(np.diff(orig_3d, axis=0)).mean()
        orig_grad_y = np.abs(np.diff(orig_3d, axis=1)).mean()
        orig_grad_z = np.abs(np.diff(orig_3d, axis=2)).mean()

        recon_grad_x = np.abs(np.diff(recon_3d, axis=0)).mean()
        recon_grad_y = np.abs(np.diff(recon_3d, axis=1)).mean()
        recon_grad_z = np.abs(np.diff(recon_3d, axis=2)).mean()

        # Plot comparison
        dims = ['X', 'Y', 'Z']
        orig_grads = [orig_grad_x, orig_grad_y, orig_grad_z]
        recon_grads = [recon_grad_x, recon_grad_y, recon_grad_z]

        x = np.arange(len(dims))
        width = 0.35

        ax.bar(x - width/2, orig_grads, width, label='Original', alpha=0.7)
        ax.bar(x + width/2, recon_grads, width, label='Reconstructed', alpha=0.7)

        self._configure_axis(ax, 'Spatial Smoothness Analysis', 'Dimension', 'Mean Absolute Gradient')
        ax.set_xticks(x)
        ax.set_xticklabels(dims)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')