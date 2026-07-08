import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F

# ============================================================================
# Deconvolution sharpening
# ============================================================================

class PixelShuffle3d(nn.Module):
    """3D PixelShuffle upsampling layer for fine spatial details."""
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale = scale_factor

    def forward(self, x):
        batch, c, d, h, w = x.size()
        r = self.scale
        assert c % (r**3) == 0, f"Channels {c} must be divisible by scale^3 ({r**3})"

        out_c = c // (r**3)
        # Split channel dimension into spatial factor components
        x = x.view(batch, out_c, r, r, r, d, h, w)
        # Permute to interleave dimensions into spatial coordinates
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        # Collapse back to 5D tensor
        return x.view(batch, out_c, d * r, h * r, w * r)





# ============================================================================
# 3D Convolutional Encoder
# ============================================================================

class Conv3DEncoder(nn.Module):
    """3D Convolutional encoder with strided convolutions"""

    def __init__(self, grid_shape, latent_size, class_size, dropout_rate=0.1):
        """
        :param grid_shape: tuple (x_dim, y_dim, z_dim) e.g., (150, 37, 37)
        :param latent_size: size of latent space
        :param class_size: number of classes for conditioning
        :param dropout_rate: dropout probability
        """
        super().__init__()

        self.grid_shape = grid_shape
        self.latent_size = latent_size

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.GroupNorm(8, 16)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.GroupNorm(8, 32)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.GroupNorm(8, 64)


        target_x = self._calc_conv_output_size(grid_shape[0], num_layers=3, kernel=3, stride=2, padding=1)
        target_y = self._calc_conv_output_size(grid_shape[1], num_layers=3, kernel=3, stride=2, padding=1)
        target_z = self._calc_conv_output_size(grid_shape[2], num_layers=3, kernel=3, stride=2, padding=1)

        self.bottleneck_shape = (target_x, target_y, target_z)
        # self.pool = nn.AdaptiveAvgPool3d(self.bottleneck_shape)
        self.flattened_size = 64 * target_x * target_y * target_z

        self.dropout = nn.Dropout(dropout_rate)

        self.class_proj = nn.Linear(class_size, self.flattened_size) if class_size > 0 else None

        combined_size = self.flattened_size * (2 if class_size > 0 else 1)
        self.mu_layer = nn.Linear(combined_size, latent_size)
        self.logvar_layer = nn.Linear(combined_size, latent_size)

        self._init_weights()

    @staticmethod
    def _calc_conv_output_size(input_size, num_layers, kernel, stride, padding):
        size = input_size
        for _ in range(num_layers):
            size = (size + 2 * padding - kernel) // stride + 1
        return size

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, c):
        """
        Forward pass
        :param x: input tensor
        :param c: class labels for conditioning
        :return: mu, logvar
        """
        batch_size = x.size(0)
        # x = x.view(batch_size, 1, self.grid_shape[0], self.grid_shape[1], self.grid_shape[2])

        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))

        # x = self.pool(x)
        x = x.view(batch_size, -1)
        x = self.dropout(x)

        if c is not None and self.class_proj is not None:
            x = torch.cat([x, self.class_proj(c)], dim=1)

        mu = self.mu_layer(x)
        logvar = torch.clamp(self.logvar_layer(x), min=-10, max=10)

        return mu, logvar

# ============================================================================
# 3D Convolutional Decoder
# ============================================================================

class Conv3DDecoder(nn.Module):
    """3D Convolutional decoder — mirror of Conv3DEncoder."""

    def __init__(self, latent_size, grid_shape, class_size, bottleneck_shape, dropout_rate=0.1):
        """
        :param latent_size: size of latent space
        :param grid_shape: tuple (x_dim, y_dim, z_dim) e.g., (150, 37, 37)
        :param class_size: number of classes for conditioning
        :param bottleneck_shape: spatial dims at encoder bottleneck (x, y, z)
        :param dropout_rate: dropout probability
        """
        super().__init__()

        self.grid_shape = grid_shape

        x_dim, y_dim, z_dim  = grid_shape
        self.n_grid_points   = x_dim * y_dim * z_dim
        self.start_x, self.start_y, self.start_z = bottleneck_shape

        combined_latent = latent_size + class_size
        start_size      = self.start_x * self.start_y * self.start_z

        # Project latent vector back up to bottleneck volume
        self.conv_proj = nn.Sequential(
            nn.Linear(combined_latent, 64 * start_size),
            nn.GELU()
        )

        # --- Layer 1: Upsample from 64 channels to 32 channels ---
        self.up1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=32 * 8, kernel_size=3, padding=1),
            PixelShuffle3d(scale_factor=2),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )

        # --- Layer 2: Upsample from 32 channels to 16 channels ---
        self.up2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=16 * 8, kernel_size=3, padding=1),
            PixelShuffle3d(scale_factor=2),
            nn.GroupNorm(8, 16),
            nn.GELU()
        )

        # --- Layer 3: Upsample from 16 channels down to 1 target channel ---
        self.up3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=1 * 8, kernel_size=3, padding=1),
            PixelShuffle3d(scale_factor=2),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
            # Final convolution refines the boundaries without upsampling
        )

        #self.deconv1 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        # self.bn1     = nn.GroupNorm(8, 32)
        # self.deconv2 = nn.ConvTranspose3d(in_channels=32, out_channels=16,  kernel_size=3, stride=2, padding=1)
        # self.bn2     = nn.GroupNorm(8, 16)
        # self.deconv3 = nn.ConvTranspose3d(in_channels=16,  out_channels=1,  kernel_size=3, stride=2, padding=1)

        #self.final_resize = nn.AdaptiveMaxPool3d((x_dim, y_dim, z_dim))
        self.grid_bias    = nn.Parameter(torch.zeros(self.n_grid_points))
        self.dropout      = nn.Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose3d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.grid_bias, -0.001)

    # def forward(self, z, c):
    #     """
    #     :param z: latent vector (batch, latent_size)
    #     :param c: class conditioning (batch, class_size)
    #     :return: reconstructed grid (batch, n_grid_points)
    #     """
    #     batch_size = z.size(0)
    #
    #     x = self.conv_proj(torch.cat([z, c], dim=1) if c is not None else z)
    #     x = x.view(batch_size, 64, self.start_x, self.start_y, self.start_z)
    #
    #     x = F.gelu(self.bn1(self.deconv1(x)))
    #     x = self.dropout(x)
    #     x = F.gelu(self.bn2(self.deconv2(x)))
    #     x = self.dropout(x)
    #     x = self.deconv3(x)
    #
    #     #x = self.final_resize(x)
    #     x = F.interpolate(x, size=self.grid_shape, mode='trilinear', align_corners=False)
    #     #x = x.view(batch_size, -1)
    #
    #     return torch.sigmoid(x.view(batch_size, -1))
    #     #return x.clamp(0,1) if not self.training else x

    def forward(self, z, c):
        batch_size = z.size(0)

        # Combine latent vector and conditioning features
        x = self.conv_proj(torch.cat([z, c], dim=1) if c is not None else z)
        x = x.view(batch_size, 64, self.start_x, self.start_y, self.start_z)

        # Clean, sharp upsampling blocks
        x = self.up1(x)
        x = self.dropout(x)
        x = self.up2(x)
        x = self.dropout(x)
        x = self.up3(x)

        # PixelShuffle3d upsamples by a fixed 8x per layer, but the encoder's
        # strided convs floor-divide the input size, so the round trip only
        # matches grid_shape when every dim is a multiple of 8 at the input.
        # Resize back to the exact target grid to correct for that.
        if x.shape[-3:] != tuple(self.grid_shape):
            x = F.interpolate(x, size=self.grid_shape, mode='trilinear', align_corners=False)

        return torch.sigmoid(x)


# ============================================================================
# Full model
# ============================================================================
class CVAE(nn.Module):
    """Standard 3D CVAE for Turbulent Flow."""

    def __init__(self, grid_shape, latent_size, class_size, dropout_rate=0.1):
        """
        :param grid_shape: tuple (x_dim, y_dim, z_dim) e.g., (150, 37, 37)
        :param latent_size: size of latent space
        :param class_size: number of classes for conditioning
        :param dropout_rate: dropout probability
        """
        super().__init__()

        self.latent_size  = latent_size
        self.grid_shape   = grid_shape
        self.class_size   = class_size
        self.dropout_rate = dropout_rate

        self.encoder = Conv3DEncoder(
            grid_shape,
            latent_size,
            class_size,
            dropout_rate=dropout_rate
        )

        self.decoder = Conv3DDecoder(
            latent_size,
            grid_shape,
            class_size,
            bottleneck_shape=self.encoder.bottleneck_shape,
            dropout_rate=dropout_rate
        )

    def encode(self, x, c):
        """Encode input to latent distribution parameters (mu, logvar)."""
        return self.encoder(x, c)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling from latent distribution."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def decode(self, z, c):
        """Decode latent sample to reconstruction."""
        return self.decoder(z, c)

    def forward(self, x, c):
        """
        :param x: input data (batch, x, y, z)
        :param c: class conditioning (batch, class_size)
        :return: reconstruction, mu, logvar
        """
        mu, logvar = self.encode(x, c)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z, c)
        return recon, mu, logvar

    def sample_prior(self, n_samples, c):
        """
        Sample from prior for generation.
        :param n_samples: number of samples to generate
        :param c: class conditioning (n_samples, class_size)
        :return: dictionary with reconstructions and latent samples
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            z     = torch.randn(n_samples, self.latent_size, device=device)
            recon = self.decode(z, c.to(device))

        return {
            'reconstructions': recon.cpu().numpy(),
            'latent_samples':  z.cpu().numpy()
        }

    def get_latent_representation(self, x, c):
        """
        Get deterministic latent representation (posterior mean).
        :param x: input data
        :param c: class conditioning
        :return: latent representation (mu)
        """
        self.eval()

        with torch.no_grad():
            mu, _ = self.encode(x, c)

        return mu.cpu().numpy()