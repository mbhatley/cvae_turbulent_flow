import numpy as np
import torch
from torch.optim import AdamW

from src.shared.loss_function import compute_loss
from src.shared.scheduler import get_scheduler


def train_model(model, train_loader, test_loader, device, grid_shape,
                epochs=150, initial_lr=1e-4, beta_start=0.1, beta_end=0.5):

    optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    scheduler, step_type = get_scheduler(optimizer, epochs, 100)

    train_losses, test_losses, learning_rates = [], [], []
    train_recon_losses, train_kl_losses = [], []
    test_recon_losses, test_kl_losses = [], []

    for epoch in range(1, epochs + 1):
        # KL annealing
        beta = beta_start + (beta_end - beta_start) * min(epoch / (epochs * 0.5), 1.0)

        # Training
        model.train()
        epoch_train_loss = 0
        epoch_train_recon = 0
        epoch_train_kl = 0

        for batch_data in train_loader:
            data, mask, labels = [item.to(device) for item in batch_data]
            conditioning = torch.zeros(data.size(0), 1, device=device)

            optimizer.zero_grad()
            recon, mu, logvar = model(data, conditioning)
            total_loss, recon_loss, kl_loss = compute_loss(recon, data, mu, logvar, grid_shape, beta=beta)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += total_loss.item()
            epoch_train_recon += recon_loss.item()
            epoch_train_kl += kl_loss.item()

        # Validation
        model.eval()
        epoch_test_loss = 0
        epoch_test_recon = 0
        epoch_test_kl = 0

        with torch.no_grad():
            for batch_data in test_loader:
                data, mask, labels = [item.to(device) for item in batch_data]
                conditioning = torch.zeros(data.size(0), 1, device=device)
                recon, mu, logvar = model(data, conditioning)
                test_loss, recon_loss, kl_loss = compute_loss(recon, data, mu, logvar, grid_shape, beta=beta)
                epoch_test_loss += test_loss.item()
                epoch_test_recon += recon_loss.item()
                epoch_test_kl += kl_loss.item()

        # Record metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_test_loss = epoch_test_loss / len(test_loader)
        avg_train_recon = epoch_train_recon / len(train_loader)
        avg_train_kl = epoch_train_kl / len(train_loader)
        avg_test_recon = epoch_test_recon / len(test_loader)
        avg_test_kl = epoch_test_kl / len(test_loader)
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_recon_losses.append(avg_train_recon)
        train_kl_losses.append(avg_train_kl)
        test_recon_losses.append(avg_test_recon)
        test_kl_losses.append(avg_test_kl)
        learning_rates.append(current_lr)

        if step_type == 'epoch':
            scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d}: Train={avg_train_loss:.4f}, Test={avg_test_loss:.4f}, LR={current_lr:.2e}')

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    loss_dict = {
        'train_total': train_losses,
        'test_total': test_losses,
        'train_recon': train_recon_losses,
        'train_kl': train_kl_losses,
        'test_recon': test_recon_losses,
        'test_kl': test_kl_losses,
        'learning_rates': learning_rates
    }

    return model, loss_dict
