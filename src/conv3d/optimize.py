import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'

from src.shared.data_loader import CVAESetup
from src.shared.seed import set_seed
from src.conv3d.model import CVAE
from src.conv3d.train import train_model

SEED = 42
TRIAL_EPOCHS = 60


def build_objective(train_loader, test_loader, grid_shape, n_modes, device, trial_epochs):
    def objective(trial):
        beta_end = trial.suggest_float('beta_end', 0.3, 0.7)
        extreme_weight = trial.suggest_float('extreme_weight', 0.5, 2.0)
        renyi_weight = trial.suggest_float('renyi_weight', 0.0, 0.5)
        grad_weight = trial.suggest_float('grad_weight', 0.0, 2.0)

        # Fixed seed per trial: init/dropout/shuffle noise held constant so
        # trial-to-trial differences are attributable to the loss weights.
        set_seed(SEED)

        model = CVAE(
            grid_shape=grid_shape,
            latent_size=128,
            class_size=n_modes,
            dropout_rate=0.05,
        ).to(device)

        _, loss_dict = train_model(
            model, train_loader, test_loader, device, grid_shape,
            epochs=trial_epochs, beta_start=0.05, beta_end=beta_end,
            extreme_weight=extreme_weight, renyi_weight=renyi_weight,
            grad_weight=grad_weight, loss='l1', trial=trial,
        )

        return loss_dict['test_total'][-1]

    return objective


def main(n_trials=30, seed=SEED, trial_epochs=TRIAL_EPOCHS):
    set_seed(seed)

    pod_file = str(DATA_DIR / 'u3_pod_basis.npz')
    pod_data = np.load(pod_file)
    n_modes = int(pod_data['n_modes'])

    setup = CVAESetup(
        numpy_file=str(DATA_DIR / 'skewnormal_gev_u3.npy'),
        batch_size=8,
        pod_file=pod_file,
    )
    device = setup.setup_device()
    train_loader, test_loader, grid_shape, grid_size = setup.load_data()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"optuna_{timestamp}_seed{seed}"
    os.makedirs(run_dir, exist_ok=True)

    study = optuna.create_study(
        study_name=f"cvae_loss_weights_{timestamp}",
        storage=f"sqlite:///{run_dir}/study.db",
        direction='minimize',
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_warmup_steps=20, interval_steps=5),
        load_if_exists=True,
    )

    objective = build_objective(train_loader, test_loader, grid_shape, n_modes, device, trial_epochs)
    study.optimize(objective, n_trials=n_trials)

    print("\nBest trial:")
    print(f"  value: {study.best_trial.value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v:.4f}")

    with open(os.path.join(run_dir, 'best_params.json'), 'w') as f:
        json.dump({
            'value': study.best_trial.value,
            'params': study.best_trial.params,
            'trial_epochs': trial_epochs,
            'n_trials': n_trials,
            'seed': seed,
        }, f, indent=2)

    study.trials_dataframe().to_csv(os.path.join(run_dir, 'trials.csv'), index=False)

    print(f"\nStudy saved to: {run_dir}")
    print(f"  - best_params.json (best trial's weights)")
    print(f"  - trials.csv (all trials)")
    print(f"  - study.db (resumable Optuna storage)")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial-epochs', type=int, default=TRIAL_EPOCHS)
    args = parser.parse_args()

    main(n_trials=args.n_trials, seed=args.seed, trial_epochs=args.trial_epochs)
