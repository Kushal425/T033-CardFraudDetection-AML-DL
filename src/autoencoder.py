"""
src/autoencoder.py
==================
Deep Autoencoder models for anomaly detection — Team T033 Phase 2.
Author: Vani Rudra (Task P2_03)

Implements:
    - UndercompleteAE: symmetric encoder/decoder with configurable bottleneck.
    - DenoisingAE: same architecture, but training injects Gaussian noise.

Score convention: higher score == more anomalous (per-sample MSE reconstruction error).

The bottleneck must be *undercomplete* (latent_dim < d_in) so the network
cannot learn the identity mapping.  If latent_dim >= d_in, every input
(including fraud) would reconstruct perfectly, and reconstruction error
would fail as an anomaly score.

The `encode()` method exposes the bottleneck representation for downstream
consumers (Apoorva's latent GMM, Prachee's hybrid models).

Theory:
    Encoder f_θ: R^d → R^k  (k < d)
    Decoder g_φ: R^k → R^d
    Objective: min E‖x − g(f(x))‖²  (MSE over features)

    A linear AE with MSE loss spans the same subspace as PCA (Baldi & Hornik 1989).
    Nonlinear activations enable learning curved manifolds beyond the principal subspace.

    Denoising AE (Vincent et al. 2008 [VERIFY]):
        Train on corrupted input x̃ = x + σε, loss = ‖x − g(f(x̃))‖²
        Learns the score function ∇ log p(x) of the data manifold.
        No noise at inference — score is MSE on clean input.

References:
    - Hinton & Salakhutdinov 2006 [VERIFY]: deep autoencoders
    - Vincent et al. 2008, 2011 [VERIFY]: denoising autoencoders
    - Sakurada & Yairi 2014 [VERIFY]: AE for anomaly detection
    - An & Cho 2015 [VERIFY]: variational AE for anomaly detection
    - He et al. 2015 [VERIFY]: Kaiming/He initialization

Dependencies: torch (>= 2.0)
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
# Encoder / Decoder building blocks
# ═══════════════════════════════════════════════════════════════════════════════

_ACT_MAP = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "linear": nn.Identity,
}


def _build_layers(
    dims: list[int],
    act: str = "relu",
    dropout: float = 0.0,
    batchnorm: bool = False,
    final_activation: bool = True,
) -> nn.Sequential:
    """
    Build a stack of Linear → [BatchNorm] → Activation → [Dropout] layers.

    Parameters
    ----------
    dims : list of int
        Successive layer widths, e.g. [31, 24, 16] → two Linear layers.
    act : str
        Activation function name (relu, leaky_relu, tanh, elu).
    dropout : float
        Dropout probability after each activation (0.0 = no dropout).
    batchnorm : bool
        Whether to insert BatchNorm1d before each activation.
    final_activation : bool
        If False, the last layer has no activation/dropout/batchnorm (for decoder output).

    Returns
    -------
    nn.Sequential
    """
    act_cls = _ACT_MAP.get(act, nn.ReLU)
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = (i == len(dims) - 2)
        if is_last and not final_activation:
            break  # no activation on the final decoder output
        if batchnorm:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        layers.append(act_cls())
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


# ═══════════════════════════════════════════════════════════════════════════════
# UndercompleteAE
# ═══════════════════════════════════════════════════════════════════════════════

class UndercompleteAE(nn.Module):
    """
    Symmetric undercomplete autoencoder with MSE reconstruction loss.

    Architecture (default hidden=[24,16], latent_dim=8, d_in=31):
        Encoder: 31 → 24 → 16 → 8     (with ReLU activations)
        Decoder: 8 → 16 → 24 → 31     (mirror; linear output)

    The bottleneck at latent_dim=8 forces the network to learn a compressed
    representation of the legitimate transaction distribution.  Anomalous
    transactions (fraud) that deviate from this distribution will have higher
    reconstruction error.

    Parameters
    ----------
    d_in : int
        Input feature dimension.
    latent_dim : int
        Bottleneck dimension (must be < d_in for undercomplete).
    hidden : list of int
        Hidden layer widths between input and bottleneck.
    act : str
        Activation function: 'relu', 'leaky_relu', 'tanh', 'elu'.
    dropout : float
        Dropout probability (0.0 = disabled).
    batchnorm : bool
        Whether to use batch normalization.

    Methods
    -------
    encode(x) -> torch.Tensor
        Returns the latent representation (bottleneck output).
    decode(z) -> torch.Tensor
        Reconstructs from the latent representation.
    forward(x) -> torch.Tensor
        Full reconstruction: decode(encode(x)).
    """

    def __init__(
        self,
        d_in: int,
        latent_dim: int = 8,
        hidden: Sequence[int] = (24, 16),
        act: str = "relu",
        dropout: float = 0.0,
        batchnorm: bool = False,
    ):
        super().__init__()
        if latent_dim >= d_in:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be < d_in ({d_in}) "
                "for an undercomplete autoencoder."
            )
        self.d_in = d_in
        self.latent_dim = latent_dim
        self.hidden = list(hidden)
        self.act = act
        self.dropout = dropout
        self.batchnorm = batchnorm

        # Encoder: d_in → hidden[0] → … → hidden[-1] → latent_dim
        enc_dims = [d_in] + list(hidden) + [latent_dim]
        self.encoder = _build_layers(
            enc_dims, act=act, dropout=dropout,
            batchnorm=batchnorm, final_activation=True,
        )

        # Decoder: latent_dim → hidden[-1] → … → hidden[0] → d_in
        dec_dims = [latent_dim] + list(reversed(hidden)) + [d_in]
        self.decoder = _build_layers(
            dec_dims, act=act, dropout=dropout,
            batchnorm=batchnorm, final_activation=False,
        )

        # Initialize weights: He/Kaiming for ReLU layers, Xavier for others
        self._init_weights()

    def _init_weights(self):
        """Apply He/Kaiming initialization for ReLU-family activations."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.act in ("relu", "leaky_relu", "elu"):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input features to the latent (bottleneck) representation.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, d_in)
            Scaled input features (float32).

        Returns
        -------
        torch.Tensor, shape (batch, latent_dim)
            Latent codes (float32).

        Notes for downstream consumers (Apoorva's latent GMM):
            - Input must be scaled with the same StandardScaler as training
              (fit on train-legit only, via ``get_features("base")``).
            - Output shape: (n_samples, latent_dim).
            - dtype: float32.
            - Transaction ordering: same as the corresponding split array
              from get_features() (chronological order).
            - Model loading: ``model = UndercompleteAE(d_in, latent_dim, ...)``
              then ``model.load_state_dict(torch.load(path, weights_only=True))``.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from the latent representation."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode → decode."""
        return self.decode(self.encode(x))

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Return the model hyperparameters as a JSON-serializable dict."""
        return {
            "model_type": "UndercompleteAE",
            "d_in": self.d_in,
            "latent_dim": self.latent_dim,
            "hidden": self.hidden,
            "act": self.act,
            "dropout": self.dropout,
            "batchnorm": self.batchnorm,
            "n_params": self.count_parameters(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DenoisingAE
# ═══════════════════════════════════════════════════════════════════════════════

class DenoisingAE(UndercompleteAE):
    """
    Denoising Autoencoder — same architecture as UndercompleteAE, but
    training adds Gaussian noise to the input: x̃ = x + σ·ε.

    The loss compares the reconstruction to the CLEAN input (not the noisy one):
        L = MSE(x, decode(encode(x̃)))

    At inference, no noise is added:
        score = MSE(x, decode(encode(x)))

    This forces the network to learn robust features of the data manifold
    and acts as a regularizer that prevents overfitting to individual samples.

    Parameters
    ----------
    noise_sigma : float
        Standard deviation of Gaussian noise added during training.
        Higher σ → stronger regularization but harder training.
        Typical range: 0.05 – 0.5 for standardized features.
    (All other parameters inherited from UndercompleteAE.)
    """

    def __init__(
        self,
        d_in: int,
        latent_dim: int = 8,
        hidden: Sequence[int] = (24, 16),
        act: str = "relu",
        dropout: float = 0.0,
        batchnorm: bool = False,
        noise_sigma: float = 0.1,
    ):
        super().__init__(
            d_in=d_in, latent_dim=latent_dim, hidden=hidden,
            act=act, dropout=dropout, batchnorm=batchnorm,
        )
        self.noise_sigma = noise_sigma

    def forward_noisy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Gaussian noise injection (training only).

        Adds noise to input before encoding, but the MSE loss is
        computed against the CLEAN input x (not the noisy x̃).

        Returns the reconstruction of the noisy input.
        """
        noise = torch.randn_like(x) * self.noise_sigma
        x_noisy = x + noise
        return self.decode(self.encode(x_noisy))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clean forward pass (no noise, used at inference)."""
        return self.decode(self.encode(x))

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg["model_type"] = "DenoisingAE"
        cfg["noise_sigma"] = self.noise_sigma
        return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring utilities
# ═══════════════════════════════════════════════════════════════════════════════

def compute_reconstruction_mse(
    model: UndercompleteAE,
    X: np.ndarray,
    batch_size: int = 2048,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute per-sample MSE reconstruction error.

    Parameters
    ----------
    model : UndercompleteAE or DenoisingAE
        Trained autoencoder (must be in eval mode).
    X : np.ndarray, shape (n_samples, d_in)
        Scaled input features (float32).
    batch_size : int
        Batch size for inference.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Per-sample MSE (mean over features). Higher = more anomalous.
    """
    model.eval()
    model = model.to(device)
    scores = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            x_hat = model(x_batch)
            mse = ((x_batch - x_hat) ** 2).mean(dim=1)  # mean over features
            scores.append(mse.cpu().numpy())

    return np.concatenate(scores).astype(np.float32)


def compute_per_feature_error(
    model: UndercompleteAE,
    X: np.ndarray,
    batch_size: int = 2048,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute per-feature squared reconstruction error.

    Returns
    -------
    np.ndarray, shape (n_samples, d_in)
        Per-feature squared error for explanation / diagnostics.
    """
    model.eval()
    model = model.to(device)
    errors = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            x_hat = model(x_batch)
            err = (x_batch - x_hat) ** 2
            errors.append(err.cpu().numpy())

    return np.concatenate(errors).astype(np.float32)


def encode_dataset(
    model: UndercompleteAE,
    X: np.ndarray,
    batch_size: int = 2048,
    device: str = "cpu",
) -> np.ndarray:
    """
    Encode a full dataset through the encoder to get latent representations.

    This is the primary interface for Apoorva's latent GMM.

    Parameters
    ----------
    model : UndercompleteAE or DenoisingAE
        Trained autoencoder (must be in eval mode).
    X : np.ndarray, shape (n_samples, d_in)
        Scaled input features (float32, from get_features("base")).
    batch_size : int
        Batch size for inference.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    np.ndarray, shape (n_samples, latent_dim), dtype float32
        Latent codes in the same transaction order as the input.

    Interface specification for Apoorva:
        Input shape: (n_samples, 31) for base features
        Latent dimension: model.latent_dim (default 8)
        Output shape: (n_samples, latent_dim)
        Output dtype: float32
        Model loading:
            model = UndercompleteAE(d_in=31, latent_dim=8, hidden=[24, 16])
            model.load_state_dict(torch.load('results/models/ae_best.pt',
                                             weights_only=True))
            model.eval()
            codes = encode_dataset(model, X)
        Transaction ordering: same as get_features("base") split arrays
            (chronological order, matching split_manifest.json).
    """
    model.eval()
    model = model.to(device)
    codes = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            z = model.encode(x_batch)
            codes.append(z.cpu().numpy())

    return np.concatenate(codes).astype(np.float32)


def load_autoencoder(
    weights_path: str | Path,
    config_path: str | Path | None = None,
    device: str = "cpu",
) -> UndercompleteAE:
    """
    Load a saved UndercompleteAE or DenoisingAE model from disk.

    Parameters
    ----------
    weights_path : str or Path
        Path to saved state_dict (e.g. ``results/models/ae_best.pt``).
    config_path : str or Path, optional
        Path to JSON config (e.g. ``results/models/ae_config.json``).
        If None, looks for ``<stem>_config.json`` or ``ae_config.json`` in
        the same directory.
    device : str
        Target device ('cpu', 'cuda', or 'mps').

    Returns
    -------
    UndercompleteAE or DenoisingAE in eval() mode.
    """
    weights_path = Path(weights_path)
    if config_path is None:
        candidate = weights_path.with_name(weights_path.stem.replace("_best", "") + "_config.json")
        config_path = candidate if candidate.exists() else weights_path.with_name("ae_config.json")
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as fh:
        cfg = json.load(fh)

    model_type = cfg.get("model_type", "UndercompleteAE")
    common_kwargs = {
        "d_in": int(cfg["d_in"]),
        "latent_dim": int(cfg["latent_dim"]),
        "hidden": list(cfg["hidden"]),
        "act": cfg.get("act", "relu"),
        "dropout": float(cfg.get("dropout", 0.0)),
        "batchnorm": bool(cfg.get("batchnorm", False)),
    }
    if model_type == "DenoisingAE":
        model: UndercompleteAE = DenoisingAE(
            **common_kwargs,
            noise_sigma=float(cfg.get("noise_sigma", 0.1)),
        )
    else:
        model = UndercompleteAE(**common_kwargs)

    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _run_module_self_tests() -> None:
    """Fast unit tests for architecture, encode(), MSE scoring, and save/load."""
    import tempfile

    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    X_syn = rng.standard_normal((64, 31)).astype(np.float32)

    ae = UndercompleteAE(d_in=31, latent_dim=8, hidden=[24, 16], act="relu", dropout=0.1)
    dae = DenoisingAE(d_in=31, latent_dim=8, hidden=[24, 16], noise_sigma=0.1)

    ae.eval()
    z = encode_dataset(ae, X_syn)
    assert z.shape == (64, 8) and z.dtype == np.float32
    mse = compute_reconstruction_mse(ae, X_syn)
    assert mse.shape == (64,) and mse.dtype == np.float32 and np.all(mse >= 0)
    per_feat = compute_per_feature_error(ae, X_syn)
    assert per_feat.shape == (64, 31)
    np.testing.assert_allclose(per_feat.mean(axis=1), mse, rtol=1e-5, atol=1e-6)

    # Test save/load equivalence
    with tempfile.TemporaryDirectory() as tmp:
        w_path = Path(tmp) / "ae_best.pt"
        c_path = Path(tmp) / "ae_config.json"
        torch.save(ae.state_dict(), w_path)
        c_path.write_text(json.dumps(ae.get_config()))
        loaded = load_autoencoder(w_path, c_path)
        z2 = encode_dataset(loaded, X_syn)
        mse2 = compute_reconstruction_mse(loaded, X_syn)
        np.testing.assert_allclose(z, z2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(mse, mse2, rtol=1e-6, atol=1e-6)

    # Test DAE noisy forward runs
    dae.train()
    x_t = torch.tensor(X_syn)
    out_noisy = dae.forward_noisy(x_t)
    assert out_noisy.shape == (64, 31)
    print("src/autoencoder.py self-tests PASSED.")


if __name__ == "__main__":
    _run_module_self_tests()



