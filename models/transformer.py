"""Time-series Transformer models and positional-encoding implementations.
Conventions:
- v1 uses Conv1d patching -> Transformer -> flatten classification, with input shape (B, C, L).
- v2 uses a multi-scale Conv front-end -> Transformer -> attention pooling, with input shape (B, L) or (B, 1, L) depending on the caller.
Main components: Patch embedding / PositionalEncoding / Transformer encoder / Attention pooling.
"""

import torch
import torch.nn as nn
import math
from warnings import deprecated


@deprecated("TimeSeriesTransformer_v1 is deprecated; please use TimeSeriesTransformer_v2")
class TimeSeriesTransformer_v1(nn.Module):
    """Time-series classifier with Patch + Transformer.
    Design highlights: Conv1d for patching, learned positional embeddings, a Pre-Norm Transformer encoder, and a final flattening plus linear projection to the number of classes.
    Input: x (B, C, L); output: logits (B, num_classes).
    """

    def __init__(
        self,
        num_classes: int,  # number of classes
        seq_len: int,  # original input sequence length
        n_layers: int = 3,  # number of encoder layers
        d_model: int = 128,  # hidden dimension
        n_heads: int = 8,  # number of attention heads
        d_ff: int = 256,  # intermediate FFN dimension
        dropout: float = 0.1,
        patch_size: int = 16,  # [A] patch length
        stride: int = 8,  # [A] patch stride
    ):
        super().__init__()

        # ---------------------------------------------------------
        # A. Patching layer
        # Use Conv1d to implement patch embedding.
        # Input: (Batch, c_in, seq_len) -> Output: (Batch, d_model, n_patches)
        # ---------------------------------------------------------
        self.patch_embedding = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=stride,
        )

        # Compute the sequence length after patching (number of patches)
        # Formula: floor((L - K) / S) + 1
        self.n_patches = (seq_len - patch_size) // stride + 1

        # ---------------------------------------------------------
        # B. Positional encoding
        # Use learnable positional embeddings.
        # This works well for fixed-length tasks and can automatically learn
        # absolute and relative positional dependencies in the sequence.
        # Shape: (1, n_patches, d_model)
        # ---------------------------------------------------------
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02
        )
        self.dropout = nn.Dropout(dropout)

        # ---------------------------------------------------------
        # C. Transformer encoder backbone
        # Pre-Norm structure is usually more stable (norm_first=True)
        # ---------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Recommended: Pre-Norm structure
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
            mask_check=False,
        )
        self._init_transformer(self.transformer_encoder)
        # ---------------------------------------------------------
        # Classification head
        # ---------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * self.n_patches, num_classes),
        )

    @staticmethod
    def _init_transformer(transformer_encoder: nn.TransformerEncoder):
        # Only traverse modules inside the TransformerEncoder.
        for m in transformer_encoder.modules():
            if isinstance(m, nn.Linear):
                # Classic Xavier initialization, suitable for Transformer residual structure.
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                # Ensure the Norm layer starts as an identity transform.
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    @torch.compile
    def forward(self, x):
        """
        x shape: (Batch, c_in, seq_len)
        """
        # 1. [A] Patching
        # x: (Batch, c_in, seq_len) -> (Batch, d_model, n_patches)
        x = self.patch_embedding(x)

        # 2. Reshape dimensions to suit the Transformer
        # (Batch, d_model, n_patches) -> (Batch, n_patches, d_model)
        x = x.transpose(1, 2)

        # 3. [B] Add positional encoding
        # Use broadcasting to add (1, n, d) to (b, n, d)
        x = x + self.pos_embedding
        x = self.dropout(x)

        # 4. [C] Transformer backbone
        # Output: (Batch, n_patches, d_model)
        x = self.transformer_encoder(x)

        # 5. [D] Mean pooling
        # Average over the time dimension (dim=1) to capture global sequence features
        # (Batch, n_patches, d_model) -> (Batch, d_model)
        # x = torch.mean(x, dim=1)
        x = x.transpose(
            1, 2
        ).contiguous()  # Transpose back to (Batch, d_model, n_patches) for the classifier head.

        # 6. Classification
        output = self.classifier(x)

        return output


class PositionalEncoding(nn.Module):
    """Classic fixed sinusoidal positional encoding with dropout. Suitable for inputs of shape (B, seq_len, d_model).
    Not updated by gradients (registered as a buffer).
    """

    pe: torch.Tensor

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the positional-encoding matrix.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as a buffer, so it does not participate in gradient updates.
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer_v2(nn.Module):
    """Time-series classifier with a multi-scale Conv front-end + Transformer + attention pooling.
    Design highlights: conv3/conv7/conv15 extract multi-scale features -> downsampling -> Transformer encoding -> attention-weighted pooling -> linear classification.
    Input: usually (B, L) or (B, 1, L) depending on the caller; output logits (B, num_classes).
    """

    def __init__(
        self,
        num_classes: int,  # number of classes
        seq_len: int = 700,  # original input sequence length
        n_layers: int = 3,  # number of encoder layers
        d_model: int = 128,  # hidden dimension
        n_heads: int = 8,  # number of attention heads
        d_ff: int = 256,  # intermediate FFN dimension
        dropout: float = 0.0,
        target_tokens: int = 1000,
    ):
        super().__init__()

        # ---------- 1) Multi-scale Conv frontend (accelerated version) ----------
        self.conv3 = nn.Conv1d(1, d_model // 4, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(1, d_model // 4, kernel_size=7, padding=3)
        self.conv15 = nn.Conv1d(1, d_model // 2, kernel_size=15, padding=7)

        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.ReLU(inplace=True)

        # Accelerated: downsample after Conv, 500 -> 250 tokens
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        # self.pool = nn.AdaptiveAvgPool1d(target_tokens)

        # ---------- 2) Positional encoding ----------
        self.pos = PositionalEncoding(d_model, dropout, max_len=seq_len // 2)
        # self.pos = PositionalEncoding(d_model, dropout, max_len=target_tokens)

        # ---------- 3) Transformer encoder ----------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,  # Reduce FFN width to speed things up further.
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
            mask_check=False,
        )

        # ---------- 4) POI attention pooling ----------
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

        # ---------- 5) Logit scaling ----------
        self.logit_scale = nn.Parameter(torch.ones(1))
        self.classifier = nn.Linear(d_model, num_classes)

    @torch.compile
    def forward(self, x):
        # x: (B, L=500)
        # x = x.unsqueeze(1)  # (B,1,500)

        # Multi-scale conv
        f = torch.cat(
            [
                self.conv3(x),
                self.conv7(x),
                self.conv15(x),
            ],
            dim=1,
        )  # (B, C=256, 500)

        # Accelerated: downsample to 250 tokens
        f = self.pool(f)  # (B, 256, 250)
        f = self.act(self.bn(f))
        f = f.transpose(1, 2)  # (B, 250, 256)

        # Transformer
        f = self.pos(f)
        f = self.encoder(f)  # (B, 250, 256)

        # Attention pooling
        attn = self.attn_pool(f).squeeze(-1)  # (B, 250)
        attn = torch.softmax(attn, dim=1)
        z = torch.sum(f * attn.unsqueeze(-1), dim=1)  # (B, 256)

        logits = self.classifier(z) / self.logit_scale.clamp(min=1e-2)
        return logits
