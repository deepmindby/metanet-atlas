"""MetaNet Architecture for Pre-computed Features

This module implements a modified version of MetaNet that works with
pre-computed features directly instead of processing images through
CLIP encoders, significantly accelerating training and inference.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from src.distributed import is_main_process


class MetaNet(nn.Module):
    """Meta-Net network for generating task vector composition coefficients from sample features"""

    def __init__(self, input_dim, output_dim, hidden_dim=None):
        """Initialize Meta-Net

        Parameters:
        ----------
        input_dim: int
            Input feature dimension
        output_dim: int
            Output dimension, equal to the number of task vectors
        hidden_dim: int, optional
            Hidden layer dimension, defaults to 1/4 of the input dimension
        """
        super().__init__()

        # If hidden layer dimension is not specified, use 1/4 of the input dimension
        if hidden_dim is None:
            hidden_dim = max(input_dim // 4, output_dim)

        # Two-layer bottleneck structure: Linear-ReLU-Linear
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Initialize with small normal distribution values for better training stability
        nn.init.normal_(self.net[0].weight, mean=0.0, std=0.01)
        nn.init.normal_(self.net[0].bias, mean=0.0, std=0.01)

    def forward(self, x):
        """Forward propagation

        Parameters:
        ----------
        x: Tensor [batch_size, input_dim]
            Sample features

        Returns:
        ----------
        coefficients: Tensor [batch_size, output_dim]
            Task vector composition coefficients for each sample
        """
        return self.net(x)


class PrecomputedMetaNet(nn.Module):
    """MetaNet for pre-computed features that directly applies task vectors
    to pre-computed features rather than re-encoding images
    """

    def __init__(self, feature_dim, task_vectors, blockwise=False, enable_causal=False, top_k_ratio=0.1):
        """Initialize PrecomputedMetaNet

        Parameters:
        ----------
        feature_dim: int
            Dimension of the pre-computed feature vectors
        task_vectors: List of task vectors
            List of task vectors for composition
        blockwise: bool
            Whether to use different coefficients for each parameter block
        enable_causal: bool
            Whether to enable causal intervention
        top_k_ratio: float
            Ratio of top blocks to use for intervention
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.task_vectors = task_vectors
        self.blockwise = blockwise
        self.enable_causal = enable_causal
        self.top_k_ratio = top_k_ratio

        # For pre-computed features, we're working directly in feature space
        # so we need a simpler model architecture
        self.num_task_vectors = len(task_vectors)

        # Initialize the meta network to predict coefficients
        if blockwise:
            # Simplified for pre-computed features - we estimate
            # the number of blocks based on a typical model
            self.num_blocks = 12  # Typical for ViT models
            self.meta_net = MetaNet(feature_dim, self.num_task_vectors * self.num_blocks)
        else:
            self.meta_net = MetaNet(feature_dim, self.num_task_vectors)

        # For feature-based transformation
        self.task_features = nn.ParameterList([
            nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.01, requires_grad=True)
            for _ in range(self.num_task_vectors)
        ])

        # Initialize a projection layer to transform task vectors
        self.projection = nn.Linear(feature_dim, feature_dim, bias=False)
        nn.init.eye_(self.projection.weight)  # Initialize as identity

        self.printed_selection_info = False

    def forward(self, features):
        """Forward pass using pre-computed features

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Pre-computed feature vectors

        Returns:
        ----------
        output: Tensor [batch_size, feature_dim]
            Transformed feature vectors
        """
        # Generate coefficients using meta network
        if self.blockwise:
            # Reshape to [batch_size, num_task_vectors, num_blocks]
            batch_coefficients = self.meta_net(features).reshape(
                -1, self.num_task_vectors, self.num_blocks
            )
            # Average over blocks for simplicity in the pre-computed case
            coefficients = batch_coefficients.mean(dim=2)
        else:
            coefficients = self.meta_net(features)

        # Apply task vectors directly in feature space
        batch_size = features.size(0)
        outputs = []

        for i in range(batch_size):
            # Get coefficients for this sample
            sample_coeffs = coefficients[i]  # [num_task_vectors]

            # Apply task vectors as feature transformations
            transformed = features[i].unsqueeze(0)  # [1, feature_dim]

            for j, task_matrix in enumerate(self.task_features):
                # Apply task vector with its coefficient
                coeff = sample_coeffs[j]
                task_effect = coeff * torch.matmul(transformed, task_matrix)
                transformed = transformed + task_effect

            # Project back to original feature space
            transformed = self.projection(transformed)
            outputs.append(transformed)

        return torch.cat(outputs, dim=0)

    def compute_intervention_loss(self, features):
        """Compute causal intervention loss for pre-computed features

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Pre-computed feature vectors

        Returns:
        ----------
        loss: Tensor
            Intervention loss
        """
        if not self.enable_causal or not self.blockwise:
            return torch.tensor(0.0, device=features.device)

        # Generate coefficients
        batch_coefficients = self.meta_net(features).reshape(
            -1, self.num_task_vectors, self.num_blocks
        )

        # Select top-k blocks based on coefficient magnitude
        avg_coef_magnitude = batch_coefficients.abs().mean(dim=(0, 1))
        k = max(1, int(self.num_blocks * self.top_k_ratio))
        _, top_k_indices = torch.topk(avg_coef_magnitude, k)

        # Only print selection info once
        if not self.printed_selection_info and is_main_process():
            print(f"Selected {k} out of {self.num_blocks} blocks for intervention")
            self.printed_selection_info = True

        # Compute regular outputs
        regular_outputs = self.forward(features)

        # Compute intervention effects
        total_variance = 0.0

        # Use tqdm only on main process
        if is_main_process():
            progress_iter = tqdm(top_k_indices, desc="Computing intervention effects", leave=False)
        else:
            progress_iter = top_k_indices

        for j in progress_iter:
            intervention_diffs = []

            for i in range(features.size(0)):
                # Create a modified coefficient tensor with zeroed block j
                modified_coeffs = batch_coefficients[i].clone()
                modified_coeffs[:, j] = 0.0

                # Compute output with this intervention
                modified_output = self._forward_with_coeffs(features[i:i + 1], modified_coeffs)

                # Compute squared difference
                diff = torch.sum((regular_outputs[i] - modified_output[0]) ** 2)
                intervention_diffs.append(diff)

            if intervention_diffs:
                total_variance += torch.stack(intervention_diffs).mean()

        return total_variance

    def _forward_with_coeffs(self, features, coefficients):
        """Forward pass with specific coefficients

        Parameters:
        ----------
        features: Tensor [batch_size, feature_dim]
            Pre-computed feature vectors
        coefficients: Tensor [num_task_vectors, num_blocks]
            Pre-defined coefficients

        Returns:
        ----------
        output: Tensor [batch_size, feature_dim]
            Transformed feature vectors
        """
        # Average over blocks
        avg_coeffs = coefficients.mean(dim=1)

        # Apply task vectors
        transformed = features.clone()

        for j, task_matrix in enumerate(self.task_features):
            coeff = avg_coeffs[j]
            task_effect = coeff * torch.matmul(transformed, task_matrix)
            transformed = transformed + task_effect

        # Project back
        return self.projection(transformed)