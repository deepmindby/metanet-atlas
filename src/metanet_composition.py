"""Dynamic Task Vector Composition Based on Meta-Net

This module implements the MetaNet-aTLAS algorithm, which dynamically generates
task vector composition coefficients for each sample through a Meta-Net network,
enabling sample-level knowledge composition.

Unlike the original aTLAS, MetaNet-aTLAS dynamically adjusts the combination weights
of different task vectors according to the characteristics of the input sample,
improving the model's adaptability and generalization ability.
"""

import torch
import torch.nn as nn
from functorch import jvp, make_functional_with_buffers


class MetaNet(nn.Module):
    """Meta-Net network for generating task vector composition coefficients from sample features

    The network structure is a two-layer bottleneck architecture (Linear-ReLU-Linear),
    with the sample features as input and the task vector composition coefficients
    for that sample as output.
    """

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

        # Initialize to zero so that the combination coefficients are close to zero in the initial state
        nn.init.zeros_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

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


class MetaNetImageEncoder(nn.Module):
    """Meta-Net based image encoder for dynamic composition of task vectors in non-linear models

    Takes samples as input, first obtains features through a base encoder, then generates
    task vector composition coefficients through Meta-Net, and finally dynamically combines
    task vectors using these coefficients.
    """

    def __init__(self, model, task_vectors, blockwise=False) -> None:
        """Initialize Meta-Net image encoder

        Parameters:
        ----------
        model: nn.Module
            Base image encoder model
        task_vectors: List[NonLinearTaskVector]
            List of task vectors
        blockwise: bool, default: False
            Whether to use different coefficients for each parameter block
        """
        super().__init__()

        # Extract the functions and parameters of the base model
        func, params, self.buffer = make_functional_with_buffers(model)
        # Avoid meta tensor no data error
        self.func = lambda p, b, x: func(p, b, x)
        self.params = torch.nn.ParameterList(params)
        for p in self.params:
            p.requires_grad = False

        # Copy attributes from the image encoder
        self.train_preprocess = model.train_preprocess
        self.val_preprocess = model.val_preprocess
        self.cache_dir = model.cache_dir

        # Store task vectors
        self.dparams = [[tv.vector[k] for k in tv.vector] for tv in task_vectors]
        self.blockwise = blockwise

        # Get the feature dimension of the model, assuming it's the last parameter object
        # Note: Different models may need different ways to get the feature dimension
        if hasattr(model, 'model') and hasattr(model.model, 'ln_final'):
            feat_dim = model.model.ln_final.bias.numel()
        else:
            # If can't determine, use a default value or infer from model structure
            # Here we assume using ViT's output dimension, typically 768 (ViT-B) or 1024 (ViT-L)
            feat_dim = 768

            # Initialize Meta-Net
        if blockwise:
            # If using blockwise, each parameter block has its own coefficients
            self.meta_net = MetaNet(feat_dim, len(task_vectors) * len(self.params))
        else:
            # Otherwise each task vector has a global coefficient
            self.meta_net = MetaNet(feat_dim, len(task_vectors))

    def _apply(self, fn):
        """Override _apply method to relocate buffer list

        Note: This function signature is for PyTorch 1.13.1.
        Newer versions have added another optional parameter `recurse=True`.
        """
        new_self = super()._apply(fn=fn)
        new_self.buffer = (fn(x) for x in new_self.buffer)
        new_self.dparams = [[fn(x) for x in tv] for tv in new_self.dparams]
        return new_self

    def train(self, mode=True):
        """Set training mode"""
        super().train(mode)

    def forward(self, x) -> torch.Tensor:
        """Forward propagation

        Parameters:
        ----------
        x: Tensor [batch_size, 3, H, W]
            Batch of input images

        Returns:
        ----------
        features: Tensor [batch_size, feature_dim]
            Encoded features
        """
        # 1. First get base features
        with torch.no_grad():
            base_features = self.func(self.params, self.buffer, x)

        # 2. Generate combination coefficients using Meta-Net
        if self.blockwise:
            # If using blockwise, reshape coefficients to [batch_size, n_task_vectors, n_params]
            batch_coefficients = self.meta_net(base_features).reshape(-1, len(self.dparams), len(self.params))
        else:
            # Otherwise coefficients shape is [batch_size, n_task_vectors]
            batch_coefficients = self.meta_net(base_features)

        # 3. Apply specific combination coefficients for each sample in the batch
        batch_size = x.size(0)
        all_outputs = []

        for i in range(batch_size):
            if self.blockwise:
                # Use different coefficients for each parameter block
                coefs = batch_coefficients[i]  # [n_task_vectors, n_params]
                dparams = [sum([p * c[i] for p, c in zip(dp, coefs)]) for i, dp in enumerate(zip(*self.dparams))]
            else:
                # Use global coefficients for each task vector
                coefs = batch_coefficients[i]  # [n_task_vectors]
                dparams = [sum([p * c for p, c in zip(dp, coefs)]) for dp in zip(*self.dparams)]

            # Apply combined parameter differences
            new_params = [dp + p for dp, p in zip(dparams, self.params)]

            # Calculate output for current sample
            output = self.func(new_params, self.buffer, x[i:i + 1])
            all_outputs.append(output)

        # 4. Combine outputs from all samples
        return torch.cat(all_outputs, dim=0)


class MetaNetLinearizedModel(nn.Module):
    """Meta-Net based linearized model for dynamic composition of task vectors in linearized models

    Similar to MetaNetImageEncoder, but designed specifically for linearized models.
    """

    def __init__(self, model, task_vectors, blockwise=False) -> None:
        """Initialize Meta-Net linearized model

        Parameters:
        ----------
        model: LinearizedModel
            Linearized base model
        task_vectors: List[LinearizedTaskVector]
            List of linearized task vectors
        blockwise: bool, default: False
            Whether to use different coefficients for each parameter block
        """
        super().__init__()

        self.params0 = model.params0
        self.func0 = model.func0
        self.buffers0 = model.buffers0
        self._model_name = model._model_name

        # Store task vectors
        self.dparams = [[tv.vector[k] for k in tv.vector if k.startswith('model.params.')] for tv in task_vectors]
        self.blockwise = blockwise

        # Get the feature dimension of the model
        # Note: This is a simplified implementation, in practice it may need to be adjusted according to the specific model
        feat_dim = 768  # Assume using ViT-B's feature dimension

        # Initialize Meta-Net
        if blockwise:
            self.meta_net = MetaNet(feat_dim, len(task_vectors) * len(self.params0))
        else:
            self.meta_net = MetaNet(feat_dim, len(task_vectors))

    def _apply(self, fn):
        """Override _apply method to relocate buffer list"""
        new_self = super()._apply(fn=fn)
        new_self.buffers0 = (fn(x) for x in new_self.buffers0)
        new_self.dparams = [[fn(x) for x in tv] for tv in new_self.dparams]
        return new_self

    def forward(self, x) -> torch.Tensor:
        """Forward propagation

        Parameters:
        ----------
        x: Tensor [batch_size, 3, H, W]
            Batch of input images

        Returns:
        ----------
        outputs: Tensor [batch_size, feature_dim]
            Model outputs
        """
        # 1. First get base features for generating coefficients
        with torch.no_grad():
            base_features = self.func0(self.params0, self.buffers0, x)

        # 2. Generate combination coefficients using Meta-Net
        if self.blockwise:
            batch_coefficients = self.meta_net(base_features).reshape(-1, len(self.dparams), len(self.params0))
        else:
            batch_coefficients = self.meta_net(base_features)

        # 3. Apply specific combination coefficients for each sample in the batch
        batch_size = x.size(0)
        all_outputs = []

        for i in range(batch_size):
            if self.blockwise:
                coefs = batch_coefficients[i]
                dparams = [sum([p * c[i] for p, c in zip(dp, coefs)]) for i, dp in enumerate(zip(*self.dparams))]
            else:
                coefs = batch_coefficients[i]
                dparams = [sum([p * c for p, c in zip(dp, coefs)]) for dp in zip(*self.dparams)]

            # Apply first-order Taylor expansion to linearized model
            out, dp = jvp(
                lambda param: self.func0(param, self.buffers0, x[i:i + 1]),
                (tuple(self.params0),),
                (tuple(dparams),),
            )
            output = out + dp
            all_outputs.append(output)

        # 4. Combine outputs from all samples
        return torch.cat(all_outputs, dim=0)