"""Dynamic Task Vector Composition Based on Meta-Net with Causal Intervention

This module implements the MetaNet-aTLAS algorithm with causal intervention,
which dynamically generates task vector composition coefficients for each sample
through a Meta-Net network, and enhances robustness through causal intervention.

The causal intervention helps the model to understand which parameter blocks are
most important for prediction, and reduces dependency on any single block.
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


class MetaNetImageEncoder(nn.Module):
    """Meta-Net based image encoder for dynamic composition of task vectors in non-linear models

    Takes samples as input, first obtains features through a base encoder, then generates
    task vector composition coefficients through Meta-Net, and finally dynamically combines
    task vectors using these coefficients.
    """

    def __init__(self, model, task_vectors, blockwise=False, enable_causal=False) -> None:
        """Initialize Meta-Net image encoder

        Parameters:
        ----------
        model: nn.Module
            Base image encoder model
        task_vectors: List[NonLinearTaskVector]
            List of task vectors
        blockwise: bool, default: False
            Whether to use different coefficients for each parameter block
        enable_causal: bool, default: False
            Whether to enable causal intervention during training
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

        # Causal intervention settings
        self.enable_causal = enable_causal

        # Get the actual feature dimension of the model using a test forward pass
        with torch.no_grad():
            # Create a test input tensor with matching dtype
            param = next(iter(params))
            dummy_input = torch.zeros(1, 3, 224, 224, dtype=param.dtype, device=param.device)
            # Get features from the base model
            features = self.func(params, self.buffer, dummy_input)
            feat_dim = features.shape[1]  # Get the actual feature dimension

        # Initialize Meta-Net with the correct dimensions
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

    def forward(self, x, intervention_block=None, intervention_mode="zero") -> torch.Tensor:
        """Forward propagation with optional causal intervention

        Parameters:
        ----------
        x: Tensor [batch_size, 3, H, W]
            Batch of input images
        intervention_block: int, optional
            Block index to intervene on, or None for no intervention
        intervention_mode: str, default="zero"
            Mode for intervention: "zero" to zero out coefficients, "perturb" to add noise

        Returns:
        ----------
        features: Tensor [batch_size, feature_dim]
            Encoded features
        """
        # Ensure input has the same dtype as model parameters
        param_dtype = next(iter(self.params)).dtype
        x = x.to(param_dtype)

        # 1. First get base features
        with torch.no_grad():
            base_features = self.func(self.params, self.buffer, x)

        # 2. Generate combination coefficients using Meta-Net
        if self.blockwise:
            # If using blockwise, reshape coefficients to [batch_size, n_task_vectors, n_params]
            batch_coefficients = self.meta_net(base_features).reshape(-1, len(self.dparams), len(self.params))

            # Apply intervention if specified (only in training mode with causal enabled)
            if intervention_block is not None and self.training and self.enable_causal:
                # Create a copy of coefficients to modify
                intervened_coeffs = batch_coefficients.clone()

                # Apply intervention based on the specified mode
                if intervention_mode == "zero":
                    # Zero out coefficients for the specified block
                    intervened_coeffs[:, :, intervention_block] = 0.0
                elif intervention_mode == "perturb":
                    # Add noise to coefficients for the specified block
                    noise = torch.randn_like(intervened_coeffs[:, :, intervention_block]) * 0.1
                    intervened_coeffs[:, :, intervention_block] += noise

                # Use intervened coefficients
                batch_coefficients = intervened_coeffs

        else:
            # Otherwise coefficients shape is [batch_size, n_task_vectors]
            batch_coefficients = self.meta_net(base_features)

            # For non-blockwise, intervention doesn't make sense
            # as we can't target specific parameter blocks

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

            # Apply combined parameter differences with proper dtype
            new_params = [(dp + p).to(p.dtype) for dp, p in zip(dparams, self.params)]

            # Calculate output for current sample
            output = self.func(new_params, self.buffer, x[i:i + 1])
            all_outputs.append(output)

        # 4. Combine outputs from all samples
        return torch.cat(all_outputs, dim=0)

    def compute_intervention_loss(self, x):
        """Compute variance penalty loss through causal intervention

        For each block, this method:
        1. Performs a forward pass with the block's coefficients set to zero
        2. Measures the difference between normal output and intervened output
        3. Returns the sum of squared differences as a penalty term

        Parameters:
        ----------
        x: Tensor [batch_size, 3, H, W]
            Batch of input images

        Returns:
        ----------
        loss: Tensor
            Variance penalty loss
        """
        if not self.blockwise or not self.enable_causal:
            # If not using blockwise coefficients or causal intervention not enabled,
            # return zero loss
            return torch.tensor(0.0, device=x.device)

        # Get regular outputs first
        regular_outputs = self.forward(x)

        # For each parameter block, compute intervention loss
        intervention_diffs = []
        for j in range(len(self.params)):
            # Get outputs with intervention on block j
            intervened_outputs = self.forward(x, intervention_block=j, intervention_mode="zero")

            # Compute squared difference
            diff = torch.pow(regular_outputs - intervened_outputs, 2).mean()
            intervention_diffs.append(diff)

        # Sum up differences across all blocks
        total_variance = sum(intervention_diffs)

        return total_variance


class MetaNetLinearizedModel(nn.Module):
    """Meta-Net based linearized model for dynamic composition of task vectors in linearized models

    Similar to MetaNetImageEncoder, but designed specifically for linearized models.
    """

    def __init__(self, model, task_vectors, blockwise=False, enable_causal=False) -> None:
        """Initialize Meta-Net linearized model

        Parameters:
        ----------
        model: LinearizedModel
            Linearized base model
        task_vectors: List[LinearizedTaskVector]
            List of linearized task vectors
        blockwise: bool, default: False
            Whether to use different coefficients for each parameter block
        enable_causal: bool, default: False
            Whether to enable causal intervention during training
        """
        super().__init__()

        self.params0 = model.params0
        self.func0 = model.func0
        self.buffers0 = model.buffers0
        self._model_name = model._model_name

        # Store task vectors
        self.dparams = [[tv.vector[k] for k in tv.vector if k.startswith('model.params.')] for tv in task_vectors]
        self.blockwise = blockwise

        # Causal intervention settings
        self.enable_causal = enable_causal

        # Get the actual feature dimension of the model using a test forward pass
        with torch.no_grad():
            # Create a test input tensor with matching dtype
            param = next(iter(model.params0))
            dummy_input = torch.zeros(1, 3, 224, 224, dtype=param.dtype, device=param.device)
            # Get features from the base model
            features = self.func0(self.params0, self.buffers0, dummy_input)
            feat_dim = features.shape[1]  # Get the actual feature dimension

        print(f"Detected feature dimension: {feat_dim}")

        # Initialize Meta-Net with the correct dimensions
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

    def forward(self, x, intervention_block=None, intervention_mode="zero") -> torch.Tensor:
        """Forward propagation with optional causal intervention

        Parameters:
        ----------
        x: Tensor [batch_size, 3, H, W]
            Batch of input images
        intervention_block: int, optional
            Block index to intervene on, or None for no intervention
        intervention_mode: str, default="zero"
            Mode for intervention: "zero" to zero out coefficients, "perturb" to add noise

        Returns:
        ----------
        outputs: Tensor [batch_size, feature_dim]
            Model outputs
        """
        # Ensure input has the same dtype as model parameters
        param_dtype = next(iter(self.params0)).dtype
        x = x.to(param_dtype)

        # 1. First get base features for generating coefficients
        with torch.no_grad():
            base_features = self.func0(self.params0, self.buffers0, x)

        # 2. Generate combination coefficients using Meta-Net
        if self.blockwise:
            batch_coefficients = self.meta_net(base_features).reshape(-1, len(self.dparams), len(self.params0))

            # Apply intervention if specified (only in training mode with causal enabled)
            if intervention_block is not None and self.training and self.enable_causal:
                # Create a copy of coefficients to modify
                intervened_coeffs = batch_coefficients.clone()

                # Apply intervention based on the specified mode
                if intervention_mode == "zero":
                    # Zero out coefficients for the specified block
                    intervened_coeffs[:, :, intervention_block] = 0.0
                elif intervention_mode == "perturb":
                    # Add noise to coefficients for the specified block
                    noise = torch.randn_like(intervened_coeffs[:, :, intervention_block]) * 0.1
                    intervened_coeffs[:, :, intervention_block] += noise

                # Use intervened coefficients
                batch_coefficients = intervened_coeffs
        else:
            batch_coefficients = self.meta_net(base_features)

            # For non-blockwise, intervention doesn't make sense
            # as we can't target specific parameter blocks

        # 3. Apply specific combination coefficients for each sample in the batch
        batch_size = x.size(0)
        all_outputs = []

        for i in range(batch_size):
            if self.blockwise:
                coefs = batch_coefficients[i]
                dparams = [sum([p * c[i] for p, c in zip(dp, coefs)]).to(param_dtype) for i, dp in enumerate(zip(*self.dparams))]
            else:
                coefs = batch_coefficients[i]
                dparams = [sum([p * c for p, c in zip(dp, coefs)]).to(param_dtype) for dp in zip(*self.dparams)]

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

    def compute_intervention_loss(self, x):
        """Compute variance penalty loss through causal intervention

        For each block, this method:
        1. Performs a forward pass with the block's coefficients set to zero
        2. Measures the difference between normal output and intervened output
        3. Returns the sum of squared differences as a penalty term

        Parameters:
        ----------
        x: Tensor [batch_size, 3, H, W]
            Batch of input images

        Returns:
        ----------
        loss: Tensor
            Variance penalty loss
        """
        if not self.blockwise or not self.enable_causal:
            # If not using blockwise coefficients or causal intervention not enabled,
            # return zero loss
            return torch.tensor(0.0, device=x.device)

        # Get regular outputs first
        regular_outputs = self.forward(x)

        # For each parameter block, compute intervention loss
        intervention_diffs = []
        for j in range(len(self.params0)):
            # Get outputs with intervention on block j
            intervened_outputs = self.forward(x, intervention_block=j, intervention_mode="zero")

            # Compute squared difference
            diff = torch.pow(regular_outputs - intervened_outputs, 2).mean()
            intervention_diffs.append(diff)

        # Sum up differences across all blocks
        total_variance = sum(intervention_diffs)

        return total_variance