"""Dynamic Task Vector Composition Based on Meta-Net with Selective Causal Intervention

This module implements the MetaNet-aTLAS algorithm with selective causal intervention,
which dynamically generates task vector composition coefficients for each sample
through a Meta-Net network, and enhances robustness through selective causal intervention.

The selective causal intervention only performs intervention on the most important parameter
blocks (selected based on gradient magnitude), reducing memory consumption significantly.
"""

import torch
import torch.nn as nn
from functorch import jvp, make_functional_with_buffers
from tqdm import tqdm
from src.distributed import is_main_process


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
    with selective causal intervention.
    """

    def __init__(self, model, task_vectors, blockwise=False, enable_causal=False, top_k_ratio=0.1) -> None:
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
        top_k_ratio: float, default: 0.1
            Ratio of parameter blocks to select for intervention (default: 10%)
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
        self.top_k_ratio = top_k_ratio  # Store the ratio of blocks to use for intervention
        self.printed_selection_info = False  # Track if selection info has been printed

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

    def forward(self, x) -> torch.Tensor:
        """Forward propagation with optional causal intervention

        Parameters:
        ----------
        x: Tensor [batch_size, 3, H, W]
            Batch of input images

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

            # Apply combined parameter differences with proper dtype
            new_params = [(dp + p).to(p.dtype) for dp, p in zip(dparams, self.params)]

            # Calculate output for current sample
            output = self.func(new_params, self.buffer, x[i:i + 1])
            all_outputs.append(output)

        # 4. Combine outputs from all samples
        return torch.cat(all_outputs, dim=0)

    def compute_intervention_loss(self, x):
        """Compute variance penalty loss through selective causal intervention

        Uses linearized approximation and only intervenes on the most important parameter
        blocks based on coefficient magnitudes, saving memory significantly.

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

        # Get base features for coefficient generation
        with torch.no_grad():
            base_features = self.func(self.params, self.buffer, x)

        # Generate coefficients with Meta-Net
        batch_coefficients = self.meta_net(base_features).reshape(-1, len(self.dparams), len(self.params))
        batch_size = x.size(0)

        # Select top-k blocks based on coefficient magnitude (average across batch and task vectors)
        avg_coef_magnitude = batch_coefficients.abs().mean(dim=(0, 1))  # Average over batch and task vectors
        num_blocks = len(self.params)
        k = max(1, int(num_blocks * self.top_k_ratio))  # At least 1 block

        # Get indices of top-k blocks with highest coefficient magnitudes
        _, top_k_indices = torch.topk(avg_coef_magnitude, k)

        # Only print selection info once
        if not self.printed_selection_info and is_main_process():
            print(f"Selected {k} out of {num_blocks} blocks for intervention based on coefficient magnitude")
            self.printed_selection_info = True

        # Compute regular outputs with original coefficients (need gradients)
        all_regular_outputs = []

        # Keep parameters for gradient calculation
        param_refs = []

        for i in range(batch_size):
            # Prepare parameters with gradients for this sample
            coefs = batch_coefficients[i]
            dparams = [sum([p * c[i] for p, c in zip(dp, coefs)]) for i, dp in enumerate(zip(*self.dparams))]
            new_params = [(dp + p).to(p.dtype).requires_grad_(True) for dp, p in zip(dparams, self.params)]
            param_refs.append(new_params)

            # Calculate output for current sample
            output = self.func(new_params, self.buffer, x[i:i+1])
            all_regular_outputs.append(output)

        regular_outputs = torch.cat(all_regular_outputs, dim=0)

        # Now use linear approximation for intervention effects, but only for top-k blocks
        total_variance = 0.0

        # Use tqdm for progress tracking, but only on main process
        if is_main_process():
            progress_iter = tqdm(top_k_indices, desc="Computing intervention effects", leave=False)
        else:
            progress_iter = top_k_indices

        for j in progress_iter:  # Only loop through top-k blocks
            # For each selected parameter block, simulate zeroing out the coefficients
            intervention_diffs = []

            for i in range(batch_size):
                # Get original parameters for this sample
                params = param_refs[i]

                # Compute delta_theta for intervention (what would happen if block j's coefficient became zero)
                coefs = batch_coefficients[i]
                # The change would be equivalent to removing the effect of task vectors for block j
                delta_coef = -coefs[:, j]  # Negate coefficients for block j

                # Calculate parameter delta that would occur if coefficient was zeroed
                delta_param = sum([dp[j] * dc for dp, dc in zip(self.dparams, delta_coef)])

                # Compute the effect using a vector-Jacobian product (linearized approximation)
                with torch.enable_grad():
                    try:
                        # Use allow_unused=True to handle unused parameters
                        grads = torch.autograd.grad(
                            regular_outputs[i].sum(),
                            params[j],
                            create_graph=True,
                            retain_graph=True,
                            allow_unused=True
                        )

                        # Check if gradient is None (parameter not used in computation)
                        grad = grads[0] if grads[0] is not None else torch.zeros_like(params[j])

                        # Linear approximation: f(θ+Δθ) ≈ f(θ) + Δθᵀ∇f(θ)
                        linear_effect = (delta_param * grad).sum()
                        intervention_diffs.append(linear_effect ** 2)
                    except Exception as e:
                        # If gradient computation fails, use a zero effect
                        if is_main_process():
                            print(f"Warning: Gradient computation failed for parameter block {j}: {e}")
                        intervention_diffs.append(torch.tensor(0.0, device=x.device))

            if intervention_diffs:  # Only add to total if we have valid differences
                total_variance += torch.stack(intervention_diffs).mean()

        return total_variance


class MetaNetLinearizedModel(nn.Module):
    """Meta-Net based linearized model for dynamic composition of task vectors in linearized models
    with selective causal intervention.
    """

    def __init__(self, model, task_vectors, blockwise=False, enable_causal=False, top_k_ratio=0.1) -> None:
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
        top_k_ratio: float, default: 0.1
            Ratio of parameter blocks to select for intervention (default: 10%)
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
        self.top_k_ratio = top_k_ratio  # Store the ratio of blocks to use for intervention
        self.printed_selection_info = False  # Track if selection info has been printed

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

    def forward(self, x) -> torch.Tensor:
        """Forward propagation for linearized model

        Parameters:
        ----------
        x: Tensor [batch_size, 3, H, W]
            Batch of input images

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
        else:
            batch_coefficients = self.meta_net(base_features)

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
        """Compute variance penalty loss through selective causal intervention

        Uses linearized model's inherent first-order approximation and only intervenes
        on the most important parameter blocks based on coefficient magnitudes.

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

        # Get base features and generate coefficients
        with torch.no_grad():
            base_features = self.func0(self.params0, self.buffers0, x)

        batch_coefficients = self.meta_net(base_features).reshape(-1, len(self.dparams), len(self.params0))
        batch_size = x.size(0)

        # Select top-k blocks based on coefficient magnitude (average across batch and task vectors)
        avg_coef_magnitude = batch_coefficients.abs().mean(dim=(0, 1))  # Average over batch and task vectors
        num_blocks = len(self.params0)
        k = max(1, int(num_blocks * self.top_k_ratio))  # At least 1 block

        # Get indices of top-k blocks with highest coefficient magnitudes
        _, top_k_indices = torch.topk(avg_coef_magnitude, k)

        # Only print selection info once
        if not self.printed_selection_info and is_main_process():
            print(f"Selected {k} out of {num_blocks} blocks for intervention based on coefficient magnitude")
            self.printed_selection_info = True

        # First compute regular outputs with original coefficients
        all_regular_outputs = []
        all_regular_dparams = []

        for i in range(batch_size):
            coefs = batch_coefficients[i]
            dparams = [sum([p * c[i] for p, c in zip(dp, coefs)]).to(next(iter(self.params0)).dtype)
                      for i, dp in enumerate(zip(*self.dparams))]
            all_regular_dparams.append(dparams)

            # Compute regular output using JVP (already linearized)
            try:
                out, dp = jvp(
                    lambda param: self.func0(param, self.buffers0, x[i:i+1]),
                    (tuple(self.params0),),
                    (tuple(dparams),),
                )
                all_regular_outputs.append(out + dp)
            except Exception as e:
                if is_main_process():
                    print(f"Warning: Failed to compute regular output for sample {i}: {e}")
                # Create a placeholder output with same shape as model output
                with torch.no_grad():
                    placeholder = self.func0(self.params0, self.buffers0, x[i:i+1])
                all_regular_outputs.append(placeholder)

        regular_outputs = torch.cat(all_regular_outputs, dim=0)

        # Now compute intervention effects, but only for selected top-k blocks
        total_variance = 0.0

        # Use tqdm for progress tracking, but only on main process
        if is_main_process():
            progress_iter = tqdm(top_k_indices, desc="Computing intervention effects", leave=False)
        else:
            progress_iter = top_k_indices

        for j in progress_iter:  # Only loop through selected top-k blocks
            intervention_diffs = []

            for i in range(batch_size):
                # Create intervened dparams by zeroing out the effect for block j
                intervened_dparams = list(all_regular_dparams[i])  # Copy original dparams
                intervened_dparams[j] = torch.zeros_like(intervened_dparams[j])

                try:
                    # Compute output with intervention using JVP
                    out, dp = jvp(
                        lambda param: self.func0(param, self.buffers0, x[i:i+1]),
                        (tuple(self.params0),),
                        (tuple(intervened_dparams),),
                    )
                    intervened_output = out + dp

                    # Compute squared difference
                    diff = torch.sum((regular_outputs[i] - intervened_output) ** 2)
                    intervention_diffs.append(diff)
                except Exception as e:
                    if is_main_process():
                        print(f"Warning: Failed to compute intervention effect for block {j}, sample {i}: {e}")

            # Add variance from this block to total only if we have valid differences
            if intervention_diffs:
                total_variance += torch.stack(intervention_diffs).mean()

        return total_variance