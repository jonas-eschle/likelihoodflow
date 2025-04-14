"""
Coupling-based normalizing flows using JAX.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import random, nn
import haiku as hk

from normflow.flows.base import Flow


def create_mlp_fn(hidden_dims: List[int],
                  activation: Callable = nn.relu,
                  output_activation: Optional[Callable] = None) -> Callable:
    """
    Create a multi-layer perceptron function using Haiku.

    Args:
        hidden_dims: List of hidden dimensions
        activation: Activation function for hidden layers
        output_activation: Optional activation for output layer

    Returns:
        Haiku transformed function that applies the MLP
    """
    def mlp_fn(x):
        mlp = hk.Sequential([
            hk.nets.MLP(hidden_dims, activation=activation),
            hk.Linear(x.shape[-1] * 2,
                      w_init=hk.initializers.VarianceScaling(0.01),
                      b_init=jnp.zeros),
        ])
        outputs = mlp(x)

        # Split into scale and shift
        d = x.shape[-1]
        shift = outputs[..., :d]
        scale = outputs[..., d:]

        # Apply tanh to constrain scale and avoid numerical instability
        scale = jnp.tanh(scale)

        return shift, scale

    return hk.without_apply_rng(hk.transform(mlp_fn))


class RealNVP(Flow):
    """
    Real-valued Non-Volume Preserving (RealNVP) flow.

    Based on "Density Estimation using Real NVP" by Dinh et al.

    RealNVP uses a coupling layer architecture that splits the input dimensions
    into two parts. One part is passed unchanged, while the other undergoes an
    affine transformation parameterized by the first part.
    """

    def __init__(self,
                 dim: int,
                 hidden_dims: List[int] = [64, 64],
                 num_layers: int = 4,
                 mask_strategy: str = "alternating",
                 activation: Callable = nn.relu,
                 name: Optional[str] = None):
        """
        Initialize a RealNVP flow.

        Args:
            dim: Dimensionality of the flow
            hidden_dims: Hidden dimensions of the coupling networks
            num_layers: Number of coupling layers
            mask_strategy: Strategy for creating masks ("alternating" or "checkerboard")
            activation: Activation function for coupling networks
            name: Name of the flow
        """
        super().__init__(dim=dim, name=name)

        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.mask_strategy = mask_strategy
        self.activation = activation

        # Create masks for each layer
        self.masks = self._create_masks()

        # Create coupling networks for each layer
        self.coupling_fns = [
            create_mlp_fn(hidden_dims, activation=activation)
            for _ in range(num_layers)
        ]

    def _create_masks(self) -> List[jnp.ndarray]:
        """
        Create binary masks for coupling layers.

        Returns:
            List of binary masks for each layer
        """
        if self.mask_strategy == "alternating":
            # Alternate between masking first and second half of dimensions
            masks = []
            for i in range(self.num_layers):
                if i % 2 == 0:
                    # Mask first half (1 = keep, 0 = transform)
                    mask = jnp.concatenate([
                        jnp.ones(self.dim // 2),
                        jnp.zeros(self.dim - self.dim // 2)
                    ])
                else:
                    # Mask second half
                    mask = jnp.concatenate([
                        jnp.zeros(self.dim // 2),
                        jnp.ones(self.dim - self.dim // 2)
                    ])
                masks.append(mask)
            return masks

        elif self.mask_strategy == "checkerboard":
            # Alternate between even and odd indices
            masks = []
            for i in range(self.num_layers):
                if i % 2 == 0:
                    # Mask even indices
                    mask = jnp.zeros(self.dim)
                    mask = mask.at[::2].set(1)
                else:
                    # Mask odd indices
                    mask = jnp.zeros(self.dim)
                    mask = mask.at[1::2].set(1)
                masks.append(mask)
            return masks

        else:
            raise ValueError(f"Unknown mask strategy: {self.mask_strategy}")

    def init_params(self, key: jnp.ndarray) -> Dict:
        """
        Initialize parameters for all coupling layers.

        Args:
            key: JAX PRNG key

        Returns:
            Dictionary of parameters for each coupling layer
        """
        keys = random.split(key, self.num_layers)
        params = {}

        # Generate some dummy data for initialization
        dummy_x = jnp.zeros((1, self.dim))

        for i, (coupling_fn, layer_key) in enumerate(zip(self.coupling_fns, keys)):
            # Initialize the i-th coupling network
            params[f"layer_{i}"] = coupling_fn.init(layer_key, dummy_x)

        return params

    def _coupling_layer(self,
                        params: Dict,
                        x: jnp.ndarray,
                        mask: jnp.ndarray,
                        coupling_fn: hk.Transformed,
                        inverse: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply a single coupling layer transformation.

        Args:
            params: Parameters for this coupling layer
            x: Input tensor
            mask: Binary mask (1 = identity, 0 = transform)
            coupling_fn: The coupling network function
            inverse: Whether to apply inverse transformation

        Returns:
            Tuple of transformed tensor and log determinant
        """
        masked_x = x * mask[:, None]
        shift, scale = coupling_fn.apply(params, masked_x)

        # Scale factor from tanh is in [-1, 1], add 1 to ensure positive
        # and clamp to ensure numerical stability
        scale_factor = jnp.exp(scale * (1 - mask[:, None]))

        if inverse:
            transformed_x = (x - shift * (1 - mask[:, None])) / scale_factor
            log_det = -jnp.sum(scale * (1 - mask[:, None]), axis=-1)
        else:
            transformed_x = x * mask[:, None] + (x * scale_factor + shift) * (1 - mask[:, None])
            log_det = jnp.sum(scale * (1 - mask[:, None]), axis=-1)

        return transformed_x, log_det

    def forward(self,
                params: Dict,
                x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply forward transformation.

        Args:
            params: Flow parameters
            x: Batch of samples from base distribution

        Returns:
            Tuple of transformed samples and log determinant
        """
        log_det_sum = jnp.zeros(x.shape[0])
        current_x = x

        for i in range(self.num_layers):
            layer_params = params[f"layer_{i}"]
            mask = self.masks[i]

            current_x, log_det = self._coupling_layer(
                layer_params, current_x, mask, self.coupling_fns[i], inverse=False
            )
            log_det_sum += log_det

        return current_x, log_det_sum

    def inverse(self,
                params: Dict,
                y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply inverse transformation.

        Args:
            params: Flow parameters
            y: Batch of samples from target distribution

        Returns:
            Tuple of inverse transformed samples and log determinant
        """
        log_det_sum = jnp.zeros(y.shape[0])
        current_y = y

        for i in range(self.num_layers - 1, -1, -1):
            layer_params = params[f"layer_{i}"]
            mask = self.masks[i]

            current_y, log_det = self._coupling_layer(
                layer_params, current_y, mask, self.coupling_fns[i], inverse=True
            )
            log_det_sum += log_det

        return current_y, log_det_sum


class NeuralSplineFlow(Flow):
    """
    Neural Spline Flow using rational-quadratic splines.

    Based on "Neural Spline Flows" by Durkan et al.

    This flow uses a coupling layer architecture similar to RealNVP,
    but replaces the affine transformation with a more flexible monotonic
    rational-quadratic spline transformation.

    Note: This is a simplified implementation focusing on coupling layers.
    A full implementation would include autoregressive versions as well.
    """

    def __init__(self,
                 dim: int,
                 hidden_dims: List[int] = [64, 64],
                 num_layers: int = 4,
                 num_bins: int = 8,
                 bound: float = 5.0,
                 mask_strategy: str = "alternating",
                 activation: Callable = nn.relu,
                 name: Optional[str] = None):
        """
        Initialize a Neural Spline Flow.

        Args:
            dim: Dimensionality of the flow
            hidden_dims: Hidden dimensions of the coupling networks
            num_layers: Number of coupling layers
            num_bins: Number of bins for the spline
            bound: Bound for the spline domain
            mask_strategy: Strategy for creating masks
            activation: Activation function for coupling networks
            name: Name of the flow
        """
        super().__init__(dim=dim, name=name)

        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.num_bins = num_bins
        self.bound = bound
        self.mask_strategy = mask_strategy
        self.activation = activation

        # Each bin has width and height parameters, plus bin boundaries
        # Each output dimension requires 3 * num_bins - 1 parameters
        self.bins_params_per_dim = 3 * num_bins - 1

        # Create masks for each layer
        self.masks = self._create_masks()

        # Create spline networks for each layer
        self.spline_fns = self._create_spline_networks()

    def _create_masks(self) -> List[jnp.ndarray]:
        """
        Create binary masks for coupling layers.

        Returns:
            List of binary masks for each layer
        """
        if self.mask_strategy == "alternating":
            masks = []
            for i in range(self.num_layers):
                if i % 2 == 0:
                    mask = jnp.concatenate([
                        jnp.ones(self.dim // 2),
                        jnp.zeros(self.dim - self.dim // 2)
                    ])
                else:
                    mask = jnp.concatenate([
                        jnp.zeros(self.dim // 2),
                        jnp.ones(self.dim - self.dim // 2)
                    ])
                masks.append(mask)
            return masks

        elif self.mask_strategy == "checkerboard":
            masks = []
            for i in range(self.num_layers):
                if i % 2 == 0:
                    mask = jnp.zeros(self.dim)
                    mask = mask.at[::2].set(1)
                else:
                    mask = jnp.zeros(self.dim)
                    mask = mask.at[1::2].set(1)
                masks.append(mask)
            return masks

        else:
            raise ValueError(f"Unknown mask strategy: {self.mask_strategy}")

    def _create_spline_networks(self) -> List[hk.Transformed]:
        """
        Create neural networks for spline parameters.

        Returns:
            List of transformed Haiku functions
        """
        def create_spline_network_fn(hidden_dims, activation):
            def fn(x):
                # Number of transformed dimensions
                n_transform = jnp.sum(1 - self.masks[0]).astype(int)

                # Output size: for each transformed dimension, we need parameters
                # for the spline (3 * num_bins - 1 parameters per dimension)
                output_size = n_transform * self.bins_params_per_dim

                mlp = hk.Sequential([
                    hk.nets.MLP(hidden_dims, activation=activation),
                    hk.Linear(output_size)
                ])

                return mlp(x)

            return hk.without_apply_rng(hk.transform(fn))

        return [
            create_spline_network_fn(self.hidden_dims, self.activation)
            for _ in range(self.num_layers)
        ]

    def _rational_quadratic_spline(self,
                                   inputs: jnp.ndarray,
                                   spline_params: jnp.ndarray,
                                   inverse: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply rational-quadratic spline transformation.

        Args:
            inputs: Input tensor to transform
            spline_params: Parameters for the spline
            inverse: Whether to apply inverse transformation

        Returns:
            Tuple of transformed tensor and log determinant
        """
        # This is a simplified implementation of the rational-quadratic spline
        # In a full implementation, we would need to:
        # 1. Unnormalize the inputs from [-bound, bound] to [0, 1]
        # 2. Split spline_params into widths, heights, and derivatives
        # 3. Ensure constraints (all positive, sum to 1 for widths and heights)
        # 4. Find which bin each input falls into
        # 5. Apply the rational-quadratic transformation
        # 6. Compute the log determinant
        # 7. Normalize the outputs back to [-bound, bound]

        # For simplicity, we'll use a placeholder implementation that mimics
        # the behavior of the spline but with a simpler transformation

        # Placeholder: use a simpler transformation that mimics the spline
        # In a real implementation, this would be the full spline transformation

        # Extract scale and shift from parameters (simplified)
        n_transform = inputs.shape[-1]
        scale = nn.softplus(spline_params[..., :n_transform]) + 0.5
        shift = spline_params[..., n_transform:2*n_transform]

        if inverse:
            outputs = (inputs - shift) / scale
            log_det = -jnp.sum(jnp.log(scale), axis=-1)
        else:
            outputs = inputs * scale + shift
            log_det = jnp.sum(jnp.log(scale), axis=-1)

        return outputs, log_det

    def _coupling_layer(self,
                        params: Dict,
                        x: jnp.ndarray,
                        mask: jnp.ndarray,
                        spline_fn: hk.Transformed,
                        inverse: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply a single coupling layer with spline transformation.

        Args:
            params: Parameters for this coupling layer
            x: Input tensor
            mask: Binary mask (1 = identity, 0 = transform)
            spline_fn: Function for spline parameters
            inverse: Whether to apply inverse transformation

        Returns:
            Tuple of transformed tensor and log determinant
        """
        # Split into identity and transform parts
        identity_part = x * mask[None, :]
        transform_part = x * (1 - mask[None, :])

        # Get spline parameters from the identity part
        spline_params = spline_fn.apply(params, identity_part)

        # Apply spline transformation to the transform part
        transformed_part, log_det = self._rational_quadratic_spline(
            transform_part, spline_params, inverse=inverse
        )

        # Combine the identity and transformed parts
        outputs = identity_part + transformed_part

        return outputs, log_det

    def init_params(self, key: jnp.ndarray) -> Dict:
        """
        Initialize parameters for all spline coupling layers.

        Args:
            key: JAX PRNG key

        Returns:
            Dictionary of parameters for each coupling layer
        """
        keys = random.split(key, self.num_layers)
        params = {}

        # Generate some dummy data for initialization
        dummy_x = jnp.zeros((1, self.dim))

        for i, (spline_fn, layer_key) in enumerate(zip(self.spline_fns, keys)):
            # Initialize the i-th spline network
            params[f"layer_{i}"] = spline_fn.init(layer_key, dummy_x)

        return params

    def forward(self,
                params: Dict,
                x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply forward transformation.

        Args:
            params: Flow parameters
            x: Batch of samples from base distribution

        Returns:
            Tuple of transformed samples and log determinant
        """
        log_det_sum = jnp.zeros(x.shape[0])
        current_x = x

        for i in range(self.num_layers):
            layer_params = params[f"layer_{i}"]
            mask = self.masks[i]

            current_x, log_det = self._coupling_layer(
                layer_params, current_x, mask, self.spline_fns[i], inverse=False
            )
            log_det_sum += log_det

        return current_x, log_det_sum

    def inverse(self,
                params: Dict,
                y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply inverse transformation.

        Args:
            params: Flow parameters
            y: Batch of samples from target distribution

        Returns:
            Tuple of inverse transformed samples and log determinant
        """
        log_det_sum = jnp.zeros(y.shape[0])
        current_y = y

        for i in range(self.num_layers - 1, -1, -1):
            layer_params = params[f"layer_{i}"]
            mask = self.masks[i]

            current_y, log_det = self._coupling_layer(
                layer_params, current_y, mask, self.spline_fns[i], inverse=True
            )
            log_det_sum += log_det

        return current_y, log_det_sum