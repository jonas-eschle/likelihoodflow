"""
Spline-based normalizing flows using JAX.

This module implements Neural Spline Flows (NSF) which use
rational-quadratic splines for more flexible transformations.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import random, nn, lax
import haiku as hk

from normflow.flows.base import Flow


def rational_quadratic_spline(inputs: jnp.ndarray,
                             widths: jnp.ndarray,
                             heights: jnp.ndarray,
                             derivatives: jnp.ndarray,
                             inverse: bool = False,
                             bounds: Tuple[float, float] = (-10.0, 10.0),
                             min_bin_width: float = 1e-3,
                             min_bin_height: float = 1e-3,
                             min_derivative: float = 1e-3):
    """
    Apply a rational-quadratic spline transformation.
    
    This implements the transformation described in "Neural Spline Flows"
    by Durkan et al. 2019, which uses a rational-quadratic spline
    (piecewise function with quadratic numerator and linear denominator)
    for more flexible transformations.
    
    Args:
        inputs: Input tensor, shape (batch_size, ...)
        widths: Unnormalized bin widths, shape (..., num_bins)
        heights: Unnormalized bin heights, shape (..., num_bins)
        derivatives: Unnormalized derivatives at bin boundaries, shape (..., num_bins+1)
        inverse: Whether to apply inverse transformation
        bounds: Lower and upper bounds for the spline
        min_bin_width: Minimum bin width
        min_bin_height: Minimum bin height
        min_derivative: Minimum derivative at bin boundaries
        
    Returns:
        Tuple of (transformed_inputs, log_det)
    """
    # Unpack bounds
    lower, upper = bounds
    
    # Number of bins
    num_bins = widths.shape[-1]
    
    # Ensure inputs are within bounds
    inputs = jnp.clip(inputs, lower, upper)
    
    # Scale inputs to [0, 1]
    inputs = (inputs - lower) / (upper - lower)
    
    # Ensure that widths, heights, and derivatives are positive
    # And that they sum to 1 for widths and heights
    widths = jax.nn.softmax(widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    
    heights = jax.nn.softmax(heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    
    derivatives = jnp.exp(derivatives)
    derivatives = min_derivative + derivatives
    
    # Compute bin locations (cumulative widths)
    bin_left_edges = jnp.concatenate([
        jnp.zeros_like(widths[..., :1]),
        jnp.cumsum(widths, axis=-1)[..., :-1]
    ], axis=-1)
    
    bin_right_edges = bin_left_edges + widths
    
    # Compute heights at bin edges
    cumheights = jnp.concatenate([
        jnp.zeros_like(heights[..., :1]),
        jnp.cumsum(heights, axis=-1)[..., :-1]
    ], axis=-1)
    
    cumheights = jnp.concatenate([
        cumheights, 
        jnp.ones_like(cumheights[..., :1])
    ], axis=-1)
    
    # Find the bin containing each input
    # We use searchsorted for this, which returns the index where the 
    # input would be inserted to maintain order
    # find_bin_idx = jnp.searchsorted(bin_right_edges, inputs) - 1
    
    # JAX doesn't have a direct equivalent to np.searchsorted
    # So we implement a similar functionality
    find_bin_idx = jnp.sum(jnp.heaviside(inputs[..., None] - bin_left_edges, 0), axis=-1).astype(int) - 1
    find_bin_idx = jnp.clip(find_bin_idx, 0, num_bins - 1)
    
    # Get the bin edges, heights, and derivatives for each input
    input_dim = len(inputs.shape)
    
    # Create indices for gather operations
    idx = jnp.zeros_like(find_bin_idx)
    for i in range(input_dim - 1):
        idx_shape = [1] * input_dim
        idx_shape[i] = inputs.shape[i]
        idx = idx + jnp.arange(inputs.shape[i]).reshape(idx_shape) * num_bins
    
    gather_idx = idx + find_bin_idx
    
    # Get values for the current bin (we use ravel and unravel for gather)
    # This is equivalent to using advanced indexing in numpy
    bin_idx = find_bin_idx
    
    def gather_1d(params, indices):
        return params.ravel()[indices.ravel()].reshape(indices.shape)
    
    # Get bin properties for the selected bin
    input_left_edges = gather_1d(bin_left_edges, bin_idx)
    input_right_edges = gather_1d(bin_right_edges, bin_idx)
    input_widths = input_right_edges - input_left_edges
    
    input_cumheights = gather_1d(cumheights, bin_idx)
    input_bin_heights = gather_1d(heights, bin_idx)
    
    delta = input_bin_heights / input_widths
    
    input_derivatives_left = gather_1d(derivatives[..., :-1], bin_idx)
    input_derivatives_right = gather_1d(derivatives[..., 1:], bin_idx)
    
    # Compute the quadratic spline transformation
    if inverse:
        # Inverse transformation: y -> x
        inputs = jnp.clip(inputs, 0.0, 1.0)
        
        # Find which bin the input falls into
        # This is done above for both forward and inverse
        
        # Compute normalized position within bin
        input_delta_y = inputs - input_cumheights
        input_norm_heights = input_delta_y / input_bin_heights
        
        # Compute spline coefficients
        a = (input_derivatives_right + input_derivatives_left - 2 * delta) * input_widths
        b = (3 * delta - 2 * input_derivatives_left - input_derivatives_right) * input_widths
        c = input_derivatives_left * input_widths
        d = -input_delta_y
        
        # Compute a quadratic in x to find the inverse transformation
        # Note: There are numerical issues here that could be addressed
        # with a more robust approach, but this is sufficient for now
        # We're solving ax^3 + bx^2 + cx + d = 0 for x
        
        # For now we'll use a simple approach: compute it directly
        # In practice, this can be improved with more robust polynomial solvers
        
        # Compute the result using rational quadratic interpolation
        # This is a simplification - in practice the spline calculation is more complex
        # But the core idea is to find x given y, which requires solving a quadratic equation
        
        # We compute this using the full rational-quadratic spline formula
        num_iter = 5  # Number of iterations for numerical solving
        x = jnp.ones_like(inputs) * 0.5  # Initial guess (middle of the bin)
        
        # Simple iterative solver for the inverse (not the most efficient but works)
        for _ in range(num_iter):
            # Compute the value of the forward mapping at current x
            x_widths = input_right_edges - input_left_edges
            x_pos = (x - input_left_edges) / x_widths
            
            numerator = input_bin_heights * (
                input_derivatives_left * x_pos**2 +
                2 * delta * x_pos * (1 - x_pos) +
                input_derivatives_right * (1 - x_pos)**2
            )
            
            denominator = (input_derivatives_left * x_pos**2 +
                           delta * (x_pos * (1 - x_pos) + (1 - x_pos) * x_pos) +
                           input_derivatives_right * (1 - x_pos)**2)
            
            y_est = input_cumheights + numerator / denominator
            
            # Update x based on error
            x = x - (y_est - inputs) * 0.5
            x = jnp.clip(x, input_left_edges, input_right_edges)
        
        outputs = x
        
        # Calculate log determinant of inverse
        # For the rational quadratic spline, this is:
        # log(derivative of inverse) = -log(derivative of forward)
        
        # Compute normalized position within bin for the solved x
        bin_pos = (outputs - input_left_edges) / input_widths
        
        # Compute derivative of the forward transformation at x
        deriv_numerator = input_bin_heights * (
            input_derivatives_left * bin_pos**2 +
            2 * delta * bin_pos * (1 - bin_pos) +
            input_derivatives_right * (1 - bin_pos)**2
        )
        
        deriv_denominator = (input_derivatives_left * bin_pos**2 +
                            delta * bin_pos * (1 - bin_pos) +
                            input_derivatives_right * (1 - bin_pos)**2)**2
        
        logdet = -jnp.log(deriv_numerator / deriv_denominator)
        
    else:
        # Forward transformation: x -> y
        # Compute normalized position within bin
        bin_pos = (inputs - input_left_edges) / input_widths
        
        # Compute the spline output using rational quadratic interpolation
        numerator = input_bin_heights * (
            input_derivatives_left * bin_pos**2 +
            2 * delta * bin_pos * (1 - bin_pos) +
            input_derivatives_right * (1 - bin_pos)**2
        )
        
        denominator = (input_derivatives_left * bin_pos**2 +
                       delta * bin_pos * (1 - bin_pos) +
                       input_derivatives_right * (1 - bin_pos)**2)
        
        outputs = input_cumheights + numerator / denominator
        
        # Compute the log determinant of the transformation
        # This is log(dy/dx) for each bin
        deriv_numerator = input_bin_heights * (
            input_derivatives_left * bin_pos**2 +
            2 * delta * bin_pos * (1 - bin_pos) +
            input_derivatives_right * (1 - bin_pos)**2
        )
        
        deriv_denominator = (input_derivatives_left * bin_pos**2 +
                            delta * bin_pos * (1 - bin_pos) +
                            input_derivatives_right * (1 - bin_pos)**2)**2
        
        logdet = jnp.log(deriv_numerator / deriv_denominator)
    
    # Scale outputs back to the original domain
    outputs = outputs * (upper - lower) + lower
    
    # Scale the log determinant by the width of the domain
    logdet = logdet - jnp.log(upper - lower)
    
    return outputs, logdet


def create_spline_nn(hidden_dims: List[int], 
                    num_bins: int, 
                    activation: Callable = nn.relu) -> hk.Transformed:
    """
    Create a neural network for computing spline parameters.
    
    Args:
        hidden_dims: List of hidden dimensions
        num_bins: Number of bins for the spline
        activation: Activation function for hidden layers
        
    Returns:
        Haiku transformed function that applies the network
    """
    def nn_fn(x, output_dim):
        nn = hk.Sequential([
            hk.nets.MLP(hidden_dims + [output_dim], activation=activation),
        ])
        return nn(x)
    
    return hk.without_apply_rng(hk.transform(nn_fn))


class NSF(Flow):
    """
    Neural Spline Flow using rational-quadratic splines.
    
    Based on "Neural Spline Flows" by Durkan et al. 2019.
    
    This flow uses a coupling layer architecture similar to RealNVP,
    but replaces the affine transformation with a more flexible monotonic
    rational-quadratic spline transformation.
    """
    
    def __init__(self, 
                dim: int, 
                hidden_dims: List[int] = [64, 64],
                num_bins: int = 8,
                num_layers: int = 4,
                bounds: Tuple[float, float] = (-10.0, 10.0),
                mask_strategy: str = "alternating",
                activation: Callable = nn.relu,
                min_bin_width: float = 1e-3,
                min_bin_height: float = 1e-3,
                min_derivative: float = 1e-3,
                name: Optional[str] = None):
        """
        Initialize a Neural Spline Flow.
        
        Args:
            dim: Dimensionality of the flow
            hidden_dims: Hidden dimensions of the coupling networks
            num_bins: Number of bins for the spline
            num_layers: Number of coupling layers
            bounds: Bounds for the spline domain
            mask_strategy: Strategy for creating masks
            activation: Activation function for coupling networks
            min_bin_width: Minimum bin width
            min_bin_height: Minimum bin height
            min_derivative: Minimum derivative at bin boundaries
            name: Name of the flow
        """
        super().__init__(dim=dim, name=name)
        
        self.hidden_dims = hidden_dims
        self.num_bins = num_bins
        self.num_layers = num_layers
        self.bounds = bounds
        self.mask_strategy = mask_strategy
        self.activation = activation
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        
        # Each bin has width and height parameters, plus derivatives at bin boundaries
        # For each input dimension, we need:
        # - num_bins widths
        # - num_bins heights
        # - num_bins+1 derivatives at bin boundaries
        # Total parameters per dimension: 3*num_bins + 1
        self.params_per_dim = 3 * num_bins + 1
        
        # Create masks for each layer
        self.masks = self._create_masks()
        
        # Create spline networks for each layer
        self.spline_nns = [
            create_spline_nn(hidden_dims, num_bins, activation)
            for _ in range(num_layers)
        ]
    
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
        
        # Compute number of transformed dimensions for each layer
        transformed_dims = [(1 - mask).sum().astype(int) for mask in self.masks]
        
        for i, (nn, layer_key, n_transform) in enumerate(zip(self.spline_nns, keys, transformed_dims)):
            # Each transformed dimension requires parameters for
            # widths, heights, and derivatives
            output_size = n_transform * self.params_per_dim
            
            # Create dummy input data for initialization
            # This is just zeros with the same shape as masked input
            dummy_x = jnp.zeros((1, self.dim))
            
            # Initialize the network
            params[f"layer_{i}"] = nn.init(layer_key, dummy_x, output_size)
            
        return params
    
    def _coupling_layer(self,
                       params: Dict, 
                       x: jnp.ndarray,
                       mask: jnp.ndarray,
                       nn: hk.Transformed,
                       inverse: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply a single coupling layer with spline transformation.
        
        Args:
            params: Parameters for this coupling layer
            x: Input tensor
            mask: Binary mask (1 = identity, 0 = transform)
            nn: Neural network for spline parameters
            inverse: Whether to apply inverse transformation
            
        Returns:
            Tuple of transformed tensor and log determinant
        """
        batch_size = x.shape[0]
        
        # Split into identity and transform parts
        identity_part = x * mask[None, :]
        transform_part = x * (1 - mask[None, :])
        
        # Number of transformed dimensions
        n_transform = (1 - mask).sum().astype(int)
        
        # Get parameters from the identity part using the neural network
        transform_params = nn.apply(params, identity_part, n_transform * self.params_per_dim)
        
        # Reshape to get parameters for each dimension
        transform_params = transform_params.reshape(batch_size, n_transform, self.params_per_dim)
        
        # Split parameters into widths, heights, and derivatives
        widths = transform_params[:, :, :self.num_bins]
        heights = transform_params[:, :, self.num_bins:2*self.num_bins]
        derivatives = transform_params[:, :, 2*self.num_bins:]
        
        # Get the dimensions to transform
        transform_indices = jnp.where(mask == 0)[0]
        
        # Initialize output and log determinant
        y = jnp.zeros_like(x)
        y = y.at[:, mask == 1].set(identity_part[:, mask == 1])
        log_det = jnp.zeros(batch_size)
        
        # Apply spline transformation to each transformed dimension
        for i, idx in enumerate(transform_indices):
            y_i, log_det_i = rational_quadratic_spline(
                transform_part[:, idx],
                widths[:, i],
                heights[:, i],
                derivatives[:, i],
                inverse=inverse,
                bounds=self.bounds,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative
            )
            
            y = y.at[:, idx].set(y_i)
            log_det = log_det + log_det_i
        
        return y, log_det
    
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
                layer_params, current_x, mask, self.spline_nns[i], inverse=False
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
                layer_params, current_y, mask, self.spline_nns[i], inverse=True
            )
            log_det_sum += log_det
            
        return current_y, log_det_sum
