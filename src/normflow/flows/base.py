"""
Base classes for normalizing flows using JAX.
"""

import abc
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import random


class Flow(abc.ABC):
    """
    Base class for normalizing flows.
    
    A normalizing flow is a transformation from a simple base distribution
    (e.g., a standard normal) to a more complex target distribution through
    a series of invertible transformations.
    """
    
    def __init__(self, 
                 dim: int, 
                 name: Optional[str] = None, 
                 **kwargs):
        """
        Initialize a flow.
        
        Args:
            dim: Dimensionality of the flow (number of input/output dimensions)
            name: Name of the flow (optional)
        """
        self.dim = dim
        self.name = name if name else self.__class__.__name__
        
    @abc.abstractmethod
    def init_params(self, key: jnp.ndarray) -> Dict:
        """
        Initialize parameters for the flow.
        
        Args:
            key: JAX PRNG key for random initialization
            
        Returns:
            Dictionary of parameters
        """
        pass
    
    @abc.abstractmethod
    def forward(self, 
               params: Dict, 
               x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward transformation from base distribution to target distribution.
        
        Args:
            params: Flow parameters
            x: Batch of samples from base distribution, shape (batch_size, dim)
            
        Returns:
            Tuple containing:
              - Transformed samples (batch_size, dim)
              - Log determinant of Jacobian of transformation (batch_size,)
        """
        pass
    
    @abc.abstractmethod
    def inverse(self, 
               params: Dict, 
               y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Inverse transformation from target distribution to base distribution.
        
        Args:
            params: Flow parameters
            y: Batch of samples from target distribution, shape (batch_size, dim)
            
        Returns:
            Tuple containing:
              - Inverse transformed samples (batch_size, dim)
              - Log determinant of Jacobian of inverse transformation (batch_size,)
        """
        pass
    
    def log_prob(self, 
                params: Dict, 
                base_log_prob_fn: Callable, 
                x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log probability of samples under the flow.
        
        Args:
            params: Flow parameters
            base_log_prob_fn: Log probability function of base distribution
            x: Batch of samples, shape (batch_size, dim)
            
        Returns:
            Log probabilities of samples (batch_size,)
        """
        z, log_det = self.inverse(params, x)
        return base_log_prob_fn(z) + log_det
    
    def sample(self, 
              params: Dict, 
              sample_base_fn: Callable, 
              num_samples: int, 
              key: jnp.ndarray) -> jnp.ndarray:
        """
        Sample from the flow.
        
        Args:
            params: Flow parameters
            sample_base_fn: Sampling function for base distribution
            num_samples: Number of samples to generate
            key: JAX PRNG key
            
        Returns:
            Batch of samples from flow, shape (num_samples, dim)
        """
        base_samples = sample_base_fn(key, (num_samples, self.dim))
        samples, _ = self.forward(params, base_samples)
        return samples


class SequentialFlow(Flow):
    """
    A sequence of flows applied one after another.
    """
    
    def __init__(self, 
                flows: List[Flow], 
                name: Optional[str] = None,
                **kwargs):
        """
        Initialize a sequential flow.
        
        Args:
            flows: List of flow objects
            name: Name of the sequential flow (optional)
        """
        if not flows:
            raise ValueError("List of flows cannot be empty")
        
        # Verify that all flows have the same dimensionality
        dim = flows[0].dim
        for flow in flows:
            if flow.dim != dim:
                raise ValueError(f"All flows must have the same dimension. "
                               f"Expected {dim}, got {flow.dim} for {flow.name}")
        
        super().__init__(dim=dim, name=name, **kwargs)
        self.flows = flows
        
    def init_params(self, key: jnp.ndarray) -> Dict:
        """
        Initialize parameters for all flows in the sequence.
        
        Args:
            key: JAX PRNG key
            
        Returns:
            Dictionary of parameters for all flows
        """
        keys = random.split(key, len(self.flows))
        return {f"flow_{i}": flow.init_params(keys[i]) 
                for i, flow in enumerate(self.flows)}
    
    def forward(self, 
               params: Dict, 
               x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply forward transformation for all flows in sequence.
        
        Args:
            params: Flow parameters
            x: Batch of samples from base distribution
            
        Returns:
            Tuple of transformed samples and log determinant
        """
        log_det_sum = jnp.zeros(x.shape[0])
        current_x = x
        
        for i, flow in enumerate(self.flows):
            flow_params = params[f"flow_{i}"]
            current_x, log_det = flow.forward(flow_params, current_x)
            log_det_sum += log_det
            
        return current_x, log_det_sum
    
    def inverse(self, 
               params: Dict, 
               y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply inverse transformation for all flows in reverse sequence.
        
        Args:
            params: Flow parameters
            y: Batch of samples from target distribution
            
        Returns:
            Tuple of inverse transformed samples and log determinant
        """
        log_det_sum = jnp.zeros(y.shape[0])
        current_y = y
        
        for i in range(len(self.flows) - 1, -1, -1):
            flow = self.flows[i]
            flow_params = params[f"flow_{i}"]
            current_y, log_det = flow.inverse(flow_params, current_y)
            log_det_sum += log_det
            
        return current_y, log_det_sum


class NormalizingFlow:
    """
    A complete normalizing flow model with base distribution and transformation.
    """
    
    def __init__(self, 
                flow: Flow, 
                base_log_prob_fn: Callable,
                base_sample_fn: Callable,
                name: Optional[str] = None):
        """
        Initialize a normalizing flow model.
        
        Args:
            flow: Flow transformation
            base_log_prob_fn: Log probability function of base distribution
            base_sample_fn: Sampling function for base distribution
            name: Name of the model (optional)
        """
        self.flow = flow
        self.base_log_prob_fn = base_log_prob_fn
        self.base_sample_fn = base_sample_fn
        self.name = name if name else f"NormalizingFlow_{flow.name}"
        self.dim = flow.dim
        
    def init_params(self, key: jnp.ndarray) -> Dict:
        """
        Initialize flow parameters.
        
        Args:
            key: JAX PRNG key
            
        Returns:
            Flow parameters
        """
        return self.flow.init_params(key)
    
    def log_prob(self, params: Dict, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log probability of samples.
        
        Args:
            params: Flow parameters
            x: Batch of samples, shape (batch_size, dim)
            
        Returns:
            Log probabilities (batch_size,)
        """
        return self.flow.log_prob(params, self.base_log_prob_fn, x)
    
    def sample(self, 
              params: Dict, 
              num_samples: int, 
              key: jnp.ndarray) -> jnp.ndarray:
        """
        Generate samples from the flow.
        
        Args:
            params: Flow parameters
            num_samples: Number of samples to generate
            key: JAX PRNG key
            
        Returns:
            Batch of samples, shape (num_samples, dim)
        """
        return self.flow.sample(params, self.base_sample_fn, num_samples, key)
    
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
        return self.flow.forward(params, x)
    
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
        return self.flow.inverse(params, y)
    
    
# Common base distributions

def standard_normal_log_prob(x: jnp.ndarray) -> jnp.ndarray:
    """Standard normal log probability density function."""
    return -0.5 * jnp.sum(x**2, axis=-1) - 0.5 * x.shape[-1] * jnp.log(2 * jnp.pi)

def standard_normal_sample(key: jnp.ndarray, shape: Tuple[int, ...]) -> jnp.ndarray:
    """Sample from standard normal distribution."""
    return random.normal(key, shape)
