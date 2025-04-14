"""
Negative log-likelihood loss functions for normalizing flow training.
"""

from typing import Dict, List, Optional, Tuple, Union, Callable

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import zfit


class NLLLoss:
    """
    Negative log-likelihood loss for training normalizing flows.
    
    This class provides various negative log-likelihood loss functions
    for training normalizing flows, with options for different objectives
    such as maximum likelihood or KL divergence minimization.
    """
    
    @staticmethod
    def maximum_likelihood(params: Dict, 
                          flow_log_prob_fn: Callable, 
                          batch: jnp.ndarray) -> jnp.ndarray:
        """
        Maximum likelihood loss function.
        
        Args:
            params: Flow parameters
            flow_log_prob_fn: Function to compute log probability under the flow
            batch: Batch of data points
            
        Returns:
            Negative log-likelihood (scalar)
        """
        # Compute log probability of the batch under the flow
        log_probs = flow_log_prob_fn(params, batch)
        
        # Return negative mean log likelihood
        return -jnp.mean(log_probs)
    
    @staticmethod
    def kl_divergence(params: Dict,
                     flow_log_prob_fn: Callable,
                     target_log_prob_fn: Callable,
                     batch: jnp.ndarray) -> jnp.ndarray:
        """
        KL divergence loss function.
        
        Estimates KL(p||q) where p is the data distribution and q is the flow.
        
        Args:
            params: Flow parameters
            flow_log_prob_fn: Function to compute log probability under the flow
            target_log_prob_fn: Function to compute log probability under the target
            batch: Batch of data points
            
        Returns:
            KL divergence estimate (scalar)
        """
        # Compute log probabilities under the flow
        flow_log_probs = flow_log_prob_fn(params, batch)
        
        # Compute log probabilities under the target
        target_log_probs = target_log_prob_fn(batch)
        
        # KL divergence: E_p[log(p) - log(q)]
        # Since E_p[log(p)] is constant w.r.t flow parameters, we can drop it
        return -jnp.mean(flow_log_probs)
    
    @staticmethod
    def reverse_kl_divergence(params: Dict,
                             flow_log_prob_fn: Callable,
                             flow_sample_fn: Callable,
                             target_log_prob_fn: Callable,
                             num_samples: int,
                             rng_key: jnp.ndarray) -> jnp.ndarray:
        """
        Reverse KL divergence loss function.
        
        Estimates KL(q||p) where q is the flow and p is the target distribution.
        
        Args:
            params: Flow parameters
            flow_log_prob_fn: Function to compute log probability under the flow
            flow_sample_fn: Function to sample from the flow
            target_log_prob_fn: Function to compute log probability under the target
            num_samples: Number of samples to use for estimation
            rng_key: JAX PRNG key
            
        Returns:
            Reverse KL divergence estimate (scalar)
        """
        # Sample from the flow
        samples = flow_sample_fn(params, num_samples, rng_key)
        
        # Compute log probabilities under the flow
        flow_log_probs = flow_log_prob_fn(params, samples)
        
        # Compute log probabilities under the target
        target_log_probs = target_log_prob_fn(samples)
        
        # KL divergence: E_q[log(q) - log(p)]
        kl = flow_log_probs - target_log_probs
        
        return jnp.mean(kl)


class ZfitNLLLoss:
    """
    Negative log-likelihood loss using zfit models.
    
    This class adapts zfit models for use in normalizing flow training.
    """
    
    def __init__(self, zfit_model: zfit.pdf.BasePDF):
        """
        Initialize a ZfitNLLLoss.
        
        Args:
            zfit_model: Fitted zfit model
        """
        self.zfit_model = zfit_model
    
    def target_log_prob_fn(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log probability under the zfit model.
        
        Args:
            x: Batch of data points
            
        Returns:
            Log probabilities
        """
        # Convert to TensorFlow tensor
        x_tf = tf.convert_to_tensor(np.asarray(x), dtype=tf.float64)
        
        # Compute log probability
        log_prob = self.zfit_model.log_prob(x_tf)
        
        # Run the TensorFlow graph and convert to JAX array
        return jnp.asarray(zfit.run(log_prob))
    
    def create_loss_fn(self, 
                      flow_log_prob_fn: Callable,
                      loss_type: str = 'maximum_likelihood') -> Callable:
        """
        Create a loss function for training a normalizing flow.
        
        Args:
            flow_log_prob_fn: Function to compute log probability under the flow
            loss_type: Type of loss to use
                       ('maximum_likelihood', 'kl_divergence', 'reverse_kl')
            
        Returns:
            Loss function compatible with JAX training
        """
        if loss_type == 'maximum_likelihood':
            def loss_fn(params: Dict, batch: jnp.ndarray) -> jnp.ndarray:
                return NLLLoss.maximum_likelihood(params, flow_log_prob_fn, batch)
                
        elif loss_type == 'kl_divergence':
            def loss_fn(params: Dict, batch: jnp.ndarray) -> jnp.ndarray:
                return NLLLoss.kl_divergence(
                    params, flow_log_prob_fn, self.target_log_prob_fn, batch)
                
        elif loss_type == 'reverse_kl':
            def loss_fn(params: Dict, batch: jnp.ndarray, 
                       flow_sample_fn: Callable, num_samples: int, 
                       rng_key: jnp.ndarray) -> jnp.ndarray:
                return NLLLoss.reverse_kl_divergence(
                    params, flow_log_prob_fn, flow_sample_fn, 
                    self.target_log_prob_fn, num_samples, rng_key)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss_fn


class CustomNLLLoss:
    """
    Custom negative log-likelihood loss for normalizing flow training.
    
    This class allows the user to provide a custom target distribution
    or loss function for training normalizing flows.
    """
    
    def __init__(self, 
                target_log_prob_fn: Optional[Callable] = None,
                custom_loss_fn: Optional[Callable] = None):
        """
        Initialize a CustomNLLLoss.
        
        Args:
            target_log_prob_fn: Function to compute log probability under the target
            custom_loss_fn: Custom loss function (overrides other options if provided)
        """
        self.target_log_prob_fn = target_log_prob_fn
        self.custom_loss_fn = custom_loss_fn
    
    def create_loss_fn(self, 
                      flow_log_prob_fn: Callable,
                      loss_type: str = 'maximum_likelihood') -> Callable:
        """
        Create a loss function for training a normalizing flow.
        
        Args:
            flow_log_prob_fn: Function to compute log probability under the flow
            loss_type: Type of loss to use
                       ('maximum_likelihood', 'kl_divergence', 'reverse_kl', 'custom')
            
        Returns:
            Loss function compatible with JAX training
        """
        if self.custom_loss_fn is not None:
            return self.custom_loss_fn
            
        if loss_type == 'maximum_likelihood':
            def loss_fn(params: Dict, batch: jnp.ndarray) -> jnp.ndarray:
                return NLLLoss.maximum_likelihood(params, flow_log_prob_fn, batch)
                
        elif loss_type == 'kl_divergence':
            if self.target_log_prob_fn is None:
                raise ValueError("Target log probability function required for KL divergence")
                
            def loss_fn(params: Dict, batch: jnp.ndarray) -> jnp.ndarray:
                return NLLLoss.kl_divergence(
                    params, flow_log_prob_fn, self.target_log_prob_fn, batch)
                
        elif loss_type == 'reverse_kl':
            if self.target_log_prob_fn is None:
                raise ValueError("Target log probability function required for reverse KL divergence")
                
            def loss_fn(params: Dict, batch: jnp.ndarray, 
                       flow_sample_fn: Callable, num_samples: int, 
                       rng_key: jnp.ndarray) -> jnp.ndarray:
                return NLLLoss.reverse_kl_divergence(
                    params, flow_log_prob_fn, flow_sample_fn, 
                    self.target_log_prob_fn, num_samples, rng_key)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss_fn
