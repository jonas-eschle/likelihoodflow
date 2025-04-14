"""
Interface between zfit and normalizing flows.
"""

from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import zfit
from zfit import z

from normflow.flows.base import NormalizingFlow


class ZfitLossWrapper:
    """
    Wrapper for using zfit loss functions to train normalizing flows.
    
    This class provides an interface between zfit loss functions (particularly
    the negative log-likelihood) and normalizing flow training.
    """
    
    def __init__(self, 
                zfit_loss: zfit.loss.BaseLoss,
                data: np.ndarray):
        """
        Initialize a ZfitLossWrapper.
        
        Args:
            zfit_loss: A zfit loss object, typically UnbinnedNLL
            data: Training data as a numpy array
        """
        self.zfit_loss = zfit_loss
        self.data = data
        
        # Extract model and parameters from the loss
        self.model = zfit_loss.model
        self.zfit_params = list(self.model[0].get_params())
        
        # Store initial values for parameters
        self.initial_param_values = {param.name: param.numpy() for param in self.zfit_params}
        
    def nll_fn(self, data: jnp.ndarray) -> float:
        """
        Compute the negative log-likelihood for the given data.
        
        Args:
            data: Data points to evaluate
            
        Returns:
            Negative log-likelihood
        """
        # Convert JAX array to TensorFlow tensor
        tf_data = tf.convert_to_tensor(np.asarray(data), dtype=tf.float64)
        
        # Create a zfit Data object
        zfit_data = zfit.Data.from_tensor(obs=self.model.space, tensor=tf_data)
        
        # Compute loss
        loss_value = self.zfit_loss.value()
        
        # Run the TensorFlow graph
        return zfit.run(loss_value)
    
    def create_flow_loss_fn(self, flow: NormalizingFlow) -> Callable:
        """
        Create a loss function for training a normalizing flow.
        
        This function uses the negative log-likelihood from zfit and
        adapts it to be used for training a normalizing flow with JAX.
        
        Args:
            flow: Normalizing flow to train
            
        Returns:
            Loss function compatible with JAX training
        """
        def loss_fn(params: Dict, batch: jnp.ndarray) -> jnp.ndarray:
            """
            Compute loss for a batch of data.
            
            Args:
                params: Flow parameters
                batch: Batch of data points
                
            Returns:
                Loss value (scalar)
            """
            # Compute log probability of the batch under the flow
            log_probs = flow.log_prob(params, batch)
            
            # Return negative mean log likelihood
            return -jnp.mean(log_probs)
        
        return loss_fn


class ZfitFlowInterface:
    """
    Interface between zfit models and normalizing flows.
    
    This class provides utilities for fitting zfit models and then
    training normalizing flows to approximate the resulting distributions.
    """
    
    def __init__(self, 
                model: zfit.pdf.BasePDF, 
                data: zfit.Data,
                minimizer: Optional["zfit.minimize.ZfitMinimizer"] = None,
                flow: Optional[NormalizingFlow] = None):
        """
        Initialize a ZfitFlowInterface.
        
        Args:
            model: zfit model to fit
            data: Data to fit the model to
            minimizer: zfit minimizer to use
            flow: Optional normalizing flow
        """
        self.model = model
        self.data = data
        self.minimizer = minimizer or zfit.minimize.Minuit()
        self.flow = flow
        
        self.fit_result = None
        self.loss = None
        self.loss_wrapper = None
        
    def fit_model(self) -> zfit.result.FitResult:
        """
        Fit the zfit model.
        
        Returns:
            Fit result
        """
        # Create loss
        self.loss = zfit.loss.UnbinnedNLL(model=self.model, data=self.data)
        
        # Perform fit
        self.fit_result = self.minimizer.minimize(self.loss)
        
        # Extract data as numpy array for flow training
        data_np = zfit.run(self.data.value())
        
        # Create loss wrapper
        self.loss_wrapper = ZfitLossWrapper(self.loss, data_np)
        
        return self.fit_result
    
    def create_flow_loss(self, flow: Optional[NormalizingFlow] = None) -> Callable:
        """
        Create a loss function for training a normalizing flow.
        
        Args:
            flow: Normalizing flow to train (if not provided at initialization)
            
        Returns:
            Loss function compatible with JAX training
        """
        if self.fit_result is None:
            raise RuntimeError("Must call fit_model() before creating flow loss")
        
        flow = flow or self.flow
        if flow is None:
            raise ValueError("No normalizing flow provided")
        
        return self.loss_wrapper.create_flow_loss_fn(flow)
    
    def get_data_numpy(self) -> np.ndarray:
        """
        Get the training data as a numpy array.
        
        Returns:
            Data array
        """
        return zfit.run(self.data.value())
    
    def evaluate_flow(self, 
                     flow: NormalizingFlow, 
                     flow_params: Dict,
                     n_samples: int = 10000) -> Dict[str, float]:
        """
        Evaluate a trained flow against the zfit model.
        
        Args:
            flow: Trained normalizing flow
            flow_params: Flow parameters
            n_samples: Number of samples to use for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.fit_result is None:
            raise RuntimeError("Must call fit_model() before evaluating flow")
        
        # Sample from the flow
        key = jax.random.PRNGKey(0)
        flow_samples = flow.sample(flow_params, n_samples, key)
        
        # Compute log probability of samples under the flow
        flow_log_prob = flow.log_prob(flow_params, flow_samples)
        
        # Compute log probability of samples under the zfit model
        model_log_prob = self.model.log_prob(
            tf.convert_to_tensor(np.asarray(flow_samples), dtype=tf.float64)
        )
        model_log_prob = zfit.run(model_log_prob)
        
        # Compute metrics
        mean_flow_log_prob = float(jnp.mean(flow_log_prob))
        mean_model_log_prob = float(np.mean(model_log_prob))
        
        # Estimate KL divergence: KL(flow || model) = E_flow[log(flow) - log(model)]
        kl_flow_model = float(jnp.mean(flow_log_prob - model_log_prob))
        
        return {
            'mean_flow_log_prob': mean_flow_log_prob,
            'mean_model_log_prob': mean_model_log_prob,
            'kl_flow_model': kl_flow_model
        }
