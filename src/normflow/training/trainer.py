"""
Training utilities for normalizing flows using JAX.
"""

import time
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad, random
import numpy as np
import optax

from normflow.flows.base import Flow, NormalizingFlow


class FlowTrainer:
    """
    Trainer for normalizing flows using JAX and optax.
    
    This class provides utilities for training normalizing flows using
    various optimization algorithms and loss functions.
    """
    
    def __init__(self, 
                flow: NormalizingFlow,
                optimizer: optax.GradientTransformation = None,
                loss_fn: Optional[Callable] = None,
                batch_size: int = 128,
                num_samples_per_step: int = 1000,
                rng_seed: int = 42):
        """
        Initialize a FlowTrainer.
        
        Args:
            flow: The normalizing flow to train
            optimizer: Optax optimizer (default: Adam with 1e-3 learning rate)
            loss_fn: Loss function (default: negative log-likelihood)
            batch_size: Batch size for training
            num_samples_per_step: Number of samples to generate per step for 
                                  loss functions that require samples
            rng_seed: Random seed
        """
        self.flow = flow
        self.optimizer = optimizer or optax.adam(1e-3)
        self.batch_size = batch_size
        self.num_samples_per_step = num_samples_per_step
        
        # Set up RNG key
        self.rng_key = random.PRNGKey(rng_seed)
        
        # Use negative log-likelihood as default loss function
        self.loss_fn = loss_fn or self._default_nll_loss
        
        # JIT-compile the loss and update functions for efficiency
        self._jitted_loss_fn = jit(self.loss_fn)
        self._jitted_update_step = jit(self._update_step)
        
        # Initialize state
        self.initialized = False
        self.opt_state = None
        self.params = None
        self.step_count = 0
        self.loss_history = []
    
    def _default_nll_loss(self, params: Dict, batch: jnp.ndarray) -> jnp.ndarray:
        """
        Default negative log-likelihood loss function.
        
        Args:
            params: Flow parameters
            batch: Batch of data points
            
        Returns:
            Negative log-likelihood (scalar)
        """
        # Compute log probabilities for each point in the batch
        log_probs = self.flow.log_prob(params, batch)
        
        # Return negative mean log-likelihood
        return -jnp.mean(log_probs)
    
    def _update_step(self, 
                    params: Dict, 
                    opt_state: optax.OptState, 
                    batch: jnp.ndarray) -> Tuple[Dict, optax.OptState, jnp.ndarray]:
        """
        Perform a single update step.
        
        Args:
            params: Flow parameters
            opt_state: Optimizer state
            batch: Batch of data points
            
        Returns:
            Tuple of (updated parameters, updated optimizer state, loss)
        """
        # Compute loss and gradients
        loss_val, grads = value_and_grad(self.loss_fn)(params, batch)
        
        # Get parameter updates from optimizer
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        
        # Apply updates to parameters
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss_val
    
    def initialize(self, data: Union[jnp.ndarray, np.ndarray], params: Optional[Dict] = None):
        """
        Initialize the training with data and parameters.
        
        Args:
            data: Training data, shape (n_samples, dim)
            params: Flow parameters (if None, initialize randomly)
        """
        # Convert data to JAX array if needed
        self.data = jnp.asarray(data) if not isinstance(data, jnp.ndarray) else data
        
        # Initialize parameters if not provided
        if params is None:
            self.rng_key, subkey = random.split(self.rng_key)
            self.params = self.flow.init_params(subkey)
        else:
            self.params = params
        
        # Initialize optimizer state
        self.opt_state = self.optimizer.init(self.params)
        
        # Reset counters
        self.step_count = 0
        self.loss_history = []
        self.initialized = True
    
    def train_step(self) -> float:
        """
        Perform a single training step.
        
        Returns:
            Loss value for this step
        """
        if not self.initialized:
            raise RuntimeError("Trainer not initialized. Call initialize() first.")
        
        # Sample a batch from the data
        self.rng_key, subkey = random.split(self.rng_key)
        indices = random.choice(
            subkey, jnp.arange(len(self.data)), shape=(self.batch_size,), replace=True
        )
        batch = self.data[indices]
        
        # Perform update step
        self.params, self.opt_state, loss = self._jitted_update_step(
            self.params, self.opt_state, batch
        )
        
        # Update counters
        self.step_count += 1
        self.loss_history.append(float(loss))
        
        return float(loss)
    
    def train(self, 
             num_steps: int, 
             verbose: bool = True, 
             callback: Optional[Callable] = None) -> Dict[str, List[float]]:
        """
        Train the flow for a specified number of steps.
        
        Args:
            num_steps: Number of training steps
            verbose: Whether to print progress
            callback: Optional callback function called after each step
                     with signature callback(training, step, loss)
            
        Returns:
            Dict with training statistics
        """
        if not self.initialized:
            raise RuntimeError("Trainer not initialized. Call initialize() first.")
        
        start_time = time.time()
        initial_step = self.step_count
        
        for step in range(num_steps):
            # Perform a training step
            loss = self.train_step()
            
            # Print progress if verbose
            if verbose and (step + 1) % max(1, num_steps // 10) == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / max(1e-6, elapsed)
                print(f"Step {initial_step + step + 1}/{initial_step + num_steps}, "
                     f"Loss: {loss:.6f}, "
                     f"Steps/sec: {steps_per_sec:.1f}")
            
            # Call the callback if provided
            if callback is not None:
                callback(self, initial_step + step, loss)
        
        total_time = time.time() - start_time
        
        # Final report
        if verbose:
            print(f"Training completed in {total_time:.2f} seconds.")
            print(f"Final loss: {self.loss_history[-1]:.6f}")
        
        return {
            "loss_history": self.loss_history[-num_steps:],
            "training_time": total_time,
            "steps_per_second": num_steps / total_time
        }


class ZfitLossTrainer(FlowTrainer):
    """
    Trainer that uses a loss function derived from a zfit fit.
    
    This class extends FlowTrainer to use loss functions based on
    the negative log-likelihood from a zfit fit.
    """
    
    def __init__(self, 
                flow: NormalizingFlow,
                zfit_loss_fn: Callable,
                optimizer: optax.GradientTransformation = None,
                batch_size: int = 128,
                num_samples_per_step: int = 1000,
                rng_seed: int = 42):
        """
        Initialize a ZfitLossTrainer.
        
        Args:
            flow: The normalizing flow to train
            zfit_loss_fn: Loss function created from a zfit fit
            optimizer: Optax optimizer (default: Adam with 1e-3 learning rate)
            batch_size: Batch size for training
            num_samples_per_step: Number of samples to generate per step
            rng_seed: Random seed
        """
        super().__init__(
            flow=flow,
            optimizer=optimizer,
            loss_fn=zfit_loss_fn,
            batch_size=batch_size,
            num_samples_per_step=num_samples_per_step,
            rng_seed=rng_seed
        )


class KLDivergenceTrainer(FlowTrainer):
    """
    Trainer for minimizing KL divergence to a target distribution.
    
    This training is useful when we have a target distribution (e.g., from zfit)
    and want to fit a normalizing flow to approximate it efficiently.
    """
    
    def __init__(self, 
                flow: NormalizingFlow,
                target_log_prob_fn: Callable,
                optimizer: optax.GradientTransformation = None,
                forward_kl: bool = False,
                batch_size: int = 128,
                num_samples_per_step: int = 1000,
                rng_seed: int = 42):
        """
        Initialize a KLDivergenceTrainer.
        
        Args:
            flow: The normalizing flow to train
            target_log_prob_fn: Function computing log probability of the target distribution
            optimizer: Optax optimizer (default: Adam with 1e-3 learning rate)
            forward_kl: Whether to use forward (True) or reverse (False) KL divergence
            batch_size: Batch size for training
            num_samples_per_step: Number of samples to generate per step for 
                                  forward KL calculation
            rng_seed: Random seed
        """
        self.target_log_prob_fn = target_log_prob_fn
        self.forward_kl = forward_kl
        
        # Choose the appropriate loss function
        if forward_kl:
            loss_fn = self._forward_kl_loss
        else:
            loss_fn = self._reverse_kl_loss
        
        super().__init__(
            flow=flow,
            optimizer=optimizer,
            loss_fn=loss_fn,
            batch_size=batch_size,
            num_samples_per_step=num_samples_per_step,
            rng_seed=rng_seed
        )
    
    def _forward_kl_loss(self, params: Dict, batch: jnp.ndarray) -> jnp.ndarray:
        """
        Forward KL divergence loss: KL(p||q)
        
        Here, p is the flow distribution and q is the target distribution.
        We use Monte Carlo estimation with samples from the flow.
        
        Args:
            params: Flow parameters
            batch: Ignored (we generate samples from the flow)
            
        Returns:
            Forward KL divergence
        """
        # Generate samples from the flow for Monte Carlo estimation
        self.rng_key, subkey = random.split(self.rng_key)
        samples = self.flow.sample(params, self.num_samples_per_step, subkey)
        
        # Compute log probabilities under both distributions
        flow_log_prob = self.flow.log_prob(params, samples)
        target_log_prob = self.target_log_prob_fn(samples)
        
        # Compute the KL divergence: E_p[log(p) - log(q)]
        kl_divs = flow_log_prob - target_log_prob
        
        return jnp.mean(kl_divs)
    
    def _reverse_kl_loss(self, params: Dict, batch: jnp.ndarray) -> jnp.ndarray:
        """
        Reverse KL divergence loss: KL(q||p)
        
        Here, q is the target distribution and p is the flow distribution.
        We use Monte Carlo estimation with samples from the target distribution.
        
        Args:
            params: Flow parameters
            batch: Batch of samples from the target distribution
            
        Returns:
            Reverse KL divergence
        """
        # Compute log probabilities under the flow
        flow_log_prob = self.flow.log_prob(params, batch)
        
        # Compute log probabilities under the target (can be pre-computed)
        target_log_prob = self.target_log_prob_fn(batch)
        
        # Compute the KL divergence: E_q[log(q) - log(p)]
        # We're minimizing, so we can drop the E_q[log(q)] term as it's constant
        # So our loss is just the negative expected log-likelihood: -E_q[log(p)]
        kl_divs = -flow_log_prob
        
        return jnp.mean(kl_divs)


class FlowTuner:
    """
    Tune the complexity of a normalizing flow model.
    
    This class provides methods for optimizing the architecture of a flow
    by pruning unnecessary components while maintaining performance.
    """
    
    def __init__(self, 
                flow: NormalizingFlow,
                params: Dict,
                evaluation_data: jnp.ndarray,
                target_complexity: Optional[float] = None,
                complexity_metric: Optional[Callable] = None):
        """
        Initialize a FlowTuner.
        
        Args:
            flow: The normalizing flow to tune
            params: Current flow parameters
            evaluation_data: Data to evaluate performance on
            target_complexity: Target complexity (e.g., parameter count, storage size)
            complexity_metric: Function to compute complexity of flow
        """
        self.flow = flow
        self.params = params
        self.evaluation_data = evaluation_data
        self.target_complexity = target_complexity
        self.complexity_metric = complexity_metric or self._default_complexity_metric
        
        # Compute initial metrics
        self.initial_likelihood = self._compute_likelihood(params)
        self.initial_complexity = self.complexity_metric(flow, params)
        
        print(f"Initial negative log-likelihood: {-self.initial_likelihood:.6f}")
        print(f"Initial complexity: {self.initial_complexity:.1f}")
    
    def _default_complexity_metric(self, flow: NormalizingFlow, params: Dict) -> float:
        """
        Default complexity metric: total number of parameters.
        
        Args:
            flow: The normalizing flow
            params: Flow parameters
            
        Returns:
            Complexity score
        """
        # Recursively count all parameters in the dictionary
        def count_params(p):
            if isinstance(p, (np.ndarray, jnp.ndarray)):
                return p.size
            elif isinstance(p, dict):
                return sum(count_params(v) for v in p.values())
            elif isinstance(p, (list, tuple)):
                return sum(count_params(v) for v in p)
            else:
                return 1
        
        return float(count_params(params))
    
    def _compute_likelihood(self, params: Dict) -> float:
        """
        Compute the mean log-likelihood on evaluation data.
        
        Args:
            params: Flow parameters
            
        Returns:
            Mean log-likelihood
        """
        log_probs = self.flow.log_prob(params, self.evaluation_data)
        return float(jnp.mean(log_probs))
    
    def prune_layers(self, min_layers: int = 1) -> Tuple[NormalizingFlow, Dict]:
        """
        Prune unnecessary layers from a sequential flow.
        
        Args:
            min_layers: Minimum number of layers to keep
            
        Returns:
            Tuple of (pruned flow, pruned parameters)
        """
        # This is a simple example for SequentialFlow
        # In a real implementation, we would need more sophisticated pruning
        from normflow.flows.base import SequentialFlow
        
        if not isinstance(self.flow.flow, SequentialFlow):
            raise ValueError("Flow must be a SequentialFlow to prune layers")
        
        sequential_flow = self.flow.flow
        original_flows = sequential_flow.flows
        original_params = self.params
        
        # We can only prune if we have more than min_layers
        if len(original_flows) <= min_layers:
            print(f"Cannot prune: flow has only {len(original_flows)} layers")
            return self.flow, self.params
        
        print(f"Original flow has {len(original_flows)} layers")
        
        # Try removing each layer and evaluate performance
        best_flow = None
        best_params = None
        best_likelihood = -float('inf')
        best_index = -1
        
        for i in range(len(original_flows)):
            # Create a new flow without the i-th layer
            new_flows = [flow for j, flow in enumerate(original_flows) if j != i]
            new_sequential_flow = SequentialFlow(flows=new_flows)
            
            # Create new parameters without the i-th layer
            new_params = {}
            for j in range(len(original_flows)):
                if j < i:
                    new_params[f"flow_{j}"] = original_params[f"flow_{j}"]
                elif j > i:
                    new_params[f"flow_{j-1}"] = original_params[f"flow_{j}"]
            
            # Evaluate the new flow
            new_flow = NormalizingFlow(
                flow=new_sequential_flow,
                base_log_prob_fn=self.flow.base_log_prob_fn,
                base_sample_fn=self.flow.base_sample_fn,
                name=self.flow.name
            )
            
            new_likelihood = self._compute_likelihood(new_params)
            print(f"Removing layer {i}: log-likelihood = {new_likelihood:.6f}")
            
            if new_likelihood > best_likelihood:
                best_likelihood = new_likelihood
                best_flow = new_flow
                best_params = new_params
                best_index = i
        
        # Compare with original performance
        likelihood_change = best_likelihood - self.initial_likelihood
        print(f"Best layer to remove: {best_index}")
        print(f"Log-likelihood change: {likelihood_change:.6f}")
        
        # Only return the pruned flow if the performance drop is acceptable
        # (e.g., less than 1% decrease in likelihood)
        if likelihood_change >= -0.01 * abs(self.initial_likelihood):
            print(f"Pruning layer {best_index} with acceptable performance change")
            return best_flow, best_params
        else:
            print("No acceptable pruning found")
            return self.flow, self.params
    
    def quantize_parameters(self, bits: int = 8) -> Dict:
        """
        Quantize parameters to reduce storage size.
        
        Args:
            bits: Number of bits to use for quantization
            
        Returns:
            Quantized parameters
        """
        # Simple implementation of parameter quantization
        def quantize(p):
            if isinstance(p, (np.ndarray, jnp.ndarray)):
                # Compute the range of values
                p_min = float(jnp.min(p))
                p_max = float(jnp.max(p))
                
                # Skip if the values are all the same
                if p_min == p_max:
                    return p
                
                # Compute the step size
                step = (p_max - p_min) / (2**bits - 1)
                
                # Quantize the values
                p_quantized = jnp.round((p - p_min) / step) * step + p_min
                
                return p_quantized
            elif isinstance(p, dict):
                return {k: quantize(v) for k, v in p.items()}
            elif isinstance(p, (list, tuple)):
                return type(p)(quantize(v) for v in p)
            else:
                return p
        
        # Quantize the parameters
        quantized_params = quantize(self.params)
        
        # Evaluate the quantized parameters
        quantized_likelihood = self._compute_likelihood(quantized_params)
        quantized_complexity = self.complexity_metric(self.flow, quantized_params)
        
        print(f"Quantized negative log-likelihood: {-quantized_likelihood:.6f}")
        print(f"Quantized complexity: {quantized_complexity:.1f}")
        print(f"Complexity reduction: {(self.initial_complexity - quantized_complexity) / self.initial_complexity * 100:.1f}%")
        
        # Return the quantized parameters
        return quantized_params
