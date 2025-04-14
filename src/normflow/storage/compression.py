"""
Compression utilities for normalizing flows.

This module provides methods for compressing normalizing flows
to reduce storage size and inference time.
"""

import time
from typing import Dict, List, Optional, Tuple, Union, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from normflow.flows.base import Flow, NormalizingFlow


class FlowCompressor:
    """
    Compress normalizing flows for efficient storage and inference.
    
    This class provides methods for compressing flow parameters
    using techniques like quantization, pruning, and distillation.
    """
    
    def __init__(self, 
                flow: NormalizingFlow, 
                original_params: Dict,
                test_data: Optional[jnp.ndarray] = None):
        """
        Initialize a FlowCompressor.
        
        Args:
            flow: The normalizing flow to compress
            original_params: Original flow parameters
            test_data: Test data for evaluating compression (optional)
        """
        self.flow = flow
        self.original_params = original_params
        self.test_data = test_data
        
        # Compute original performance if test data is provided
        if test_data is not None:
            self.original_nll = -jnp.mean(flow.log_prob(original_params, test_data))
            self.original_size = self._get_params_size(original_params)
            print(f"Original NLL: {self.original_nll:.6f}")
            print(f"Original parameter size: {self.original_size / 1024:.2f} KB")
    
    def _get_params_size(self, params: Dict) -> int:
        """
        Compute the size of parameters in bytes.
        
        Args:
            params: Flow parameters
            
        Returns:
            Size in bytes
        """
        # Recursively compute the size of parameters
        def get_size(p):
            if isinstance(p, (np.ndarray, jnp.ndarray)):
                return p.nbytes
            elif isinstance(p, dict):
                return sum(get_size(v) for v in p.values())
            elif isinstance(p, (list, tuple)):
                return sum(get_size(v) for v in p)
            else:
                return 8  # Default size for scalar values
        
        return get_size(params)
    
    def _evaluate_compression(self, compressed_params: Dict) -> Dict:
        """
        Evaluate the performance of compressed parameters.
        
        Args:
            compressed_params: Compressed flow parameters
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.test_data is None:
            return {}
        
        # Compute performance metrics
        compressed_nll = -jnp.mean(self.flow.log_prob(compressed_params, self.test_data))
        compressed_size = self._get_params_size(compressed_params)
        
        metrics = {
            'original_nll': float(self.original_nll),
            'compressed_nll': float(compressed_nll),
            'nll_change': float(compressed_nll - self.original_nll),
            'nll_ratio': float(compressed_nll / self.original_nll),
            'original_size': self.original_size,
            'compressed_size': compressed_size,
            'size_reduction': self.original_size - compressed_size,
            'compression_ratio': float(self.original_size / compressed_size if compressed_size > 0 else float('inf'))
        }
        
        return metrics
    
    def quantize(self, 
                bits: int = 8, 
                method: str = 'uniform',
                per_tensor: bool = False) -> Tuple[Dict, Dict]:
        """
        Quantize flow parameters to reduce storage size.
        
        Args:
            bits: Number of bits for quantization
            method: Quantization method ('uniform', 'minmax', 'kmeans')
            per_tensor: Whether to use per-tensor quantization instead of per-parameter
            
        Returns:
            Tuple of (quantized parameters, evaluation metrics)
        """
        print(f"Quantizing parameters to {bits} bits using {method} method...")
        start_time = time.time()
        
        if method == 'uniform':
            quantized_params = self._quantize_uniform(self.original_params, bits, per_tensor)
        elif method == 'minmax':
            quantized_params = self._quantize_minmax(self.original_params, bits, per_tensor)
        elif method == 'kmeans':
            quantized_params = self._quantize_kmeans(self.original_params, bits, per_tensor)
        else:
            raise ValueError(f"Unknown quantization method: {method}")
        
        # Evaluate compression
        metrics = self._evaluate_compression(quantized_params)
        metrics['time'] = time.time() - start_time
        
        print(f"Quantization completed in {metrics['time']:.2f} seconds")
        print(f"Compressed size: {metrics['compressed_size'] / 1024:.2f} KB "
             f"({metrics['compression_ratio']:.2f}x compression)")
        
        if self.test_data is not None:
            print(f"NLL change: {metrics['nll_change']:.6f} "
                f"({metrics['nll_ratio']:.4f}x original)")
        
        return quantized_params, metrics
    
    def _quantize_uniform(self, 
                         params: Dict, 
                         bits: int,
                         per_tensor: bool) -> Dict:
        """
        Quantize parameters using uniform quantization.
        
        Args:
            params: Flow parameters
            bits: Number of bits for quantization
            per_tensor: Whether to use per-tensor quantization
            
        Returns:
            Quantized parameters
        """
        # Define a function to quantize a single tensor
        def quantize_tensor(p, bits):
            # Skip small tensors
            if isinstance(p, (np.ndarray, jnp.ndarray)) and p.size > 1:
                # Compute the range of values
                p_min = float(jnp.min(p))
                p_max = float(jnp.max(p))
                
                # Skip if all values are the same
                if p_min == p_max:
                    return p
                
                # Compute the step size
                step = (p_max - p_min) / (2**bits - 1)
                
                # Quantize the values
                p_quantized = jnp.round((p - p_min) / step) * step + p_min
                
                return p_quantized
            else:
                return p
        
        # Apply quantization to all parameters
        def apply_quantization(p, bits, per_tensor):
            if isinstance(p, dict):
                return {k: apply_quantization(v, bits, per_tensor) for k, v in p.items()}
            elif isinstance(p, (list, tuple)):
                return type(p)(apply_quantization(v, bits, per_tensor) for v in p)
            elif isinstance(p, (np.ndarray, jnp.ndarray)):
                if per_tensor:
                    # Quantize the entire tensor
                    return quantize_tensor(p, bits)
                else:
                    # Quantize each parameter separately
                    # This is a simplification - in practice, we might want to
                    # quantize per channel or per layer
                    if p.size == 1:
                        return p
                    
                    # For 1D tensors, just quantize the entire tensor
                    if len(p.shape) <= 1:
                        return quantize_tensor(p, bits)
                    
                    # For 2D+ tensors, quantize each output channel separately
                    # This is a common approach in neural network quantization
                    p_flat = p.reshape(-1)
                    return quantize_tensor(p_flat, bits).reshape(p.shape)
            else:
                return p
        
        return apply_quantization(params, bits, per_tensor)
    
    def _quantize_minmax(self, 
                        params: Dict, 
                        bits: int,
                        per_tensor: bool) -> Dict:
        """
        Quantize parameters using min-max quantization.
        
        This method is similar to uniform quantization but scales
        each tensor to the range [0, 1] before quantization.
        
        Args:
            params: Flow parameters
            bits: Number of bits for quantization
            per_tensor: Whether to use per-tensor quantization
            
        Returns:
            Quantized parameters
        """
        # Very similar to uniform quantization
        # The main difference is that we normalize to [0, 1]
        # before quantization and then scale back
        
        # Define a function to quantize a single tensor
        def quantize_tensor(p, bits):
            # Skip small tensors
            if isinstance(p, (np.ndarray, jnp.ndarray)) and p.size > 1:
                # Compute the range of values
                p_min = float(jnp.min(p))
                p_max = float(jnp.max(p))
                
                # Skip if all values are the same
                if p_min == p_max:
                    return p
                
                # Normalize to [0, 1]
                p_norm = (p - p_min) / (p_max - p_min)
                
                # Quantize the values
                levels = 2**bits - 1
                p_quantized = jnp.round(p_norm * levels) / levels
                
                # Scale back
                p_quantized = p_quantized * (p_max - p_min) + p_min
                
                return p_quantized
            else:
                return p
        
        # Apply quantization to all parameters
        def apply_quantization(p, bits, per_tensor):
            if isinstance(p, dict):
                return {k: apply_quantization(v, bits, per_tensor) for k, v in p.items()}
            elif isinstance(p, (list, tuple)):
                return type(p)(apply_quantization(v, bits, per_tensor) for v in p)
            elif isinstance(p, (np.ndarray, jnp.ndarray)):
                if per_tensor:
                    # Quantize the entire tensor
                    return quantize_tensor(p, bits)
                else:
                    # Similar to uniform quantization
                    if p.size == 1:
                        return p
                    
                    if len(p.shape) <= 1:
                        return quantize_tensor(p, bits)
                    
                    p_flat = p.reshape(-1)
                    return quantize_tensor(p_flat, bits).reshape(p.shape)
            else:
                return p
        
        return apply_quantization(params, bits, per_tensor)
    
    def _quantize_kmeans(self, 
                        params: Dict, 
                        bits: int,
                        per_tensor: bool) -> Dict:
        """
        Quantize parameters using k-means clustering.
        
        This method clusters parameter values and represents each value
        by its cluster centroid, allowing for more adaptive quantization.
        
        Args:
            params: Flow parameters
            bits: Number of bits for quantization
            per_tensor: Whether to use per-tensor quantization
            
        Returns:
            Quantized parameters
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("scikit-learn is required for k-means quantization. "
                            "Install it with 'pip install scikit-learn'.")
        
        # Number of centroids (cluster centers)
        n_clusters = min(2**bits, 256)  # Cap at 256 to avoid memory issues
        
        # Define a function to quantize a single tensor using k-means
        def quantize_tensor_kmeans(p, n_clusters):
            # Skip small tensors
            if isinstance(p, (np.ndarray, jnp.ndarray)) and p.size > n_clusters:
                # Convert to numpy if needed
                p_np = np.array(p)
                p_flat = p_np.reshape(-1, 1)  # KMeans expects 2D input
                
                # Skip if all values are the same
                if np.min(p_flat) == np.max(p_flat):
                    return jnp.array(p_np)
                
                # Apply k-means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(p_flat)
                
                # Replace each value with its cluster centroid
                p_quantized = kmeans.cluster_centers_[kmeans.labels_].reshape(p_np.shape)
                
                return jnp.array(p_quantized)
            else:
                return p
        
        # Apply quantization to all parameters
        def apply_quantization(p, n_clusters, per_tensor):
            if isinstance(p, dict):
                return {k: apply_quantization(v, n_clusters, per_tensor) for k, v in p.items()}
            elif isinstance(p, (list, tuple)):
                return type(p)(apply_quantization(v, n_clusters, per_tensor) for v in p)
            elif isinstance(p, (np.ndarray, jnp.ndarray)):
                if per_tensor:
                    # Quantize the entire tensor
                    return quantize_tensor_kmeans(p, n_clusters)
                else:
                    # Similar approach to the other methods
                    if p.size <= n_clusters:
                        return p
                    
                    if len(p.shape) <= 1:
                        return quantize_tensor_kmeans(p, n_clusters)
                    
                    # For 2D+ tensors, consider quantizing per-channel
                    # For simplicity, we'll just quantize the flattened tensor
                    p_flat = p.reshape(-1)
                    return quantize_tensor_kmeans(p_flat, n_clusters).reshape(p.shape)
            else:
                return p
        
        return apply_quantization(params, n_clusters, per_tensor)
    
    def prune(self, 
             sparsity: float = 0.9, 
             method: str = 'magnitude',
             fine_tune_steps: int = 0) -> Tuple[Dict, Dict]:
        """
        Prune flow parameters to reduce model size.
        
        Args:
            sparsity: Target sparsity (fraction of parameters to prune)
            method: Pruning method ('magnitude', 'l1_norm', 'random')
            fine_tune_steps: Number of fine-tuning steps after pruning
            
        Returns:
            Tuple of (pruned parameters, evaluation metrics)
        """
        print(f"Pruning parameters with {sparsity:.2f} sparsity using {method} method...")
        start_time = time.time()
        
        if method == 'magnitude':
            pruned_params = self._prune_magnitude(self.original_params, sparsity)
        elif method == 'l1_norm':
            pruned_params = self._prune_l1_norm(self.original_params, sparsity)
        elif method == 'random':
            pruned_params = self._prune_random(self.original_params, sparsity)
        else:
            raise ValueError(f"Unknown pruning method: {method}")
        
        # Fine-tune if requested
        if fine_tune_steps > 0 and self.test_data is not None:
            pruned_params = self._fine_tune(pruned_params, fine_tune_steps)
        
        # Evaluate compression
        metrics = self._evaluate_compression(pruned_params)
        metrics['time'] = time.time() - start_time
        
        print(f"Pruning completed in {metrics['time']:.2f} seconds")
        print(f"Compressed size: {metrics['compressed_size'] / 1024:.2f} KB "
             f"({metrics['compression_ratio']:.2f}x compression)")
        
        if self.test_data is not None:
            print(f"NLL change: {metrics['nll_change']:.6f} "
                f"({metrics['nll_ratio']:.4f}x original)")
        
        return pruned_params, metrics
    
    def _prune_magnitude(self, 
                        params: Dict, 
                        sparsity: float) -> Dict:
        """
        Prune parameters based on their magnitude.
        
        Args:
            params: Flow parameters
            sparsity: Target sparsity
            
        Returns:
            Pruned parameters
        """
        # Collect all weights into a flat array
        all_weights = []
        
        def collect_weights(p):
            if isinstance(p, (np.ndarray, jnp.ndarray)) and p.size > 1:
                all_weights.append(np.array(p).flatten())
            elif isinstance(p, dict):
                for v in p.values():
                    collect_weights(v)
            elif isinstance(p, (list, tuple)):
                for v in p:
                    collect_weights(v)
        
        collect_weights(params)
        
        # Flatten all weights
        all_weights_flat = np.concatenate(all_weights)
        
        # Compute threshold based on magnitude
        threshold = np.sort(np.abs(all_weights_flat))[int(sparsity * len(all_weights_flat))]
        
        # Apply pruning
        def apply_pruning(p, threshold):
            if isinstance(p, dict):
                return {k: apply_pruning(v, threshold) for k, v in p.items()}
            elif isinstance(p, (list, tuple)):
                return type(p)(apply_pruning(v, threshold) for v in p)
            elif isinstance(p, (np.ndarray, jnp.ndarray)) and p.size > 1:
                mask = jnp.abs(p) <= threshold
                p_pruned = jnp.where(mask, 0.0, p)
                return p_pruned
            else:
                return p
        
        return apply_pruning(params, threshold)
    
    def _prune_l1_norm(self, 
                      params: Dict, 
                      sparsity: float) -> Dict:
        """
        Prune parameters based on their L1 norm.
        
        This method computes the L1 norm of each parameter tensor
        and prunes the ones with the lowest norms.
        
        Args:
            params: Flow parameters
            sparsity: Target sparsity
            
        Returns:
            Pruned parameters
        """
        # Collect parameter tensors and their L1 norms
        param_norms = []
        
        def collect_norms(p, path=[]):
            if isinstance(p, (np.ndarray, jnp.ndarray)) and p.size > 1:
                param_norms.append((path, jnp.sum(jnp.abs(p))))
            elif isinstance(p, dict):
                for k, v in p.items():
                    collect_norms(v, path + [k])
            elif isinstance(p, (list, tuple)):
                for i, v in enumerate(p):
                    collect_norms(v, path + [i])
        
        collect_norms(params)
        
        # Sort parameters by L1 norm
        param_norms.sort(key=lambda x: float(x[1]))
        
        # Compute the number of parameters to prune
        n_prune = int(sparsity * len(param_norms))
        
        # Get paths of parameters to prune
        prune_paths = [path for path, _ in param_norms[:n_prune]]
        
        # Apply pruning
        def apply_pruning(p, path=[], prune_paths=prune_paths):
            if isinstance(p, dict):
                return {k: apply_pruning(v, path + [k], prune_paths) for k, v in p.items()}
            elif isinstance(p, (list, tuple)):
                return type(p)(apply_pruning(v, path + [i], prune_paths) for i, v in enumerate(p))
            elif isinstance(p, (np.ndarray, jnp.ndarray)) and p.size > 1:
                if path in prune_paths:
                    return jnp.zeros_like(p)
                else:
                    return p
            else:
                return p
        
        return apply_pruning(params)
    
    def _prune_random(self, 
                     params: Dict, 
                     sparsity: float) -> Dict:
        """
        Randomly prune parameters.
        
        Args:
            params: Flow parameters
            sparsity: Target sparsity
            
        Returns:
            Pruned parameters
        """
        # Set random seed for reproducibility
        np.random.seed(0)
        
        # Apply random pruning
        def apply_random_pruning(p, sparsity):
            if isinstance(p, dict):
                return {k: apply_random_pruning(v, sparsity) for k, v in p.items()}
            elif isinstance(p, (list, tuple)):
                return type(p)(apply_random_pruning(v, sparsity) for v in p)
            elif isinstance(p, (np.ndarray, jnp.ndarray)) and p.size > 1:
                # Create random mask
                mask = np.random.random(p.shape) > sparsity
                p_pruned = jnp.where(jnp.array(mask), p, 0.0)
                return p_pruned
            else:
                return p
        
        return apply_random_pruning(params, sparsity)
    
    def _fine_tune(self, 
                  pruned_params: Dict, 
                  steps: int) -> Dict:
        """
        Fine-tune pruned parameters to recover performance.
        
        Args:
            pruned_params: Pruned flow parameters
            steps: Number of fine-tuning steps
            
        Returns:
            Fine-tuned parameters
        """
        print(f"Fine-tuning pruned parameters for {steps} steps...")
        
        # Create a mask to identify pruned parameters
        def create_mask(original, pruned):
            if isinstance(original, dict):
                return {k: create_mask(original[k], pruned[k]) for k in original}
            elif isinstance(original, (list, tuple)):
                return type(original)(create_mask(original[i], pruned[i]) for i in range(len(original)))
            elif isinstance(original, (np.ndarray, jnp.ndarray)) and original.size > 1:
                return jnp.array(pruned != 0, dtype=jnp.float32)
            else:
                return 1.0
        
        mask = create_mask(self.original_params, pruned_params)
        
        # Define a function to apply a mask during optimization
        def apply_mask(params, mask):
            if isinstance(params, dict):
                return {k: apply_mask(params[k], mask[k]) for k in params}
            elif isinstance(params, (list, tuple)):
                return type(params)(apply_mask(params[i], mask[i]) for i in range(len(params)))
            elif isinstance(params, (np.ndarray, jnp.ndarray)) and params.size > 1:
                return params * mask
            else:
                return params
        
        # Create optimizer
        optimizer = optax.adam(learning_rate=1e-4)
        opt_state = optimizer.init(pruned_params)
        
        # Define loss function
        @jax.jit
        def loss_fn(params):
            log_probs = self.flow.log_prob(params, self.test_data)
            return -jnp.mean(log_probs)
        
        # Define update step
        @jax.jit
        def update_step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            # Apply mask to keep pruned parameters at zero
            new_params = apply_mask(new_params, mask)
            return new_params, new_opt_state, loss
        
        # Fine-tune
        params = pruned_params
        for step in range(steps):
            params, opt_state, loss = update_step(params, opt_state)
            
            if (step + 1) % max(1, steps // 10) == 0:
                print(f"  Step {step + 1}/{steps}, Loss: {loss:.6f}")
        
        return params
    
    def distill(self, 
               hidden_dims: List[int] = None,
               num_layers: int = None,
               num_samples: int = 10000,
               num_steps: int = 5000) -> Tuple[NormalizingFlow, Dict, Dict]:
        """
        Distill the flow into a smaller model.
        
        Args:
            hidden_dims: Hidden dimensions for the distilled model (default: same as original)
            num_layers: Number of layers for the distilled model (default: same as original)
            num_samples: Number of samples to generate for distillation
            num_steps: Number of training steps
            
        Returns:
            Tuple of (distilled flow, distilled parameters, evaluation metrics)
        """
        print(f"Distilling flow into a smaller model...")
        start_time = time.time()
        
        # Get original flow configuration
        if hasattr(self.flow.flow, 'hidden_dims'):
            original_hidden_dims = self.flow.flow.hidden_dims
        else:
            original_hidden_dims = [64, 64]  # Default
        
        if hasattr(self.flow.flow, 'num_layers'):
            original_num_layers = self.flow.flow.num_layers
        else:
            original_num_layers = 4  # Default
        
        # Set distilled model configuration
        hidden_dims = hidden_dims or original_hidden_dims
        num_layers = num_layers or original_num_layers
        
        print(f"Original model: {original_num_layers} layers, {original_hidden_dims} hidden dims")
        print(f"Distilled model: {num_layers} layers, {hidden_dims} hidden dims")
        
        # Create a smaller flow with the same type
        flow_class = self.flow.flow.__class__
        
        if flow_class.__name__ == "RealNVP":
            from normflow.flows.coupling import RealNVP
            distilled_flow = RealNVP(
                dim=self.flow.dim,
                hidden_dims=hidden_dims,
                num_layers=num_layers
            )
        elif flow_class.__name__ == "NSF":
            from normflow.flows.spline import NSF
            distilled_flow = NSF(
                dim=self.flow.dim,
                hidden_dims=hidden_dims,
                num_layers=num_layers
            )
        else:
            print(f"Distillation not supported for {flow_class.__name__}")
            return None, None, {}
        
        # Wrap in a NormalizingFlow object
        distilled_normflow = NormalizingFlow(
            flow=distilled_flow,
            base_log_prob_fn=self.flow.base_log_prob_fn,
            base_sample_fn=self.flow.base_sample_fn,
            name=f"Distilled_{self.flow.name}"
        )
        
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        distilled_params = distilled_normflow.init_params(key)
        
        # Sample from the original flow for distillation
        key = jax.random.PRNGKey(1)
        samples = self.flow.sample(self.original_params, num_samples, key)
        
        # Compute log probabilities from the original flow
        original_log_probs = self.flow.log_prob(self.original_params, samples)
        
        # Define loss function for distillation
        @jax.jit
        def distillation_loss(params):
            # Compute log probabilities from the distilled flow
            distilled_log_probs = distilled_normflow.log_prob(params, samples)
            
            # Compute KL divergence: KL(original || distilled)
            # KL(p||q) = E_p[log(p) - log(q)]
            kl_div = jnp.mean(original_log_probs - distilled_log_probs)
            
            return kl_div
        
        # Create optimizer
        optimizer = optax.adam(learning_rate=1e-3)
        opt_state = optimizer.init(distilled_params)
        
        # Define update step
        @jax.jit
        def update_step(params, opt_state):
            loss, grads = jax.value_and_grad(distillation_loss)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss
        
        # Train the distilled model
        params = distilled_params
        loss_history = []
        
        for step in range(num_steps):
            params, opt_state, loss = update_step(params, opt_state)
            loss_history.append(float(loss))
            
            if (step + 1) % max(1, num_steps // 10) == 0:
                print(f"  Step {step + 1}/{num_steps}, Loss: {loss:.6f}")
        
        # Evaluate distillation
        metrics = self._evaluate_compression(params)
        metrics['time'] = time.time() - start_time
        metrics['final_kl_divergence'] = float(loss_history[-1])
        
        print(f"Distillation completed in {metrics['time']:.2f} seconds")
        print(f"Compressed size: {metrics['compressed_size'] / 1024:.2f} KB "
             f"({metrics['compression_ratio']:.2f}x compression)")
        
        if self.test_data is not None:
            print(f"NLL change: {metrics['nll_change']:.6f} "
                f"({metrics['nll_ratio']:.4f}x original)")
        
        return distilled_normflow, params, metrics
