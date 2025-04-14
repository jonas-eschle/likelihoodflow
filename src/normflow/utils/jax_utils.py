"""
JAX-specific utility functions.
"""

from typing import Dict, List, Tuple, Callable, Any, Union, Optional

import jax
import jax.numpy as jnp
import numpy as np


def ensure_jax_array(x: Union[np.ndarray, jnp.ndarray, list, tuple]) -> jnp.ndarray:
    """
    Ensure that the input is a JAX array.

    Args:
        x: Input data

    Returns:
        JAX array
    """
    if isinstance(x, jnp.ndarray):
        return x
    return jnp.asarray(x)


def batch_apply(func: Callable, 
               data: jnp.ndarray, 
               batch_size: int, 
               *args, 
               **kwargs) -> List[Any]:
    """
    Apply a function to batches of data.

    This is useful for large datasets that don't fit in memory
    or when we want to avoid excessive memory usage.

    Args:
        func: Function to apply
        data: Input data
        batch_size: Batch size
        *args: Additional arguments to pass to func
        **kwargs: Additional keyword arguments to pass to func

    Returns:
        List of results for each batch
    """
    n_samples = len(data)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    results = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch = data[start_idx:end_idx]
        
        result = func(batch, *args, **kwargs)
        results.append(result)
    
    return results


def compute_parameter_norm(params: Dict, norm_type: str = 'l2') -> float:
    """
    Compute the norm of all parameters in a dictionary.

    Args:
        params: Parameter dictionary
        norm_type: Type of norm ('l1', 'l2', 'inf')

    Returns:
        Norm value
    """
    # Flatten parameters
    all_params = []
    
    def collect_params(p):
        if isinstance(p, (jnp.ndarray, np.ndarray)):
            all_params.append(jnp.asarray(p).flatten())
        elif isinstance(p, dict):
            for v in p.values():
                collect_params(v)
        elif isinstance(p, (list, tuple)):
            for v in p:
                collect_params(v)
    
    collect_params(params)
    
    if not all_params:
        return 0.0
    
    # Concatenate all parameters
    flat_params = jnp.concatenate(all_params)
    
    # Compute norm
    if norm_type == 'l1':
        return float(jnp.sum(jnp.abs(flat_params)))
    elif norm_type == 'l2':
        return float(jnp.sqrt(jnp.sum(flat_params**2)))
    elif norm_type == 'inf':
        return float(jnp.max(jnp.abs(flat_params)))
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def get_gpu_memory_usage():
    """
    Get GPU memory usage if available.

    Returns:
        Dictionary with memory usage information or None if not available
    """
    try:
        import subprocess
        import re
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                                          '--format=csv,nounits,noheader'])
        memory_info = output.decode('utf-8').strip().split('\n')
        gpu_memory = []
        
        for info in memory_info:
            used, total = map(int, info.split(','))
            gpu_memory.append({'used': used, 'total': total, 'free': total - used})
        
        return gpu_memory
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def configure_jax_for_64bit():
    """Configure JAX to use 64-bit precision."""
    jax.config.update("jax_enable_x64", True)
