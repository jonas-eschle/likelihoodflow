"""
zfit-specific utility functions.
"""

from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import tensorflow as tf
import zfit
from zfit import z


def convert_data_to_zfit(data: np.ndarray, 
                        obs_names: Optional[List[str]] = None,
                        limits: Optional[List[Tuple[float, float]]] = None) -> zfit.Data:
    """
    Convert numpy data to zfit Data.

    Args:
        data: Input data as numpy array
        obs_names: Observable names (default: 'x0', 'x1', etc.)
        limits: Limits for each dimension (default: determined from data)

    Returns:
        zfit Data object
    """
    # Get dimensions
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    dim = data.shape[1]
    
    # Create observable names if not provided
    if obs_names is None:
        obs_names = [f'x{i}' for i in range(dim)]
    
    # Determine limits if not provided
    if limits is None:
        limits = []
        for i in range(dim):
            min_val = np.min(data[:, i])
            max_val = np.max(data[:, i])
            # Add some padding
            padding = (max_val - min_val) * 0.1
            limits.append((min_val - padding, max_val + padding))
    
    # Create observable space
    obs_space = zfit.Space(obs=obs_names, limits=limits)
    
    # Create zfit data
    zfit_data = zfit.Data.from_numpy(obs=obs_space, array=data)
    
    return zfit_data


def create_gaussian_mixture(dim: int,
                          n_components: int,
                          obs_space: zfit.Space,
                          means: Optional[List[np.ndarray]] = None,
                          covs: Optional[List[np.ndarray]] = None,
                          weights: Optional[List[float]] = None) -> zfit.pdf.SumPDF:
    """
    Create a Gaussian mixture model.

    Args:
        dim: Dimensionality of the model
        n_components: Number of components
        obs_space: Observable space
        means: List of mean vectors (default: random)
        covs: List of covariance matrices (default: identity)
        weights: List of mixture weights (default: equal)

    Returns:
        zfit SumPDF
    """
    # Generate random means if not provided
    if means is None:
        rng = np.random.RandomState(0)
        means = [rng.randn(dim) * 3 for _ in range(n_components)]
    
    # Generate identity covariance matrices if not provided
    if covs is None:
        covs = [np.eye(dim) for _ in range(n_components)]
    
    # Generate equal weights if not provided
    if weights is None:
        weights = [1.0 / n_components] * n_components
    
    # Create components
    components = []
    fracs = []
    
    for i in range(n_components):
        # Create parameters
        if dim == 1:
            mu = zfit.Parameter(f"mu_{i}", means[i][0], -10, 10)
            sigma = zfit.Parameter(f"sigma_{i}", np.sqrt(covs[i][0, 0]), 0.1, 10)
            pdf = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs_space)
        else:
            mu = zfit.Parameter(f"mu_{i}", means[i], -10, 10)
            sigma = zfit.Parameter(f"sigma_{i}", covs[i], 0.1, 10)
            pdf = zfit.pdf.MultivariateNormal(mu=mu, sigma=sigma, obs=obs_space)
        
        components.append(pdf)
        
        # Create fraction parameters (except for the last component)
        if i < n_components - 1:
            frac = zfit.Parameter(f"frac_{i}", weights[i], 0.0, 1.0)
            fracs.append(frac)
    
    # Handle the last fraction to ensure sum to 1
    if n_components > 1:
        fracs_sum = sum(fracs)
        last_frac = 1.0 - fracs_sum
        fracs.append(last_frac)
        
        # Create the mixture
        mixture = zfit.pdf.SumPDF(pdfs=components, fracs=fracs)
    else:
        # Only one component
        mixture = components[0]
    
    return mixture


def extract_fit_parameters(fit_result: zfit.result.FitResult) -> Dict[str, float]:
    """
    Extract parameter values and errors from a fit result.

    Args:
        fit_result: zfit FitResult

    Returns:
        Dictionary of parameter values and errors
    """
    params = fit_result.params
    
    result_dict = {}
    for param, values in params.items():
        result_dict[param.name] = {
            'value': float(values['value']),
            'error': float(values.get('error', 0.0))
        }
    
    return result_dict


def evaluate_pdf_on_grid(pdf: zfit.pdf.BasePDF,
                       x_range: Tuple[float, float],
                       y_range: Tuple[float, float],
                       n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a 2D PDF on a grid.

    Args:
        pdf: zfit PDF
        x_range: Range for x dimension
        y_range: Range for y dimension
        n_points: Number of points in each dimension

    Returns:
        Tuple of (X, Y, Z) grids
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Combine into points
    points = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    # Evaluate PDF
    Z = pdf.pdf(points)
    Z = zfit.run(Z)
    
    # Reshape to grid
    Z = Z.reshape(n_points, n_points)
    
    return X, Y, Z
