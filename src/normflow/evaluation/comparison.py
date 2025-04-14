"""
Comparison utilities for normalizing flows and covariance matrices.
"""

import time
from typing import Dict, List, Optional, Tuple, Union, Callable

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
import tensorflow as tf
import zfit
from zfit import z

from normflow.flows.base import Flow, NormalizingFlow


class CovarianceComparer:
    """
    Compare normalizing flows with multivariate Gaussian approximations.
    
    This class provides methods for comparing the performance of normalizing
    flows against covariance matrix approximations (multivariate Gaussians).
    """
    
    def __init__(self, 
                flow: NormalizingFlow,
                flow_params: Dict,
                data: Union[np.ndarray, jnp.ndarray, zfit.Data],
                metrics: Optional[List[str]] = None):
        """
        Initialize a CovarianceComparer.
        
        Args:
            flow: The normalizing flow to evaluate
            flow_params: Flow parameters
            data: Data used for fitting/evaluation
            metrics: List of metrics to compute (default: ['likelihood', 'kl_divergence', 'entropy'])
        """
        self.flow = flow
        self.flow_params = flow_params
        
        # Convert data to numpy array if needed
        if isinstance(data, zfit.Data):
            self.data = zfit.run(data.value())
        elif isinstance(data, jnp.ndarray):
            self.data = np.array(data)
        else:
            self.data = data
        
        # Check data shape
        if self.data.shape[1] != flow.dim:
            raise ValueError(f"Data dimensions ({self.data.shape[1]}) do not match "
                            f"flow dimensions ({flow.dim})")
        
        # Set up metrics
        self.metrics = metrics or ['likelihood', 'kl_divergence', 'entropy', 
                                  'wasserstein', 'ks_test']
        
        # Fit multivariate Gaussian to the data
        self.mean, self.cov = self._fit_multivariate_gaussian()
        
        # Create a function to compute log probability under the Gaussian
        self.gauss_log_prob_fn = self._create_gaussian_log_prob_fn()
    
    def _fit_multivariate_gaussian(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a multivariate Gaussian to the data.
        
        Returns:
            Tuple of (mean, covariance)
        """
        # Compute mean and covariance
        mean = np.mean(self.data, axis=0)
        cov = np.cov(self.data, rowvar=False)
        
        return mean, cov
    
    def _create_gaussian_log_prob_fn(self) -> Callable:
        """
        Create a function to compute log probability under the Gaussian.
        
        Returns:
            Function to compute log probability
        """
        mean = jnp.array(self.mean)
        
        # Make sure the covariance matrix is positive definite
        try:
            # Try to compute Cholesky decomposition
            if self.flow.dim > 1:
                L = np.linalg.cholesky(self.cov)
                cov = jnp.array(self.cov)
            else:
                cov = jnp.array(self.cov).reshape(1, 1)
        except np.linalg.LinAlgError:
            # If not positive definite, add a small regularization
            print("Warning: Covariance matrix not positive definite, adding small diagonal term")
            reg_cov = self.cov + np.eye(self.cov.shape[0]) * 1e-6
            cov = jnp.array(reg_cov)
        
        # For 1D case
        if self.flow.dim == 1:
            def log_prob_fn(x):
                return jax.scipy.stats.norm.logpdf(
                    x.flatten(), loc=mean[0], scale=jnp.sqrt(cov[0, 0])
                )
        else:
            # For multivariate case
            # Precompute some terms
            dim = mean.shape[0]
            inv_cov = jnp.linalg.inv(cov)
            logdet_cov = jnp.log(jnp.linalg.det(cov))
            norm_const = -0.5 * (dim * jnp.log(2 * jnp.pi) + logdet_cov)
            
            def log_prob_fn(x):
                # Compute Mahalanobis distance: (x - μ)^T Σ^-1 (x - μ)
                x_centered = x - mean
                mahalanobis = jnp.sum(x_centered * (x_centered @ inv_cov), axis=1)
                
                # Compute log prob
                return norm_const - 0.5 * mahalanobis
        
        return log_prob_fn
    
    def compare_likelihood(self, 
                          test_data: Optional[Union[np.ndarray, jnp.ndarray, zfit.Data]] = None) -> Tuple[float, float]:
        """
        Compare log-likelihood of flow and multivariate Gaussian.
        
        Args:
            test_data: Data to evaluate on (default: same as training data)
            
        Returns:
            Tuple of (flow log-likelihood, Gaussian log-likelihood)
        """
        # Use training data if test data not provided
        if test_data is None:
            test_data = self.data
        elif isinstance(test_data, zfit.Data):
            test_data = zfit.run(test_data.value())
        elif isinstance(test_data, jnp.ndarray):
            test_data = np.array(test_data)
        
        # Convert to JAX array
        test_data_jax = jnp.asarray(test_data)
        
        # Compute log-likelihood for flow
        flow_log_probs = self.flow.log_prob(self.flow_params, test_data_jax)
        flow_mean_log_like = jnp.mean(flow_log_probs)
        
        # Compute log-likelihood for Gaussian
        gauss_log_probs = self.gauss_log_prob_fn(test_data_jax)
        gauss_mean_log_like = jnp.mean(gauss_log_probs)
        
        return float(flow_mean_log_like), float(gauss_mean_log_like)
    
    def estimate_kl_divergence(self, n_samples: int = 10000) -> Tuple[float, float]:
        """
        Estimate KL divergence between flow and Gaussian using Monte Carlo.
        
        Computes both KL(flow||gauss) and KL(gauss||flow).
        
        Args:
            n_samples: Number of samples to use for estimation
            
        Returns:
            Tuple of (KL(flow||gauss), KL(gauss||flow))
        """
        # Sample from flow
        key = jax.random.PRNGKey(0)
        flow_samples = self.flow.sample(self.flow_params, n_samples, key)
        
        # Sample from Gaussian
        key = jax.random.PRNGKey(1)
        if self.flow.dim == 1:
            gauss_samples = jax.random.normal(key, (n_samples,)) * jnp.sqrt(self.cov[0, 0]) + self.mean[0]
            gauss_samples = gauss_samples.reshape(-1, 1)
        else:
            gauss_samples = jax.random.multivariate_normal(key, self.mean, self.cov, (n_samples,))
        
        # Compute log probabilities for flow samples
        flow_log_prob_flow = self.flow.log_prob(self.flow_params, flow_samples)
        gauss_log_prob_flow = self.gauss_log_prob_fn(flow_samples)
        
        # Compute log probabilities for Gaussian samples
        flow_log_prob_gauss = self.flow.log_prob(self.flow_params, gauss_samples)
        gauss_log_prob_gauss = self.gauss_log_prob_fn(gauss_samples)
        
        # Compute KL divergence estimates
        # KL(flow||gauss) = E_{x~flow}[log(flow(x)) - log(gauss(x))]
        kl_flow_gauss = jnp.mean(flow_log_prob_flow - gauss_log_prob_flow)
        
        # KL(gauss||flow) = E_{x~gauss}[log(gauss(x)) - log(flow(x))]
        kl_gauss_flow = jnp.mean(gauss_log_prob_gauss - flow_log_prob_gauss)
        
        return float(kl_flow_gauss), float(kl_gauss_flow)
    
    def estimate_entropy(self, n_samples: int = 10000) -> Tuple[float, float]:
        """
        Estimate entropy of flow and Gaussian.
        
        Args:
            n_samples: Number of samples to use for estimation
            
        Returns:
            Tuple of (flow entropy, Gaussian entropy)
        """
        # Sample from flow
        key = jax.random.PRNGKey(1)
        flow_samples = self.flow.sample(self.flow_params, n_samples, key)
        
        # Compute log probability for flow samples
        flow_log_prob = self.flow.log_prob(self.flow_params, flow_samples)
        
        # Estimate flow entropy: -E_{x~flow}[log(flow(x))]
        flow_entropy = -jnp.mean(flow_log_prob)
        
        # Compute Gaussian entropy (analytical)
        k = self.flow.dim
        gauss_entropy = k / 2 * (1 + jnp.log(2 * jnp.pi)) + 0.5 * jnp.log(jnp.linalg.det(jnp.array(self.cov)))
        
        return float(flow_entropy), float(gauss_entropy)
    
    def estimate_wasserstein(self, n_samples: int = 1000, p: int = 2) -> float:
        """
        Estimate Wasserstein distance between flow and Gaussian using samples.
        
        Args:
            n_samples: Number of samples to use for estimation
            p: Order of the Wasserstein distance
            
        Returns:
            Estimated Wasserstein distance
        """
        try:
            from scipy.stats import wasserstein_distance
        except ImportError:
            raise ImportError("SciPy is required for this feature. "
                            "Install it with 'pip install scipy'.")
        
        # For multivariate distributions, approximate using sliced Wasserstein distance
        if self.flow.dim > 1:
            return self._estimate_sliced_wasserstein(n_samples, p)
        
        # For 1D, we can use SciPy's implementation
        
        # Sample from flow
        key = jax.random.PRNGKey(2)
        flow_samples = self.flow.sample(self.flow_params, n_samples, key)
        flow_samples = np.array(flow_samples).flatten()
        
        # Sample from Gaussian
        key = jax.random.PRNGKey(3)
        gauss_samples = jax.random.normal(key, (n_samples,)) * np.sqrt(self.cov[0, 0]) + self.mean[0]
        gauss_samples = np.array(gauss_samples).flatten()
        
        # Compute Wasserstein distance
        w_dist = wasserstein_distance(flow_samples, gauss_samples)
        
        return w_dist
    
    def _estimate_sliced_wasserstein(self, n_samples: int = 1000, p: int = 2, n_projections: int = 100) -> float:
        """
        Estimate sliced Wasserstein distance for multivariate distributions.
        
        Args:
            n_samples: Number of samples to use for estimation
            p: Order of the Wasserstein distance
            n_projections: Number of random projections
            
        Returns:
            Estimated sliced Wasserstein distance
        """
        try:
            from scipy.stats import wasserstein_distance
        except ImportError:
            raise ImportError("SciPy is required for this feature. "
                             "Install it with 'pip install scipy'.")
        
        # Sample from flow
        key = jax.random.PRNGKey(3)
        flow_samples = self.flow.sample(self.flow_params, n_samples, key)
        flow_samples = np.array(flow_samples)
        
        # Sample from Gaussian
        key = jax.random.PRNGKey(4)
        gauss_samples = jax.random.multivariate_normal(key, self.mean, self.cov, (n_samples,))
        gauss_samples = np.array(gauss_samples)
        
        # Generate random projections
        key = jax.random.PRNGKey(4)
        dim = self.flow.dim
        
        # Initialize sum of distances
        total_dist = 0.0
        
        for i in range(n_projections):
            # Generate a random unit vector for projection
            key, subkey = jax.random.split(key)
            direction = jax.random.normal(subkey, (dim,))
            direction = direction / np.linalg.norm(direction)
            
            # Project samples onto this direction
            flow_proj = np.dot(flow_samples, direction)
            gauss_proj = np.dot(gauss_samples, direction)
            
            # Compute Wasserstein distance of the projections
            w_dist = wasserstein_distance(flow_proj, gauss_proj)
            
            # Add to total
            total_dist += w_dist ** p
        
        # Return average distance
        return (total_dist / n_projections) ** (1/p)
    
    def run_ks_test(self) -> Dict[str, Tuple[float, float]]:
        """
        Run Kolmogorov-Smirnov test for each dimension.
        
        Returns:
            Dictionary mapping dimension index to (KS statistic, p-value)
        """
        try:
            from scipy.stats import ks_2samp
        except ImportError:
            raise ImportError("SciPy is required for this feature. "
                             "Install it with 'pip install scipy'.")
        
        # Sample from flow
        key = jax.random.PRNGKey(5)
        n_samples = len(self.data)
        flow_samples = self.flow.sample(self.flow_params, n_samples, key)
        flow_samples = np.array(flow_samples)
        
        # Sample from Gaussian
        key = jax.random.PRNGKey(6)
        if self.flow.dim == 1:
            gauss_samples = jax.random.normal(key, (n_samples,)) * np.sqrt(self.cov[0, 0]) + self.mean[0]
            gauss_samples = np.array(gauss_samples).reshape(-1, 1)
        else:
            gauss_samples = jax.random.multivariate_normal(key, self.mean, self.cov, (n_samples,))
            gauss_samples = np.array(gauss_samples)
        
        # Run KS test for each dimension
        results = {}
        for i in range(self.flow.dim):
            # Extract samples for this dimension
            flow_dim = flow_samples[:, i]
            gauss_dim = gauss_samples[:, i]
            
            # Run KS test
            ks_stat, p_value = ks_2samp(flow_dim, gauss_dim)
            
            # Store result
            results[i] = (ks_stat, p_value)
        
        return results
    
    def run_all_comparisons(self) -> Dict[str, Dict[str, float]]:
        """
        Run all comparison metrics.
        
        Returns:
            Dictionary with all comparison results
        """
        results = {}
        
        if 'likelihood' in self.metrics:
            flow_ll, gauss_ll = self.compare_likelihood()
            results['likelihood'] = {
                'flow': float(flow_ll),
                'gaussian': float(gauss_ll),
                'difference': float(flow_ll - gauss_ll)
            }
        
        if 'kl_divergence' in self.metrics:
            kl_flow_gauss, kl_gauss_flow = self.estimate_kl_divergence()
            results['kl_divergence'] = {
                'flow_to_gaussian': float(kl_flow_gauss),
                'gaussian_to_flow': float(kl_gauss_flow),
                'symmetric': float(kl_flow_gauss + kl_gauss_flow) / 2
            }
        
        if 'entropy' in self.metrics:
            flow_entropy, gauss_entropy = self.estimate_entropy()
            results['entropy'] = {
                'flow': float(flow_entropy),
                'gaussian': float(gauss_entropy),
                'difference': float(flow_entropy - gauss_entropy)
            }
        
        if 'wasserstein' in self.metrics:
            w_dist = self.estimate_wasserstein()
            results['wasserstein'] = {
                'distance': float(w_dist)
            }
        
        if 'ks_test' in self.metrics:
            ks_results = self.run_ks_test()
            results['ks_test'] = {
                f'dim_{i}': {
                    'statistic': float(stat),
                    'p_value': float(p_value)
                } for i, (stat, p_value) in ks_results.items()
            }
        
        return results


class CovarianceBasedSampler:
    """
    Sample from a multivariate Gaussian approximation of the flow.
    
    This class provides methods for sampling from a multivariate Gaussian
    approximation of a normalizing flow, which can be faster than sampling
    from the flow directly.
    """
    
    def __init__(self, 
                flow: NormalizingFlow, 
                flow_params: Dict,
                n_samples_for_estimation: int = 10000):
        """
        Initialize a CovarianceBasedSampler.
        
        Args:
            flow: The normalizing flow
            flow_params: Flow parameters
            n_samples_for_estimation: Number of samples to use for estimating
                                      the multivariate Gaussian
        """
        self.flow = flow
        self.flow_params = flow_params
        
        # Sample from the flow to estimate mean and covariance
        key = jax.random.PRNGKey(0)
        self.samples = flow.sample(flow_params, n_samples_for_estimation, key)
        
        # Compute mean and covariance
        self.mean = jnp.mean(self.samples, axis=0)
        self.cov = jnp.cov(self.samples, rowvar=False)
    
    def sample(self, n_samples: int, key: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Sample from the multivariate Gaussian approximation.
        
        Args:
            n_samples: Number of samples to generate
            key: JAX random key
            
        Returns:
            Samples from the multivariate Gaussian
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Sample from multivariate Gaussian
        if self.flow.dim == 1:
            samples = jax.random.normal(key, (n_samples,)) * jnp.sqrt(self.cov[0, 0]) + self.mean[0]
            samples = samples.reshape(-1, 1)
        else:
            samples = jax.random.multivariate_normal(
                key, self.mean, self.cov, shape=(n_samples,))
        
        return samples
    
    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log probability of samples under the multivariate Gaussian.
        
        Args:
            x: Batch of samples, shape (batch_size, dim)
            
        Returns:
            Log probabilities, shape (batch_size,)
        """
        # Compute log probability of multivariate Gaussian
        dim = self.mean.shape[0]
        
        if dim == 1:
            # For 1D case, use normal distribution
            return jax.scipy.stats.norm.logpdf(
                x.flatten(), loc=self.mean[0], scale=jnp.sqrt(self.cov[0, 0])
            )
        else:
            # Compute Mahalanobis distance: (x - μ)^T Σ^-1 (x - μ)
            x_centered = x - self.mean
            
            # Use Cholesky decomposition for numerical stability
            L = jnp.linalg.cholesky(self.cov)
            solved = jax.scipy.linalg.solve_triangular(L, x_centered.T, lower=True)
            mahalanobis = jnp.sum(solved**2, axis=0)
            
            # Compute log determinant of covariance
            log_det_cov = 2 * jnp.sum(jnp.log(jnp.diag(L)))
            
            # Compute log probability
            log_prob = -0.5 * (dim * jnp.log(2 * jnp.pi) + log_det_cov + mahalanobis)
            
            return log_prob
    
    def compare_sampling_speed(self, 
                              n_samples: int = 1000, 
                              n_trials: int = 10) -> Dict[str, float]:
        """
        Compare sampling speed of flow and multivariate Gaussian.
        
        Args:
            n_samples: Number of samples to generate in each trial
            n_trials: Number of trials
            
        Returns:
            Dictionary with timing results
        """
        # Time sampling from flow
        flow_times = []
        for i in range(n_trials):
            key = jax.random.PRNGKey(i)
            
            start_time = time.time()
            _ = self.flow.sample(self.flow_params, n_samples, key)
            end_time = time.time()
            
            flow_times.append(end_time - start_time)
        
        # Time sampling from multivariate Gaussian
        mvn_times = []
        for i in range(n_trials):
            key = jax.random.PRNGKey(i + n_trials)
            
            start_time = time.time()
            _ = self.sample(n_samples, key)
            end_time = time.time()
            
            mvn_times.append(end_time - start_time)
        
        # Compute statistics
        flow_mean = np.mean(flow_times)
        flow_std = np.std(flow_times)
        mvn_mean = np.mean(mvn_times)
        mvn_std = np.std(mvn_times)
        
        return {
            'flow_mean_time': flow_mean,
            'flow_std_time': flow_std,
            'mvn_mean_time': mvn_mean,
            'mvn_std_time': mvn_std,
            'speedup_factor': flow_mean / mvn_mean
        }


class ZfitComparer:
    """
    Compare normalizing flows with zfit models.
    
    This class provides methods for comparing the performance of normalizing
    flows against fitted zfit models.
    """
    
    def __init__(self, 
                flow: NormalizingFlow,
                flow_params: Dict,
                zfit_model: zfit.pdf.BasePDF,
                data: Union[np.ndarray, jnp.ndarray, zfit.Data],
                metrics: Optional[List[str]] = None):
        """
        Initialize a ZfitComparer.
        
        Args:
            flow: The normalizing flow to evaluate
            flow_params: Flow parameters
            zfit_model: Fitted zfit model
            data: Data used for fitting/evaluation
            metrics: List of metrics to compute (default: ['likelihood', 'kl_divergence'])
        """
        self.flow = flow
        self.flow_params = flow_params
        self.zfit_model = zfit_model
        
        # Convert data to numpy array if needed
        if isinstance(data, zfit.Data):
            self.data = zfit.run(data.value())
        elif isinstance(data, jnp.ndarray):
            self.data = np.array(data)
        else:
            self.data = data
        
        # Check data shape
        if self.data.shape[1] != flow.dim:
            raise ValueError(f"Data dimensions ({self.data.shape[1]}) do not match "
                            f"flow dimensions ({flow.dim})")
        
        # Set up metrics
        self.metrics = metrics or ['likelihood', 'kl_divergence']
    
    def _zfit_log_prob_fn(self, x: np.ndarray) -> np.ndarray:
        """
        Compute log probability under the zfit model.
        
        Args:
            x: Batch of data points
            
        Returns:
            Log probabilities
        """
        # Convert to TensorFlow tensor
        x_tf = tf.convert_to_tensor(x, dtype=tf.float64)
        
        # Compute log probability
        log_prob = self.zfit_model.log_prob(x_tf)
        
        # Run the TensorFlow graph
        return zfit.run(log_prob)
    
    def compare_likelihood(self, 
                          test_data: Optional[Union[np.ndarray, jnp.ndarray, zfit.Data]] = None) -> Tuple[float, float]:
        """
        Compare log-likelihood of flow and zfit model.
        
        Args:
            test_data: Data to evaluate on (default: same as training data)
            
        Returns:
            Tuple of (flow log-likelihood, zfit model log-likelihood)
        """
        # Use training data if test data not provided
        if test_data is None:
            test_data = self.data
        elif isinstance(test_data, zfit.Data):
            test_data = zfit.run(test_data.value())
        elif isinstance(test_data, jnp.ndarray):
            test_data = np.array(test_data)
        
        # Convert to JAX array for flow
        test_data_jax = jnp.asarray(test_data)
        
        # Compute log-likelihood for flow
        flow_log_probs = self.flow.log_prob(self.flow_params, test_data_jax)
        flow_mean_log_like = jnp.mean(flow_log_probs)
        
        # Compute log-likelihood for zfit model
        zfit_log_probs = self._zfit_log_prob_fn(test_data)
        zfit_mean_log_like = np.mean(zfit_log_probs)
        
        return float(flow_mean_log_like), float(zfit_mean_log_like)
    
    def estimate_kl_divergence(self, n_samples: int = 10000) -> Tuple[float, float]:
        """
        Estimate KL divergence between flow and zfit model using Monte Carlo.
        
        Computes both KL(flow||zfit) and KL(zfit||flow).
        
        Args:
            n_samples: Number of samples to use for estimation
            
        Returns:
            Tuple of (KL(flow||zfit), KL(zfit||flow))
        """
        # Sample from flow
        key = jax.random.PRNGKey(0)
        flow_samples = self.flow.sample(self.flow_params, n_samples, key)
        flow_samples_np = np.array(flow_samples)
        
        # Sample from zfit model
        zfit_samples = self.zfit_model.sample(n_samples)
        zfit_samples_np = zfit.run(zfit_samples)
        zfit_samples_jax = jnp.array(zfit_samples_np)
        
        # Compute log probabilities for flow samples
        flow_log_prob_flow = self.flow.log_prob(self.flow_params, flow_samples)
        zfit_log_prob_flow = self._zfit_log_prob_fn(flow_samples_np)
        
        # Compute log probabilities for zfit samples
        flow_log_prob_zfit = self.flow.log_prob(self.flow_params, zfit_samples_jax)
        zfit_log_prob_zfit = self._zfit_log_prob_fn(zfit_samples_np)
        
        # Compute KL divergence estimates
        # KL(flow||zfit) = E_{x~flow}[log(flow(x)) - log(zfit(x))]
        kl_flow_zfit = np.mean(flow_log_prob_flow - zfit_log_prob_flow)
        
        # KL(zfit||flow) = E_{x~zfit}[log(zfit(x)) - log(flow(x))]
        kl_zfit_flow = np.mean(zfit_log_prob_zfit - flow_log_prob_zfit)
        
        return float(kl_flow_zfit), float(kl_zfit_flow)
    
    def run_all_comparisons(self) -> Dict[str, Dict[str, float]]:
        """
        Run all comparison metrics.
        
        Returns:
            Dictionary with all comparison results
        """
        results = {}
        
        if 'likelihood' in self.metrics:
            flow_ll, zfit_ll = self.compare_likelihood()
            results['likelihood'] = {
                'flow': float(flow_ll),
                'zfit_model': float(zfit_ll),
                'difference': float(flow_ll - zfit_ll)
            }
        
        if 'kl_divergence' in self.metrics:
            kl_flow_zfit, kl_zfit_flow = self.estimate_kl_divergence()
            results['kl_divergence'] = {
                'flow_to_zfit': float(kl_flow_zfit),
                'zfit_to_flow': float(kl_zfit_flow),
                'symmetric': float(kl_flow_zfit + kl_zfit_flow) / 2
            }
        
        return results
