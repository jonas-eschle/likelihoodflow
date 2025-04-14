"""
Tests for normalizing flow implementations.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from normflow.flows.base import Flow, NormalizingFlow, SequentialFlow
from normflow.flows.base import standard_normal_log_prob, standard_normal_sample
from normflow.flows.coupling import RealNVP
from normflow.flows.spline import NSF


@pytest.fixture
def key():
    """Fixture for JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_data_1d():
    """Fixture for 1D sample data."""
    return np.random.normal(0, 1, (1000, 1))


@pytest.fixture
def sample_data_2d():
    """Fixture for 2D sample data."""
    return np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 1000)


def test_realnvp_forward_inverse_1d(key, sample_data_1d):
    """Test RealNVP forward and inverse consistency in 1D."""
    flow = RealNVP(dim=1, hidden_dims=[32, 32], num_layers=4)
    params = flow.init_params(key)
    
    # Convert data to JAX array
    data = jnp.array(sample_data_1d)
    
    # Forward pass
    transformed, log_det = flow.forward(params, data)
    
    # Inverse pass
    reconstructed, inv_log_det = flow.inverse(params, transformed)
    
    # Check reconstruction
    np.testing.assert_allclose(data, reconstructed, rtol=1e-5, atol=1e-5)
    
    # Check log determinants
    np.testing.assert_allclose(-log_det, inv_log_det, rtol=1e-5, atol=1e-5)


def test_realnvp_forward_inverse_2d(key, sample_data_2d):
    """Test RealNVP forward and inverse consistency in 2D."""
    flow = RealNVP(dim=2, hidden_dims=[32, 32], num_layers=4)
    params = flow.init_params(key)
    
    # Convert data to JAX array
    data = jnp.array(sample_data_2d)
    
    # Forward pass
    transformed, log_det = flow.forward(params, data)
    
    # Inverse pass
    reconstructed, inv_log_det = flow.inverse(params, transformed)
    
    # Check reconstruction
    np.testing.assert_allclose(data, reconstructed, rtol=1e-5, atol=1e-5)
    
    # Check log determinants
    np.testing.assert_allclose(-log_det, inv_log_det, rtol=1e-5, atol=1e-5)


def test_nsf_forward_inverse_1d(key, sample_data_1d):
    """Test Neural Spline Flow forward and inverse consistency in 1D."""
    flow = NSF(dim=1, hidden_dims=[32, 32], num_layers=4, num_bins=8)
    params = flow.init_params(key)
    
    # Convert data to JAX array
    data = jnp.array(sample_data_1d)
    
    # Forward pass
    transformed, log_det = flow.forward(params, data)
    
    # Inverse pass
    reconstructed, inv_log_det = flow.inverse(params, transformed)
    
    # Check reconstruction
    np.testing.assert_allclose(data, reconstructed, rtol=1e-4, atol=1e-4)
    
    # Check log determinants
    np.testing.assert_allclose(-log_det, inv_log_det, rtol=1e-4, atol=1e-4)


def test_nsf_forward_inverse_2d(key, sample_data_2d):
    """Test Neural Spline Flow forward and inverse consistency in 2D."""
    flow = NSF(dim=2, hidden_dims=[32, 32], num_layers=4, num_bins=8)
    params = flow.init_params(key)
    
    # Convert data to JAX array
    data = jnp.array(sample_data_2d)
    
    # Forward pass
    transformed, log_det = flow.forward(params, data)
    
    # Inverse pass
    reconstructed, inv_log_det = flow.inverse(params, transformed)
    
    # Check reconstruction
    np.testing.assert_allclose(data, reconstructed, rtol=1e-4, atol=1e-4)
    
    # Check log determinants
    np.testing.assert_allclose(-log_det, inv_log_det, rtol=1e-4, atol=1e-4)


def test_sequential_flow(key, sample_data_2d):
    """Test sequential flow composition."""
    # Create component flows
    flow1 = RealNVP(dim=2, hidden_dims=[32], num_layers=2)
    flow2 = NSF(dim=2, hidden_dims=[32], num_layers=2, num_bins=8)
    
    # Create sequential flow
    seq_flow = SequentialFlow(flows=[flow1, flow2])
    params = seq_flow.init_params(key)
    
    # Convert data to JAX array
    data = jnp.array(sample_data_2d)
    
    # Forward pass
    transformed, log_det = seq_flow.forward(params, data)
    
    # Inverse pass
    reconstructed, inv_log_det = seq_flow.inverse(params, transformed)
    
    # Check reconstruction
    np.testing.assert_allclose(data, reconstructed, rtol=1e-4, atol=1e-4)
    
    # Check log determinants
    np.testing.assert_allclose(-log_det, inv_log_det, rtol=1e-4, atol=1e-4)


def test_normalizing_flow_log_prob(key, sample_data_2d):
    """Test normalizing flow log probability computation."""
    # Create flow
    flow = RealNVP(dim=2, hidden_dims=[32, 32], num_layers=4)
    
    # Wrap in NormalizingFlow
    normflow = NormalizingFlow(
        flow=flow,
        base_log_prob_fn=standard_normal_log_prob,
        base_sample_fn=standard_normal_sample,
        name="TestFlow"
    )
    
    # Initialize parameters
    params = normflow.init_params(key)
    
    # Convert data to JAX array
    data = jnp.array(sample_data_2d)
    
    # Compute log probability
    log_prob = normflow.log_prob(params, data)
    
    # Check shape
    assert log_prob.shape == (len(data),)
    
    # Check finite values
    assert np.all(np.isfinite(log_prob))


def test_normalizing_flow_sampling(key):
    """Test normalizing flow sampling."""
    # Create flow
    flow = RealNVP(dim=2, hidden_dims=[32, 32], num_layers=4)
    
    # Wrap in NormalizingFlow
    normflow = NormalizingFlow(
        flow=flow,
        base_log_prob_fn=standard_normal_log_prob,
        base_sample_fn=standard_normal_sample,
        name="TestFlow"
    )
    
    # Initialize parameters
    params = normflow.init_params(key)
    
    # Sample from flow
    n_samples = 1000
    samples = normflow.sample(params, n_samples, key)
    
    # Check shape
    assert samples.shape == (n_samples, 2)
    
    # Check finite values
    assert np.all(np.isfinite(samples))
