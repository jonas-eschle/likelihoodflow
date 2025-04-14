"""
Example of using normalizing flows with zfit for unbinned fitting.

This example demonstrates:
1. Fitting a zfit model to data 
2. Training a normalizing flow with the NLL from the fit
3. Comparing the flow and zfit models
4. Optimizing the flow for storage
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import zfit
from zfit import z

from normflow.flows.base import NormalizingFlow, standard_normal_log_prob, standard_normal_sample
from normflow.flows import RealNVP
from normflow.fitting.zfit_interface import ZfitFlowInterface
from normflow.fitting.nll_loss import ZfitNLLLoss
from normflow.training.trainer import ZfitLossTrainer
from normflow.evaluation.comparison import CovarianceComparer, ZfitComparer
from normflow.storage.serialization import FlowSerializer

from src.normflow.evaluation.comparison import CovarianceBasedSampler
from src.normflow.training import trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Example of normalizing flows with zfit')
    parser.add_argument('--num-samples', type=int, default=10000,
                        help='Number of samples for training/testing')
    parser.add_argument('--dim', type=int, default=2,
                        help='Dimensionality of the samples')
    parser.add_argument('--num-params', type=int, default=5,
                        help='Number of free parameters in the fit')
    parser.add_argument('--num-train-steps', type=int, default=5000,
                        help='Number of flow training steps')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for flow training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-path', type=str, default='flow_model.json',
                        help='Path to save the trained flow')
    parser.add_argument('--num-flow-layers', type=int, default=4,
                        help='Number of coupling layers in the flow')
    parser.add_argument('--hidden-dims', type=str, default='64,64',
                        help='Hidden dimensions for coupling networks (comma-separated)')
    return parser.parse_args()


def generate_toy_data(num_samples, dim, num_params, seed=42):
    """
    Generate toy data for testing.
    
    This creates a multimodal distribution with some correlations.
    """
    rng = np.random.RandomState(seed)
    
    # Create a mixture of Gaussians
    means = rng.randn(2, dim) * 2  # Two components
    covs = [np.eye(dim) for _ in range(2)]
    
    # Add some correlations
    for i in range(len(covs)):
        if dim > 1:
            covs[i][0, 1] = 0.7
            covs[i][1, 0] = 0.7
    
    # Sample from the mixture
    weights = [0.6, 0.4]  # Mixture weights
    component = rng.choice(2, size=num_samples, p=weights)
    
    samples = np.zeros((num_samples, dim))
    for i in range(num_samples):
        c = component[i]
        samples[i] = rng.multivariate_normal(means[c], covs[c])
    
    return samples


def create_zfit_model(obs_space, num_params):
    """
    Create a zfit model with the specified number of free parameters.
    
    For simplicity, we use a Gaussian mixture model.
    """
    # Create parameters for the mixture
    dim = obs_space.n_obs
    n_components = num_params // (dim * 2)  # Each component has mean and cov params
    n_components = max(1, n_components)  # At least one component
    
    # Create mixture components
    components = []
    fracs = []
    
    for i in range(n_components):
        # Create mean and covariance parameters
        mu_name = f"mu_{i}"
        for d, ob in enumerate(obs_space.obs):
            mu = zfit.Parameter(mu_name, 0.0 + d ** 0.5, -50.0, 50.0)
            sigma_name = f"sigma_{i}"
            sigma = zfit.Parameter(sigma_name, 1.0 + (0.5 * d ** 0.5), 0.1, 50.0)
            component = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs_space.with_obs(ob))
        
            components.append(component)
        
        # Create fraction parameters (except for the last component)
        if i < n_components - 1:
            frac_name = f"frac_{i}"
            frac = zfit.Parameter(frac_name, 1.0 / n_components, 0.0, 1.0)
            fracs.append(frac)
    
    # Handle the last fraction to ensure sum to 1
    if n_components > 1:
        fracs_sum = sum(fracs)
        last_frac = 1.0 - fracs_sum
        fracs.append(last_frac)
        
        # Create the mixture
        model = zfit.pdf.SumPDF(pdfs=components, fracs=fracs)
    else:
        # Only one component
        model = components[0]
    
    return model


def create_flow(dim, num_layers, hidden_dims):
    """
    Create a normalizing flow model.
    """
    # Parse hidden dimensions
    hidden_dims = [int(d) for d in hidden_dims.split(',')]
    
    # Create a RealNVP flow
    flow = RealNVP(
        dim=dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        mask_strategy="alternating"
    )
    
    # Wrap in a NormalizingFlow object
    normflow = NormalizingFlow(
        flow=flow,
        base_log_prob_fn=standard_normal_log_prob,
        base_sample_fn=standard_normal_sample,
        name="RealNVP"
    )
    
    return normflow


def fit_and_train_flow(data, obs_space, args):
    """
    Fit a zfit model to data and train a normalizing flow.
    """
    # Create zfit model
    print(f"Creating zfit model with {args.num_params} free parameters...")
    model = create_zfit_model(obs_space, args.num_params)
    
    # Create zfit data
    zfit_data = zfit.Data.from_numpy(obs=obs_space, array=data)
    
    # Create minimizer
    minimizer = zfit.minimize.Minuit()
    
    # Create interface between zfit and flow
    print("Fitting zfit model...")
    start_time = time.time()
    interface = ZfitFlowInterface(model=model, data=zfit_data, minimizer=minimizer)
    fit_result = interface.fit_model()
    fit_time = time.time() - start_time
    
    print(f"Fit completed in {fit_time:.2f} seconds")
    print(f"Fit result: {fit_result}")
    
    # Create flow
    print(f"Creating flow with {args.num_flow_layers} layers...")
    flow = create_flow(args.dim, args.num_flow_layers, args.hidden_dims)
    
    # Create flow loss function
    flow_loss_fn = interface.create_flow_loss(flow)
    
    # Create training
    trainer = ZfitLossTrainer(
        flow=flow,
        zfit_loss_fn=flow_loss_fn,
        batch_size=args.batch_size,
        rng_seed=args.seed
    )
    
    # Initialize training
    data_np = interface.get_data_numpy()
    key = jax.random.PRNGKey(args.seed)
    flow_params = flow.init_params(key)
    trainer.initialize(data=data_np, params=flow_params)
    
    # Train flow
    print(f"Training flow for {args.num_train_steps} steps...")
    start_time = time.time()
    train_results = trainer.train(args.num_train_steps, verbose=True)
    train_time = time.time() - start_time
    
    print(f"Flow training completed in {train_time:.2f} seconds")
    print(f"Final loss: {train_results['loss_history'][-1]:.6f}")
    
    return model, flow, trainer.params, interface


def evaluate_models(model, flow, flow_params, data, obs_space):
    """
    Evaluate the zfit model and normalizing flow.
    """
    print("Evaluating models...")
    
    # Create CovarianceComparer
    cov_comparer = CovarianceComparer(
        flow=flow,
        flow_params=flow_params,
        data=data
    )
    
    # Run all covariance comparisons
    cov_results = cov_comparer.run_all_comparisons()
    
    print("Covariance comparison results:")
    for metric, values in cov_results.items():
        print(f"  {metric}:")
        for key, value in values.items():
            print(f"    {key}: {value:.6f}")
    
    # Create ZfitComparer
    zfit_comparer = ZfitComparer(
        flow=flow,
        flow_params=flow_params,
        zfit_model=model,
        data=data
    )
    
    # Run all zfit comparisons
    zfit_results = zfit_comparer.run_all_comparisons()
    
    print("zfit comparison results:")
    for metric, values in zfit_results.items():
        print(f"  {metric}:")
        for key, value in values.items():
            print(f"    {key}: {value:.6f}")
    
    return cov_results, zfit_results


def save_flow(flow, flow_params, save_path):
    """
    Save the trained flow model.
    """
    print(f"Saving flow to {save_path}...")
    
    # Create serializer
    serializer = FlowSerializer(
        compression_level=9,
        use_float16=True
    )
    
    # Save flow
    serializer.save_flow(
        flow=flow,
        params=flow_params,
        file_path=save_path,
        include_metadata=True,
        overwrite=True
    )
    
    print(f"Flow saved to {save_path}")


def plot_results(data, flow, flow_params, cov_comparer, save_prefix):
    """
    Plot the results of the fitting and training.
    """
    if data.shape[1] > 2:
        print("Plotting only supported for 1D and 2D data")
        return
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    if data.shape[1] == 1:
        # 1D plot
        plt.subplot(2, 2, 1)
        plt.hist(data, bins=50, density=True, alpha=0.5, label='Data')
        
        # Sample from flow
        key = jax.random.PRNGKey(0)
        flow_samples = flow.sample(flow_params, 10000, key)
        plt.hist(flow_samples, bins=50, density=True, alpha=0.5, label='Flow')
        
        # Sample from Gaussian approximation
        gauss_samples = cov_comparer.sample(10000)
        plt.hist(gauss_samples, bins=50, density=True, alpha=0.5, label='Gaussian')
        
        plt.title('Distribution Comparison')
        plt.legend()
        
        # Plot loss history
        plt.subplot(2, 2, 2)
        plt.plot(trainer.loss_history)
        plt.title('Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('NLL')
        
    else:
        # 2D scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5, label='Data')
        plt.title('Data')
        
        # Sample from flow
        key = jax.random.PRNGKey(0)
        flow_samples = flow.sample(flow_params, 10000, key)
        plt.subplot(2, 2, 2)
        plt.scatter(flow_samples[:, 0], flow_samples[:, 1], s=1, alpha=0.5, label='Flow')
        plt.title('Flow Samples')
        
        # Sample from Gaussian approximation
        gauss_samples = cov_comparer.sample(10000)
        plt.subplot(2, 2, 3)
        plt.scatter(gauss_samples[:, 0], gauss_samples[:, 1], s=1, alpha=0.5, label='Gaussian')
        plt.title('Gaussian Samples')
        
        # Plot loss history
        plt.subplot(2, 2, 4)
        plt.plot(trainer.loss_history)
        plt.title('Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('NLL')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_results.png")
    plt.close()
    
    print(f"Plots saved to {save_prefix}_results.png")


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Generate toy data
    print(f"Generating {args.num_samples} toy samples with {args.dim} dimensions...")
    data = generate_toy_data(args.num_samples, args.dim, args.num_params, args.seed)
    
    # Create observable space
    obs_names = [f"x{i}" for i in range(args.dim)]
    obs_space = zfit.Space(obs=obs_names[0], limits=(-10, 10))
    for name in obs_names[1:]:
        obs_space *= zfit.Space(obs=name, limits=(-10, 10))
    
    # Fit model and train flow
    model, flow, flow_params, interface = fit_and_train_flow(data, obs_space, args)
    
    # Evaluate models
    cov_results, zfit_results = evaluate_models(model, flow, flow_params, data, obs_space)
    
    # Save flow
    save_flow(flow, flow_params, args.save_path)
    
    # Create CovarianceComparer for plotting
    cov_comparer = CovarianceBasedSampler(flow, flow_params)
    
    # Plot results
    save_prefix = args.save_path.rsplit('.', 1)[0]
    plot_results(data, flow, flow_params, cov_comparer, save_prefix)
    
    print("Example completed successfully!")
