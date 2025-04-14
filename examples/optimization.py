"""
Example of optimizing normalizing flows for reduced complexity.

This example demonstrates:
1. Training a normalizing flow for a multimodal distribution
2. Optimizing the flow for reduced complexity
3. Compressing the flow for efficient storage
4. Evaluating the compressed flow
"""

import argparse
import time
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from normflow.flows.base import NormalizingFlow, standard_normal_log_prob, standard_normal_sample
from normflow.flows.coupling import RealNVP
from normflow.flows.spline import NSF
from normflow.training.trainer import FlowTrainer
from normflow.storage.compression import FlowCompressor
from normflow.evaluation.plotting import FlowPlotter
from normflow.storage.serialization import FlowSerializer


def parse_args():
    parser = argparse.ArgumentParser(description='Example of optimizing normalizing flows')
    parser.add_argument('--num-samples', type=int, default=10000,
                        help='Number of samples for training/testing')
    parser.add_argument('--dim', type=int, default=2,
                        help='Dimensionality of the samples')
    parser.add_argument('--flow-type', type=str, default='realnvp',
                        choices=['realnvp', 'nsf'], help='Type of normalizing flow')
    parser.add_argument('--num-train-steps', type=int, default=5000,
                        help='Number of flow training steps')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for flow training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--num-layers', type=int, default=8,
                        help='Number of coupling layers in the flow')
    parser.add_argument('--hidden-dims', type=str, default='64,64',
                        help='Hidden dimensions for coupling networks (comma-separated)')
    parser.add_argument('--quantize-bits', type=int, default=8,
                        help='Number of bits for quantization')
    parser.add_argument('--prune-sparsity', type=float, default=0.9,
                        help='Sparsity level for pruning')
    parser.add_argument('--distill-layers', type=int, default=4,
                        help='Number of layers for distilled model')
    return parser.parse_args()


def generate_mixture_of_gaussians(num_samples, dim, num_components=3, seed=42):
    """
    Generate a mixture of Gaussians for testing.
    
    Args:
        num_samples: Number of samples to generate
        dim: Dimensionality of the samples
        num_components: Number of components in the mixture
        seed: Random seed
        
    Returns:
        Numpy array of samples
    """
    rng = np.random.RandomState(seed)
    
    # Create mixture components
    means = rng.randn(num_components, dim) * 3
    covs = [rng.uniform(0.5, 1.5) * np.eye(dim) for _ in range(num_components)]
    
    # Add correlations to some components
    for i in range(num_components):
        if dim > 1 and i % 2 == 0:
            covs[i][0, 1] = 0.7
            covs[i][1, 0] = 0.7
    
    # Sample from the mixture
    weights = rng.dirichlet(np.ones(num_components))
    component = rng.choice(num_components, size=num_samples, p=weights)
    
    samples = np.zeros((num_samples, dim))
    for i in range(num_samples):
        c = component[i]
        samples[i] = rng.multivariate_normal(means[c], covs[c])
    
    return samples


def create_flow(flow_type, dim, num_layers, hidden_dims):
    """
    Create a flow model.
    
    Args:
        flow_type: Type of flow ('realnvp' or 'nsf')
        dim: Dimensionality of the flow
        num_layers: Number of flow layers
        hidden_dims: Hidden dimensions for coupling networks
        
    Returns:
        NormalizingFlow object
    """
    # Parse hidden dimensions
    hidden_dims = [int(d) for d in hidden_dims.split(',')]
    
    if flow_type == 'realnvp':
        flow = RealNVP(
            dim=dim,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            mask_strategy="alternating"
        )
    elif flow_type == 'nsf':
        flow = NSF(
            dim=dim,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            num_bins=8,
            mask_strategy="alternating"
        )
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")
    
    # Wrap in a NormalizingFlow object
    normflow = NormalizingFlow(
        flow=flow,
        base_log_prob_fn=standard_normal_log_prob,
        base_sample_fn=standard_normal_sample,
        name=flow_type.upper()
    )
    
    return normflow


def train_flow(flow, data, args):
    """
    Train a flow on the given data.
    
    Args:
        flow: NormalizingFlow object
        data: Training data
        args: Command-line arguments
        
    Returns:
        Trained flow parameters and training results
    """
    print(f"Training {flow.name} flow...")
    
    # Create training
    trainer = FlowTrainer(
        flow=flow,
        batch_size=args.batch_size,
        rng_seed=args.seed
    )
    
    # Initialize parameters
    key = jax.random.PRNGKey(args.seed)
    flow_params = flow.init_params(key)
    
    # Initialize training
    trainer.initialize(data=data, params=flow_params)
    
    # Train flow
    start_time = time.time()
    train_results = trainer.train(args.num_train_steps, verbose=True)
    train_time = time.time() - start_time
    
    print(f"Training completed in {train_time:.2f} seconds")
    print(f"Final loss: {train_results['loss_history'][-1]:.6f}")
    
    # Evaluate on test data
    test_nll = -jnp.mean(flow.log_prob(trainer.params, jnp.array(data)))
    print(f"Test NLL: {test_nll:.6f}")
    
    # Add additional metrics to results
    train_results['train_time'] = train_time
    train_results['test_nll'] = float(test_nll)
    
    return trainer.params, train_results


def optimize_flow(flow, flow_params, data, args):
    """
    Optimize a flow for reduced complexity and efficient storage.
    
    Args:
        flow: NormalizingFlow object
        flow_params: Flow parameters
        data: Test data
        args: Command-line arguments
        
    Returns:
        Dictionary of optimized models and results
    """
    print(f"Optimizing flow for reduced complexity...")
    
    # Create flow compressor
    compressor = FlowCompressor(
        flow=flow,
        original_params=flow_params,
        test_data=jnp.array(data)
    )
    
    # Apply different compression techniques
    results = {}
    
    # 1. Quantization
    print("\n=== Quantization ===")
    quantized_params, quant_metrics = compressor.quantize(
        bits=args.quantize_bits,
        method='minmax',
        per_tensor=False
    )
    results['quantized'] = {
        'params': quantized_params,
        'metrics': quant_metrics
    }
    
    # 2. Pruning
    print("\n=== Pruning ===")
    pruned_params, prune_metrics = compressor.prune(
        sparsity=args.prune_sparsity,
        method='magnitude',
        fine_tune_steps=1000
    )
    results['pruned'] = {
        'params': pruned_params,
        'metrics': prune_metrics
    }
    
    # 3. Distillation
    print("\n=== Distillation ===")
    distilled_flow, distilled_params, distill_metrics = compressor.distill(
        hidden_dims=[32, 32],
        num_layers=args.distill_layers,
        num_samples=10000,
        num_steps=3000
    )
    results['distilled'] = {
        'flow': distilled_flow,
        'params': distilled_params,
        'metrics': distill_metrics
    }
    
    # 4. Combined: Quantize + Prune
    print("\n=== Combined Quantize + Prune ===")
    quantized_pruned_params, quant_prune_metrics = compressor.quantize(
        bits=args.quantize_bits,
        method='minmax',
        per_tensor=False,
    )
    quantized_pruned_params, quant_prune_metrics = compressor.prune(
        sparsity=args.prune_sparsity / 2,  # Lower sparsity for combined approach
        method='magnitude',
        fine_tune_steps=1000
    )
    results['quantized_pruned'] = {
        'params': quantized_pruned_params,
        'metrics': quant_prune_metrics
    }
    
    return results


def save_results(flow, flow_params, optimized_results, data, save_dir, args):
    """
    Save and visualize results.
    
    Args:
        flow: NormalizingFlow object
        flow_params: Original flow parameters
        optimized_results: Results from optimization
        data: Test data
        save_dir: Directory to save results
        args: Command-line arguments
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a plotter for visualization
    plotter = FlowPlotter(flow, flow_params)
    
    # Plot original data and samples
    key = jax.random.PRNGKey(args.seed + 1)
    samples = flow.sample(flow_params, 10000, key)
    
    fig = plotter.plot_samples(
        n_samples=10000,
        data=data,
        title=f"Original {flow.name} Samples",
        save_path=os.path.join(save_dir, "original_samples.png")
    )
    
    # Plot density if 2D
    if args.dim == 2:
        fig = plotter.plot_density(
            grid_size=100,
            bounds=[(-6, 6), (-6, 6)],
            data=data,
            title=f"Original {flow.name} Density",
            save_path=os.path.join(save_dir, "original_density.png")
        )
    
    # Visualize flow transformation if 2D and has multiple layers
    if args.dim == 2 and hasattr(flow.flow, 'flows'):
        fig = plotter.plot_flow_transformation(
            n_samples=5000,
            n_steps=min(5, len(flow.flow.flows)),
            title=f"{flow.name} Transformation",
            save_path=os.path.join(save_dir, "flow_transformation.png")
        )
    
    # Compare optimized models
    print("\nComparison of optimized models:")
    print(f"{'Method':<20} {'NLL':<10} {'Size (KB)':<15} {'Compression':<15}")
    print("-" * 60)
    
    original_size = FlowCompressor(flow, flow_params)._get_params_size(flow_params)
    original_nll = -jnp.mean(flow.log_prob(flow_params, jnp.array(data)))
    
    print(f"{'Original':<20} {float(original_nll):<10.6f} {original_size/1024:<15.2f} {1.0:<15.2f}x")
    
    compression_results = {
        'original': {
            'nll': float(original_nll),
            'size': original_size,
            'ratio': 1.0
        }
    }
    
    # Sample and evaluate each optimized model
    for method, results in optimized_results.items():
        if method == 'distilled':
            # Use distilled flow
            distilled_flow = results['flow']
            distilled_params = results['params']
            metrics = results['metrics']
            
            # Sample from distilled flow
            key = jax.random.PRNGKey(args.seed + 2)
            distilled_samples = distilled_flow.sample(distilled_params, 10000, key)
            
            # Plot samples
            distilled_plotter = FlowPlotter(distilled_flow, distilled_params)
            fig = distilled_plotter.plot_samples(
                n_samples=10000,
                data=data,
                title=f"Distilled {distilled_flow.name} Samples",
                save_path=os.path.join(save_dir, f"{method}_samples.png")
            )
            
            # Plot density if 2D
            if args.dim == 2:
                fig = distilled_plotter.plot_density(
                    grid_size=100,
                    bounds=[(-6, 6), (-6, 6)],
                    data=data,
                    title=f"Distilled {distilled_flow.name} Density",
                    save_path=os.path.join(save_dir, f"{method}_density.png")
                )
            
            # Get metrics
            size = metrics['compressed_size']
            nll = metrics['compressed_nll']
            ratio = metrics['compression_ratio']
        else:
            # Use original flow with optimized parameters
            params = results['params']
            metrics = results['metrics']
            
            # Sample from optimized flow
            key = jax.random.PRNGKey(args.seed + 3)
            opt_samples = flow.sample(params, 10000, key)
            
            # Create plotter for optimized flow
            opt_plotter = FlowPlotter(flow, params)
            
            # Plot samples
            fig = opt_plotter.plot_samples(
                n_samples=10000,
                data=data,
                title=f"{method.capitalize()} {flow.name} Samples",
                save_path=os.path.join(save_dir, f"{method}_samples.png")
            )
            
            # Plot density if 2D
            if args.dim == 2:
                fig = opt_plotter.plot_density(
                    grid_size=100,
                    bounds=[(-6, 6), (-6, 6)],
                    data=data,
                    title=f"{method.capitalize()} {flow.name} Density",
                    save_path=os.path.join(save_dir, f"{method}_density.png")
                )
            
            # Get metrics
            size = metrics['compressed_size']
            nll = metrics['compressed_nll']
            ratio = metrics['compression_ratio']
        
        print(f"{method.capitalize():<20} {float(nll):<10.6f} {size/1024:<15.2f} {ratio:<15.2f}x")
        
        compression_results[method] = {
            'nll': float(nll),
            'size': size,
            'ratio': ratio
        }
    
    # Plot comparison of NLL vs Size
    methods = list(compression_results.keys())
    nlls = [compression_results[m]['nll'] for m in methods]
    sizes = [compression_results[m]['size'] / 1024 for m in methods]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(sizes, nlls, s=100)
    
    for i, method in enumerate(methods):
        plt.annotate(method.capitalize(), (sizes[i], nlls[i]), 
                   xytext=(10, 10), textcoords='offset points')
    
    plt.xlabel('Model Size (KB)')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Compression Comparison: Model Size vs. NLL')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compression_comparison.png"))
    
    # Save the original and optimized models
    serializer = FlowSerializer(compression_level=9)
    
    # Save original model
    serializer.save_flow(
        flow=flow,
        params=flow_params,
        file_path=os.path.join(save_dir, "original_model.json"),
        overwrite=True
    )
    
    # Save optimized models
    for method, results in optimized_results.items():
        if method == 'distilled':
            # Save distilled model
            serializer.save_flow(
                flow=results['flow'],
                params=results['params'],
                file_path=os.path.join(save_dir, f"{method}_model.json"),
                overwrite=True
            )
        else:
            # Save original flow with optimized parameters
            serializer.save_flow(
                flow=flow,
                params=results['params'],
                file_path=os.path.join(save_dir, f"{method}_model.json"),
                overwrite=True
            )
    
    # Save all results as pickle for later analysis
    results_dict = {
        'args': vars(args),
        'compression_results': compression_results,
        'optimized_results': {
            # Don't save the full flow objects, just metrics
            k: {kk: v for kk, v in v.items() if kk != 'flow' and kk != 'params'}
            for k, v in optimized_results.items()
        }
    }
    
    with open(os.path.join(save_dir, "results.pkl"), 'wb') as f:
        pickle.dump(results_dict, f)


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Generate toy data
    print(f"Generating {args.num_samples} mixture of Gaussians samples with {args.dim} dimensions...")
    data = generate_mixture_of_gaussians(args.num_samples, args.dim, num_components=3, seed=args.seed)
    
    # Split into train/test
    train_data = data[:int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]
    
    # Create flow
    flow = create_flow(args.flow_type, args.dim, args.num_layers, args.hidden_dims)
    
    # Train flow
    flow_params, train_results = train_flow(flow, train_data, args)
    
    # Optimize flow for reduced complexity
    optimized_results = optimize_flow(flow, flow_params, test_data, args)
    
    # Save and visualize results
    save_results(flow, flow_params, optimized_results, test_data, args.save_dir, args)
    
    print(f"\nResults saved to {args.save_dir}")
    print("Example completed successfully!")
