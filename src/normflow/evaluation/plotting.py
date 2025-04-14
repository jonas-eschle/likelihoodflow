"""
Plotting utilities for normalizing flows.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import jax
import jax.numpy as jnp

from normflow.flows.base import NormalizingFlow


class FlowPlotter:
    """
    Plotting utilities for normalizing flows.
    
    This class provides methods for visualizing normalizing flows,
    their distributions, and comparison with other models.
    """
    
    def __init__(self, flow: NormalizingFlow, flow_params: dict):
        """
        Initialize a FlowPlotter.
        
        Args:
            flow: The normalizing flow to visualize
            flow_params: Flow parameters
        """
        self.flow = flow
        self.flow_params = flow_params
        self.dim = flow.dim
        
    def plot_samples(self, 
                    n_samples: int = 1000, 
                    data: np.ndarray = None,
                    compare_samples: dict = None,
                    title: str = "Flow Samples",
                    save_path: str = None,
                    figsize: tuple = (10, 8)):
        """
        Plot samples from the flow and optionally compare with data and other samples.
        
        Args:
            n_samples: Number of samples to generate
            data: Original data to compare with (optional)
            compare_samples: Dictionary of {name: samples} for comparison (optional)
            title: Plot title
            save_path: Path to save the figure (optional)
            figsize: Figure size
        """
        if self.dim > 3:
            print(f"Cannot plot {self.dim}-dimensional samples directly. Using PCA or first 2 dimensions.")
        
        # Sample from the flow
        key = jax.random.PRNGKey(0)
        flow_samples = self.flow.sample(self.flow_params, n_samples, key)
        flow_samples_np = np.array(flow_samples)
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        if self.dim == 1:
            # 1D histogram
            alpha = 0.7
            bins = 50
            plt.hist(flow_samples_np, bins=bins, alpha=alpha, density=True, label='Flow')
            
            if data is not None:
                plt.hist(data, bins=bins, alpha=alpha, density=True, label='Data')
            
            if compare_samples is not None:
                for name, samples in compare_samples.items():
                    plt.hist(samples, bins=bins, alpha=alpha, density=True, label=name)
            
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(title)
            plt.legend()
            
        elif self.dim == 2:
            # 2D scatter plot
            plt.scatter(flow_samples_np[:, 0], flow_samples_np[:, 1], s=5, alpha=0.5, label='Flow')
            
            if data is not None:
                plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.5, label='Data')
            
            if compare_samples is not None:
                for name, samples in compare_samples.items():
                    plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, label=name)
            
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title(title)
            plt.legend()
            
        else:
            # For higher dimensions, use first two or PCA
            # For simplicity, just use first two dimensions here
            plt.scatter(flow_samples_np[:, 0], flow_samples_np[:, 1], s=5, alpha=0.5, label='Flow (dim 1-2)')
            
            if data is not None:
                plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.5, label='Data (dim 1-2)')
            
            if compare_samples is not None:
                for name, samples in compare_samples.items():
                    plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, label=f'{name} (dim 1-2)')
            
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title(f"{title} (first 2 dimensions)")
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_density(self, 
                    grid_size: int = 100, 
                    bounds: list = [(-4, 4), (-4, 4)],
                    data: np.ndarray = None,
                    title: str = "Flow Density",
                    save_path: str = None,
                    figsize: tuple = (10, 8),
                    cmap: str = 'viridis',
                    log_scale: bool = False):
        """
        Plot the density of a 2D flow.
        
        Args:
            grid_size: Size of the grid for density evaluation
            bounds: Bounds for each dimension [(x_min, x_max), (y_min, y_max)]
            data: Original data to overlay (optional)
            title: Plot title
            save_path: Path to save the figure (optional)
            figsize: Figure size
            cmap: Colormap for the density
            log_scale: Whether to use log scale for the density
        """
        if self.dim != 2:
            print(f"Density plot is only supported for 2D flows, but flow has {self.dim} dimensions.")
            return None
        
        # Create grid
        x = np.linspace(bounds[0][0], bounds[0][1], grid_size)
        y = np.linspace(bounds[1][0], bounds[1][1], grid_size)
        X, Y = np.meshgrid(x, y)
        grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
        
        # Evaluate density on grid
        log_probs = self.flow.log_prob(self.flow_params, jnp.array(grid_points))
        probs = np.exp(np.array(log_probs))
        
        # Reshape to grid
        if log_scale:
            Z = np.array(log_probs).reshape(grid_size, grid_size)
        else:
            Z = probs.reshape(grid_size, grid_size)
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Plot density contour
        if log_scale:
            plt.contourf(X, Y, Z, levels=50, cmap=cmap)
            plt.colorbar(label='Log Probability Density')
        else:
            plt.contourf(X, Y, Z, levels=50, cmap=cmap)
            plt.colorbar(label='Probability Density')
        
        # Overlay data if provided
        if data is not None:
            plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5, color='red', label='Data')
            plt.legend()
        
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_flow_transformation(self, 
                               n_samples: int = 1000, 
                               n_steps: int = 5,
                               title: str = "Flow Transformation",
                               save_path: str = None,
                               figsize: tuple = (15, 8)):
        """
        Visualize the transformation applied by the flow from base to target distribution.
        
        Args:
            n_samples: Number of samples to generate
            n_steps: Number of intermediate steps to visualize
            title: Plot title
            save_path: Path to save the figure (optional)
            figsize: Figure size
        """
        if self.dim != 2:
            print(f"Transformation visualization is only supported for 2D flows, but flow has {self.dim} dimensions.")
            return None
        
        # Sample from base distribution
        key = jax.random.PRNGKey(0)
        base_samples = self.flow.base_sample_fn(key, (n_samples, self.dim))
        
        if not hasattr(self.flow.flow, 'flows'):
            print(f"Flow transformation visualization is only supported for SequentialFlow.")
            return None
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        n_plots = min(n_steps, len(self.flow.flow.flows) + 1)
        
        # Plot base distribution
        plt.subplot(1, n_plots, 1)
        plt.scatter(base_samples[:, 0], base_samples[:, 1], s=2, alpha=0.5)
        plt.title('Base Distribution')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        # Initialize current samples
        current_samples = base_samples
        
        # Apply transformations one by one
        steps = np.linspace(0, len(self.flow.flow.flows) - 1, n_plots - 1).astype(int)
        
        for i, step in enumerate(steps):
            # Apply transformation
            log_det_sum = jnp.zeros(current_samples.shape[0])
            
            for j in range(step + 1):
                flow = self.flow.flow.flows[j]
                flow_params = self.flow_params[f"flow_{j}"]
                current_samples, log_det = flow.forward(flow_params, current_samples)
            
            # Plot transformed samples
            plt.subplot(1, n_plots, i + 2)
            plt.scatter(current_samples[:, 0], current_samples[:, 1], s=2, alpha=0.5)
            plt.title(f'Step {step+1} / {len(self.flow.flow.flows)}')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_comparison(self, 
                       other_models: dict,
                       test_points: np.ndarray,
                       title: str = "Model Comparison",
                       save_path: str = None,
                       figsize: tuple = (12, 10)):
        """
        Compare the likelihood of the flow with other models.
        
        Args:
            other_models: Dictionary of {name: model} where model is a callable taking x and returning log_prob
            test_points: Points to evaluate
            title: Plot title
            save_path: Path to save the figure (optional)
            figsize: Figure size
        """
        # Convert test points to JAX array
        test_points_jax = jnp.array(test_points)
        
        # Evaluate flow log probability
        flow_log_probs = self.flow.log_prob(self.flow_params, test_points_jax)
        flow_log_probs_np = np.array(flow_log_probs)
        
        # Evaluate other models
        other_log_probs = {}
        for name, model in other_models.items():
            if callable(model):
                other_log_probs[name] = model(test_points)
            else:
                print(f"Model {name} is not callable, skipping")
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        if self.dim == 1:
            # 1D scatter plot
            plt.scatter(test_points, flow_log_probs_np, s=10, alpha=0.7, label='Flow')
            
            for name, log_probs in other_log_probs.items():
                plt.scatter(test_points, log_probs, s=10, alpha=0.7, label=name)
            
            plt.xlabel('Value')
            plt.ylabel('Log Probability')
            plt.title(title)
            plt.legend()
            
        elif self.dim == 2:
            # Create a 3D plot with x, y as the point coordinates and z as the log prob
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(test_points[:, 0], test_points[:, 1], flow_log_probs_np, 
                              c=flow_log_probs_np, cmap='viridis', s=10, alpha=0.7, label='Flow')
            plt.colorbar(scatter, ax=ax, label='Log Probability')
            
            # For other models, we could add them to the same plot, but it gets cluttered
            # Instead, we'll just add one model for comparison
            if other_log_probs:
                name = list(other_log_probs.keys())[0]
                log_probs = other_log_probs[name]
                ax.scatter(test_points[:, 0], test_points[:, 1], log_probs, 
                        c=log_probs, cmap='plasma', s=10, alpha=0.7, label=name)
            
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Log Probability')
            ax.set_title(title)
            ax.legend()
            
        else:
            # For higher dimensions, use a 2D scatter plot with color as log prob
            # Use first two dimensions for x, y coordinates
            scatter = plt.scatter(test_points[:, 0], test_points[:, 1], c=flow_log_probs_np, 
                               cmap='viridis', s=10, alpha=0.7)
            plt.colorbar(scatter, label='Flow Log Probability')
            
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title(f"{title} (colored by flow log prob)")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_training_history(self, 
                            loss_history: list,
                            title: str = "Training History",
                            save_path: str = None,
                            figsize: tuple = (10, 6)):
        """
        Plot the training loss history.
        
        Args:
            loss_history: List of loss values
            title: Plot title
            save_path: Path to save the figure (optional)
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)
        
        plt.plot(loss_history)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_corner(self, 
                   samples: np.ndarray,
                   names: list = None,
                   title: str = "Corner Plot",
                   save_path: str = None,
                   figsize: tuple = (12, 12)):
        """
        Create a corner plot showing all pairwise dimensions.
        
        Args:
            samples: Samples to plot
            names: Names for each dimension
            title: Plot title
            save_path: Path to save the figure (optional)
            figsize: Figure size
        """
        if self.dim < 2:
            print("Corner plot requires at least 2 dimensions.")
            return None
        
        # Generate dimension names if not provided
        if names is None:
            names = [f"Dim {i+1}" for i in range(self.dim)]
        elif len(names) != self.dim:
            print(f"Number of names ({len(names)}) doesn't match dimensions ({self.dim}), using default names.")
            names = [f"Dim {i+1}" for i in range(self.dim)]
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(self.dim, self.dim)
        
        # Create all pairwise plots
        for i in range(self.dim):
            for j in range(self.dim):
                ax = plt.subplot(gs[i, j])
                
                if i == j:
                    # Diagonal: histogram
                    ax.hist(samples[:, i], bins=30, density=True)
                    ax.set_title(names[i])
                elif i > j:
                    # Lower triangle: scatter plot
                    ax.scatter(samples[:, j], samples[:, i], s=1, alpha=0.5)
                    ax.set_xlabel(names[j])
                    ax.set_ylabel(names[i])
                else:
                    # Upper triangle: leave empty
                    ax.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")
        
        return fig
