# NormFlow

A Python package for training, storing, and evaluating normalizing flow models to approximate probability distributions from zfit fits.

## Overview

NormFlow provides tools for:

- Training normalizing flow models using JAX for efficient GPU acceleration
- Integration with zfit fits for high-energy physics analyses
- Reducing model complexity for efficient storage and inference
- Comparing flow models with covariance matrix approximations
- Creating visualizations of flow transformations and distributions

## Installation

```bash
# Install from PyPI
pip install normflow

# Install with extra dependencies
pip install normflow[dev,compression]

# Install from source
git clone https://github.com/yourusername/normflow.git
cd normflow
pip install -e .
```

## Dependencies

- JAX and JAXlib for GPU-accelerated computation
- Haiku for neural network layers
- Optax for optimization
- zfit for fitting and integration with HEP analyses
- Matplotlib for visualization

## Quick Start

### 1. Fitting a model with zfit and training a flow

```python
import numpy as np
import zfit
from zfit import z
import jax
import jax.numpy as jnp
from normflow.flows.coupling import RealNVP
from normflow.flows.base import NormalizingFlow, standard_normal_log_prob, standard_normal_sample
from normflow.fitting.zfit_interface import ZfitFlowInterface
from normflow.training.trainer import ZfitLossTrainer

# Create zfit model and fit it to data
obs = zfit.Space('x', -10, 10)
mu = zfit.Parameter("mu", 1.2)
sigma = zfit.Parameter("sigma", 1.3)
model = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)

# Generate data
data = np.random.normal(0, 1, 10000)
zfit_data = zfit.Data.from_numpy(obs=obs, array=data.reshape(-1, 1))

# Fit with zfit
nll = zfit.loss.UnbinnedNLL(model=model, data=zfit_data)
minimizer = zfit.minimize.Minuit()
result = minimizer.minimize(nll)
print(result)

# Create flow and interface with zfit
flow = RealNVP(dim=1, hidden_dims=[64, 64], num_layers=4)
normflow = NormalizingFlow(
    flow=flow,
    base_log_prob_fn=standard_normal_log_prob,
    base_sample_fn=standard_normal_sample
)

# Create interface
interface = ZfitFlowInterface(model=model, data=zfit_data)
interface.fit_model()

# Train flow with zfit NLL
flow_loss = interface.create_flow_loss(normflow)
trainer = ZfitLossTrainer(flow=normflow, zfit_loss_fn=flow_loss)
trainer.initialize(data=data.reshape(-1, 1))
result = trainer.train(num_steps=2000)

# Sample from the flow
key = jax.random.PRNGKey(0)
samples = normflow.sample(trainer.params, 10000, key)
```

### 2. Optimizing a flow for reduced complexity

```python
from normflow.storage.compression import FlowCompressor

# Create compressor
compressor = FlowCompressor(
    flow=normflow,
    original_params=trainer.params,
    test_data=jnp.array(data.reshape(-1, 1))
)

# Quantize parameters
quantized_params, metrics = compressor.quantize(bits=8, method='minmax')
print(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
print(f"NLL change: {metrics['nll_change']:.6f}")

# Prune parameters
pruned_params, metrics = compressor.prune(sparsity=0.9, method='magnitude')
print(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
print(f"NLL change: {metrics['nll_change']:.6f}")

# Save the flow
from normflow.storage.serialization import FlowSerializer
serializer = FlowSerializer()
serializer.save_flow(
    flow=normflow,
    params=quantized_params,
    file_path="optimized_flow.json"
)
```

### 3. Comparing with covariance matrix

```python
from normflow.evaluation.comparison import CovarianceComparer

# Create comparer
comparer = CovarianceComparer(
    flow=normflow,
    flow_params=trainer.params,
    data=data.reshape(-1, 1)
)

# Run comparisons
results = comparer.run_all_comparisons()
print(results)
```

### 4. Visualization

```python
from normflow.evaluation.plotting import FlowPlotter

# Create plotter
plotter = FlowPlotter(normflow, trainer.params)

# Plot samples
plotter.plot_samples(
    n_samples=10000, 
    data=data.reshape(-1, 1),
    title="Flow vs Data",
    save_path="flow_samples.png"
)

# For 2D flows, plot density
if normflow.dim == 2:
    plotter.plot_density(
        grid_size=100,
        bounds=[(-5, 5), (-5, 5)],
        data=data,
        title="Flow Density",
        save_path="flow_density.png"
    )
```

## Examples

The package includes several example scripts:

- `examples/unbinned_fit.py`: Fit a zfit model to data and train a flow
- `examples/optimization.py`: Optimize a flow for reduced complexity
- `examples/multi_modal.py`: Fit multimodal distributions

Run an example:

```bash
python examples/unbinned_fit.py --num-samples 10000 --dim 2 --num-params 5
```

## Flow Architectures

Currently supported architectures:

1. **RealNVP**: Coupling-based normalizing flows
2. **Neural Spline Flows**: Spline-based flows for more flexible transformations

## License

MIT License - See LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```
@software{normflow,
  author = {Your Name},
  title = {NormFlow: Normalizing Flows for High-Energy Physics},
  year = {2023},
  url = {https://github.com/yourusername/normflow}
}
```
