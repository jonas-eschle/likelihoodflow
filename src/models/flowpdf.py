"""
Integration of normalizing flows with zfit.
"""

from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import zfit
from zfit import z

from normflow_zfit.flows.base import Flow, NormalizingFlow


class FlowPDF(zfit.pdf.BasePDF):
    """
    A zfit PDF created from a JAX normalizing flow.

    This class wraps a normalizing flow implemented in JAX to be used
    as a zfit PDF for unbinned likelihood fitting.
    """

    def __init__(self,
                 flow: NormalizingFlow,
                 flow_params: Dict,
                 obs: zfit.Space,
                 name: Optional[str] = None):
        """
        Initialize a FlowPDF.

        Args:
            flow: A normalizing flow object
            flow_params: Parameters for the flow
            obs: The observable space
            name: Name of the PDF
        """
        super().__init__(obs=obs, name=name or f"FlowPDF_{flow.name}")

        self.flow = flow
        self._flow_params = flow_params

        # Store the dimensionality
        self.dim = flow.dim

        # Make sure that the dimensions match
        if self.dim != obs.n_obs:
            raise ValueError(f"Flow dimensions ({self.dim}) do not match "
                             f"the observable space dimensions ({obs.n_obs})")

        # Convert log_prob to a jitted function for efficiency
        self._jitted_log_prob = jax.jit(flow.log_prob)

        # Convert sample to a jitted function for efficiency
        self._jitted_sample = jax.jit(flow.sample, static_argnums=1)

    def _unnormalized_pdf(self, x):
        """
        Evaluate the unnormalized PDF.

        Args:
            x: Input data points

        Returns:
            Unnormalized PDF values
        """
        # Convert TensorFlow tensor to NumPy array
        x_np = np.asarray(x)

        # Use JAX to evaluate the log probability
        log_probs = jnp.asarray(self._jitted_log_prob(self._flow_params, x_np))

        # Convert to probability density
        probs = jnp.exp(log_probs)

        # Convert back to TensorFlow
        return tf.convert_to_tensor(np.asarray(probs), dtype=tf.float64)

    def _sample(self, n_samples, rng=None):
        """
        Sample from the PDF.

        Args:
            n_samples: Number of samples to generate
            rng: Random number generator

        Returns:
            Samples from the PDF
        """
        # Create a JAX random key
        key = jax.random.PRNGKey(np.random.randint(0, 2**32))

        # Sample from the flow
        samples = self._jitted_sample(self._flow_params, n_samples, key)

        # Convert to TensorFlow
        return tf.convert_to_tensor(np.asarray(samples), dtype=tf.float64)


class FlowParameter(zfit.Parameter):
    """
    A zfit Parameter that is mapped to normalizing flow parameters.

    This class extends zfit's Parameter to represent a parameter of a
    normalizing flow that can be fitted using zfit's optimization tools.
    """

    def __init__(self,
                 name: str,
                 flow_pdf: FlowPDF,
                 param_path: List[str],
                 initial_value: float,
                 lower_limit: Optional[float] = None,
                 upper_limit: Optional[float] = None,
                 step_size: Optional[float] = None,
                 **kwargs):
        """
        Initialize a FlowParameter.

        Args:
            name: Name of the parameter
            flow_pdf: The FlowPDF this parameter belongs to
            param_path: Path to the parameter in the flow's parameter dictionary
            initial_value: Initial value of the parameter
            lower_limit: Lower limit for the parameter
            upper_limit: Upper limit for the parameter
            step_size: Step size for optimization
            **kwargs: Additional arguments to pass to zfit.Parameter
        """
        super().__init__(
            name=name,
            value=initial_value,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            step_size=step_size,
            **kwargs
        )

        self.flow_pdf = flow_pdf
        self.param_path = param_path

        # Register callback for parameter value changes
        self.register_hook(self._update_flow_param, "value_change")

    def _update_flow_param(self, value, *args, **kwargs):
        """
        Update the corresponding flow parameter when this parameter changes.

        Args:
            value: New parameter value
            *args, **kwargs: Additional arguments from the hook

        Returns:
            Modified return value (typically unchanged)
        """
        # Convert to NumPy
        value_np = np.asarray(value)

        # Navigate through the nested dictionary to find the parameter
        target = self.flow_pdf._flow_params
        for key in self.param_path[:-1]:
            target = target[key]

        # Update the parameter value
        target[self.param_path[-1]] = value_np

        return value


class ParameterizedFlowPDF(FlowPDF):
    """
    A FlowPDF with zfit Parameters for flow parameters.

    This class extends FlowPDF to create zfit Parameters for some or all
    of the flow parameters, allowing them to be optimized during fitting.
    """

    def __init__(self,
                 flow: NormalizingFlow,
                 flow_params: Dict,
                 obs: zfit.Space,
                 param_config: Optional[Dict] = None,
                 name: Optional[str] = None):
        """
        Initialize a ParameterizedFlowPDF.

        Args:
            flow: A normalizing flow object
            flow_params: Parameters for the flow
            obs: The observable space
            param_config: Configuration for which flow parameters to expose as zfit Parameters
                If None, no parameters are exposed
                Dict format:
                    {
                        "param_name": {
                            "path": ["layer_0", "w"],  # Path in flow_params dict
                            "lower_limit": -10.0,      # Optional
                            "upper_limit": 10.0,       # Optional
                            "step_size": 0.1           # Optional
                        },
                        ...
                    }
            name: Name of the PDF
        """
        super().__init__(flow=flow, flow_params=flow_params, obs=obs, name=name)

        # Create zfit Parameters for specified flow parameters
        self.flow_parameters = {}

        if param_config:
            for param_name, config in param_config.items():
                path = config["path"]

                # Navigate through the nested dictionary to find the parameter
                target = self._flow_params
                for key in path[:-1]:
                    target = target[key]

                # Get current value
                current_value = target[path[-1]]

                # Check if the parameter is a scalar or array
                if np.isscalar(current_value) or current_value.size == 1:
                    # Create a scalar parameter
                    param = FlowParameter(
                        name=param_name,
                        flow_pdf=self,
                        param_path=path,
                        initial_value=float(current_value),
                        lower_limit=config.get("lower_limit", None),
                        upper_limit=config.get("upper_limit", None),
                        step_size=config.get("step_size", None)
                    )
                    self.flow_parameters[param_name] = param
                else:
                    # For array parameters, could create multiple parameters
                    # or create a custom composite parameter
                    # This is a simplified version that doesn't handle arrays
                    print(f"Warning: Array parameter {param_name} not supported yet.")

    @property
    def params(self):
        """Return all parameters of the PDF."""
        return self.flow_parameters


class HybridFlowPDF(zfit.pdf.SumPDF):
    """
    A sum of a FlowPDF and an analytical PDF.

    This class combines a normalizing flow with a traditional analytical
    PDF (e.g., Gaussian) to create a hybrid model that can capture both
    the main structure and specific features of the data.
    """

    def __init__(self,
                 flow_pdf: FlowPDF,
                 analytical_pdf: zfit.pdf.BasePDF,
                 flow_fraction: Union[float, zfit.Parameter],
                 obs: zfit.Space,
                 name: Optional[str] = None):
        """
        Initialize a HybridFlowPDF.

        Args:
            flow_pdf: A FlowPDF
            analytical_pdf: An analytical PDF
            flow_fraction: Fraction of the flow PDF in the mixture (0 to 1)
                Either a float or a zfit.Parameter
            obs: The observable space
            name: Name of the PDF
        """
        # Ensure fractions are between 0 and 1
        if isinstance(flow_fraction, float):
            if not 0 <= flow_fraction <= 1:
                raise ValueError("flow_fraction must be between 0 and 1")

            # Create a zfit Parameter for the fraction
            self.flow_frac = zfit.Parameter(
                f"frac_{flow_pdf.name}",
                flow_fraction,
                lower_limit=0.0,
                upper_limit=1.0
            )
        else:
            # Use the provided Parameter
            self.flow_frac = flow_fraction

        # Create a Parameter for the analytical fraction
        self.analytical_frac = zfit.ComposedParameter(
            f"frac_{analytical_pdf.name}",
            lambda f: 1.0 - f,
            params=[self.flow_frac]
        )

        # Create the SumPDF
        super().__init__(
            pdfs=[flow_pdf, analytical_pdf],
            fracs=[self.flow_frac, self.analytical_frac],
            obs=obs,
            name=name or f"HybridFlow_{flow_pdf.name}_{analytical_pdf.name}"
        )

        # Store component PDFs for easy access
        self.flow_pdf = flow_pdf
        self.analytical_pdf = analytical_pdf