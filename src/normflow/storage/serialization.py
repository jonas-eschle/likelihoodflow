"""
Serialization utilities for normalizing flows.
"""

import os
import json
import pickle
import gzip
import zlib
import base64
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any, BinaryIO

import jax
import jax.numpy as jnp
import numpy as np

from normflow.flows.base import Flow, NormalizingFlow


class FlowSerializer:
    """
    Serializes and deserializes normalizing flows.
    
    This class provides methods for saving and loading normalizing flow
    models, including their architecture and parameters.
    """
    
    def __init__(self, 
                compression_level: int = 9,
                use_float16: bool = False,
                use_chunking: bool = True,
                chunk_size: int = 1024*1024):
        """
        Initialize a FlowSerializer.
        
        Args:
            compression_level: Compression level (0-9, 0=none, 9=max)
            use_float16: Whether to downcast float32 parameters to float16
            use_chunking: Whether to split large arrays into chunks
            chunk_size: Maximum size of each chunk in bytes
        """
        self.compression_level = compression_level
        self.use_float16 = use_float16
        self.use_chunking = use_chunking
        self.chunk_size = chunk_size
    
    def _process_array(self, arr: Union[np.ndarray, jnp.ndarray]) -> Dict:
        """
        Process a numpy/jax array for serialization.
        
        Args:
            arr: Array to process
            
        Returns:
            Dict representation of the array
        """
        # Convert JAX array to numpy if needed
        if isinstance(arr, jnp.ndarray):
            arr = np.array(arr)
        
        # Cast to float16 if requested and applicable
        if self.use_float16 and arr.dtype in (np.float32, np.float64):
            arr = arr.astype(np.float16)
        
        # Special case for small arrays
        if arr.size < 1000:
            return {
                "type": "small_array",
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "data": arr.tobytes().hex()
            }
        
        # For larger arrays, use chunking if enabled
        if self.use_chunking and arr.nbytes > self.chunk_size:
            # Calculate number of chunks
            arr_bytes = arr.tobytes()
            n_chunks = (len(arr_bytes) + self.chunk_size - 1) // self.chunk_size
            
            # Split the array into chunks
            chunks = []
            for i in range(n_chunks):
                start = i * self.chunk_size
                end = min((i + 1) * self.chunk_size, len(arr_bytes))
                chunk = arr_bytes[start:end]
                
                # Compress the chunk
                if self.compression_level > 0:
                    compressed_chunk = zlib.compress(chunk, self.compression_level)
                else:
                    compressed_chunk = chunk
                
                # Encode the chunk
                encoded_chunk = base64.b64encode(compressed_chunk).decode("ascii")
                chunks.append(encoded_chunk)
            
            return {
                "type": "chunked_array",
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "chunks": chunks,
                "compressed": self.compression_level > 0
            }
        else:
            # Compress the full array
            arr_bytes = arr.tobytes()
            if self.compression_level > 0:
                compressed_bytes = zlib.compress(arr_bytes, self.compression_level)
            else:
                compressed_bytes = arr_bytes
            
            encoded_data = base64.b64encode(compressed_bytes).decode("ascii")
            
            return {
                "type": "array",
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "data": encoded_data,
                "compressed": self.compression_level > 0
            }
    
    def _restore_array(self, arr_dict: Dict) -> np.ndarray:
        """
        Restore an array from its serialized representation.
        
        Args:
            arr_dict: Dict representation of the array
            
        Returns:
            Numpy array
        """
        # Parse shape and dtype
        shape = tuple(arr_dict["shape"])
        dtype = np.dtype(arr_dict["dtype"])
        
        if arr_dict["type"] == "small_array":
            # Decode the hex data
            data = bytes.fromhex(arr_dict["data"])
            arr = np.frombuffer(data, dtype=dtype).reshape(shape)
            return arr
        
        elif arr_dict["type"] == "chunked_array":
            # Decode and concatenate chunks
            chunks = []
            for encoded_chunk in arr_dict["chunks"]:
                compressed_chunk = base64.b64decode(encoded_chunk)
                
                # Decompress if needed
                if arr_dict["compressed"]:
                    chunk = zlib.decompress(compressed_chunk)
                else:
                    chunk = compressed_chunk
                
                chunks.append(chunk)
            
            # Concatenate all chunks
            arr_bytes = b"".join(chunks)
            
            # Create the array
            arr = np.frombuffer(arr_bytes, dtype=dtype).reshape(shape)
            return arr
        
        else:  # "array"
            # Decode the data
            compressed_bytes = base64.b64decode(arr_dict["data"])
            
            # Decompress if needed
            if arr_dict["compressed"]:
                arr_bytes = zlib.decompress(compressed_bytes)
            else:
                arr_bytes = compressed_bytes
            
            # Create the array
            arr = np.frombuffer(arr_bytes, dtype=dtype).reshape(shape)
            return arr
    
    def _serialize_params(self, params: Dict) -> Dict:
        """
        Serialize flow parameters.
        
        Args:
            params: Flow parameters
            
        Returns:
            Serialized parameters
        """
        result = {}
        
        for key, value in params.items():
            if isinstance(value, (np.ndarray, jnp.ndarray)):
                result[key] = self._process_array(value)
            elif isinstance(value, dict):
                result[key] = self._serialize_params(value)
            elif isinstance(value, (list, tuple)):
                result[key] = [
                    self._serialize_params(item) if isinstance(item, dict)
                    else self._process_array(item) if isinstance(item, (np.ndarray, jnp.ndarray))
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        
        return result
    
    def _deserialize_params(self, serialized_params: Dict) -> Dict:
        """
        Deserialize flow parameters.
        
        Args:
            serialized_params: Serialized parameters
            
        Returns:
            Deserialized parameters
        """
        result = {}
        
        for key, value in serialized_params.items():
            if isinstance(value, dict) and "type" in value and value["type"] in ("array", "chunked_array", "small_array"):
                result[key] = self._restore_array(value)
            elif isinstance(value, dict):
                result[key] = self._deserialize_params(value)
            elif isinstance(value, list):
                result[key] = [
                    self._deserialize_params(item) if isinstance(item, dict) and "type" not in item
                    else self._restore_array(item) if isinstance(item, dict) and "type" in item
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        
        return result
    
    def serialize_flow(self, 
                      flow: NormalizingFlow, 
                      params: Dict,
                      include_metadata: bool = True) -> Dict:
        """
        Serialize a normalizing flow model.
        
        Args:
            flow: Normalizing flow to serialize
            params: Flow parameters
            include_metadata: Whether to include metadata
            
        Returns:
            Serialized flow as a dictionary
        """
        # Serialize parameters
        serialized_params = self._serialize_params(params)
        
        # Create serialized model
        serialized = {
            "flow_type": flow.__class__.__name__,
            "flow_dim": flow.dim,
            "flow_name": flow.name,
            "params": serialized_params,
        }
        
        # Add metadata if requested
        if include_metadata:
            serialized["metadata"] = {
                "serializer_version": "1.0.0",
                "compression_level": self.compression_level,
                "use_float16": self.use_float16,
                "use_chunking": self.use_chunking,
                "timestamp": jax.process_time()
            }
        
        return serialized
    
    def deserialize_flow(self, 
                        serialized: Dict, 
                        flow_class: type,
                        base_log_prob_fn: Optional[callable] = None,
                        base_sample_fn: Optional[callable] = None) -> Tuple[NormalizingFlow, Dict]:
        """
        Deserialize a normalizing flow model.
        
        Args:
            serialized: Serialized flow dictionary
            flow_class: Class to instantiate for the flow
            base_log_prob_fn: Log probability function for base distribution (optional)
            base_sample_fn: Sampling function for base distribution (optional)
            
        Returns:
            Tuple of (Flow object, parameters)
        """
        # Verify that the serialized flow type matches the provided class
        if serialized["flow_type"] != flow_class.__name__:
            raise ValueError(f"Flow type mismatch. Expected {flow_class.__name__}, "
                            f"got {serialized['flow_type']}")
        
        # Deserialize parameters
        params = self._deserialize_params(serialized["params"])
        
        # Create the flow object
        # Note: In a real implementation, we would need to recreate the
        # specific flow architecture based on the serialized information
        flow = flow_class(
            dim=serialized["flow_dim"],
            name=serialized["flow_name"],
            base_log_prob_fn=base_log_prob_fn,
            base_sample_fn=base_sample_fn
        )
        
        return flow, params
    
    def save_flow(self, 
                 flow: NormalizingFlow, 
                 params: Dict, 
                 file_path: str,
                 include_metadata: bool = True,
                 overwrite: bool = False):
        """
        Save a normalizing flow model to a file.
        
        Args:
            flow: Normalizing flow to save
            params: Flow parameters
            file_path: Path to save the file
            include_metadata: Whether to include metadata
            overwrite: Whether to overwrite existing files
        """
        # Check if file exists
        if os.path.exists(file_path) and not overwrite:
            raise FileExistsError(f"File {file_path} already exists. Use overwrite=True to replace it.")
        
        # Serialize the flow
        serialized = self.serialize_flow(flow, params, include_metadata)
        
        # Save to file
        with open(file_path, "w") as f:
            json.dump(serialized, f)
    
    def load_flow(self, 
                 file_path: str, 
                 flow_class: type,
                 base_log_prob_fn: Optional[callable] = None,
                 base_sample_fn: Optional[callable] = None) -> Tuple[NormalizingFlow, Dict]:
        """
        Load a normalizing flow model from a file.
        
        Args:
            file_path: Path to load the file from
            flow_class: Class to instantiate for the flow
            base_log_prob_fn: Log probability function for base distribution (optional)
            base_sample_fn: Sampling function for base distribution (optional)
            
        Returns:
            Tuple of (Flow object, parameters)
        """
        # Load from file
        with open(file_path, "r") as f:
            serialized = json.load(f)
        
        # Deserialize the flow
        return self.deserialize_flow(serialized, flow_class, base_log_prob_fn, base_sample_fn)


class OptimizedFlowExporter:
    """
    Exports flows in an optimized format for deployment.
    
    This class provides methods for exporting normalizing flows in formats
    that are optimized for deployment, such as TensorFlow SavedModel or ONNX.
    """
    
    def __init__(self, 
                optimize_for_inference: bool = True,
                use_xla: bool = True,
                target_platform: str = "cpu"):
        """
        Initialize an OptimizedFlowExporter.
        
        Args:
            optimize_for_inference: Whether to optimize the model for inference
            use_xla: Whether to use XLA compilation
            target_platform: Target platform ("cpu", "gpu", "tpu")
        """
        self.optimize_for_inference = optimize_for_inference
        self.use_xla = use_xla
        self.target_platform = target_platform
    
    def export_flow_to_saved_model(self,
                                  flow: NormalizingFlow,
                                  params: Dict,
                                  export_dir: str,
                                  overwrite: bool = False):
        """
        Export a flow to TensorFlow SavedModel format.
        
        Args:
            flow: Normalizing flow to export
            params: Flow parameters
            export_dir: Directory to export to
            overwrite: Whether to overwrite existing directory
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required for this feature. "
                            "Install it with 'pip install tensorflow'.")
        
        # Check if directory exists
        if os.path.exists(export_dir) and not overwrite:
            raise FileExistsError(f"Directory {export_dir} already exists. "
                                 "Use overwrite=True to replace it.")
        
        # Create a TensorFlow module for the flow
        class FlowModule(tf.Module):
            def __init__(self, flow, params):
                super().__init__()
                self.flow = flow
                self.params = params
            
            @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
            def log_prob(self, x):
                # Convert TensorFlow tensor to NumPy
                x_np = x.numpy()
                
                # Use JAX to compute log_prob
                log_probs = self.flow.log_prob(self.params, x_np)
                
                # Convert back to TensorFlow
                return tf.constant(np.asarray(log_probs), dtype=tf.float32)
            
            @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.int32)])
            def sample(self, n_samples):
                # Generate a random key
                key = jax.random.PRNGKey(np.random.randint(0, 2**32))
                
                # Sample from the flow
                samples = self.flow.sample(self.params, n_samples.numpy(), key)
                
                # Convert to TensorFlow
                return tf.constant(np.asarray(samples), dtype=tf.float32)
        
        # Create the module
        module = FlowModule(flow, params)
        
        # Export to SavedModel
        tf.saved_model.save(module, export_dir)
        
        print(f"Model exported to {export_dir}")
    
    def export_flow_to_onnx(self,
                           flow: NormalizingFlow,
                           params: Dict,
                           output_path: str,
                           overwrite: bool = False):
        """
        Export a flow to ONNX format.
        
        Args:
            flow: Normalizing flow to export
            params: Flow parameters
            output_path: Path to export to
            overwrite: Whether to overwrite existing file
        """
        try:
            import onnx
            import onnxruntime
        except ImportError:
            raise ImportError("ONNX and ONNX Runtime are required for this feature. "
                             "Install them with 'pip install onnx onnxruntime'.")
        
        # Check if file exists
        if os.path.exists(output_path) and not overwrite:
            raise FileExistsError(f"File {output_path} already exists. "
                                 "Use overwrite=True to replace it.")
        
        # This is a placeholder - actual ONNX export would require more work
        # In a real implementation, we would need to:
        # 1. Create a function to compute log_prob that can be traced
        # 2. Use JAX's jax2tf to convert to TensorFlow
        # 3. Use tf2onnx to convert from TensorFlow to ONNX
        
        print(f"ONNX export is not fully implemented. This is a placeholder.")
