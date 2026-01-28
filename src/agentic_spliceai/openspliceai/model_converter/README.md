# Model Converter

Model conversion and interoperability tools for OpenSpliceAI models.

## Features

- Conversion between different deep learning frameworks
- ONNX export for cross-platform inference
- TensorFlow and PyTorch interoperability
- Model compression and optimization
- Genomic workflow integration tools

## Usage

```python
from openspliceai.model_converter import to_onnx, to_tensorflow

# Convert PyTorch model to ONNX format
model_path = "models/splice_model.pt"
to_onnx(model_path, "models/splice_model.onnx")

# Convert PyTorch model to TensorFlow format
tf_model = to_tensorflow(model_path)
tf_model.save("models/splice_model_tf")
```

## Components

- `to_onnx.py`: Conversion utilities for ONNX format
- `to_tensorflow.py`: Conversion utilities for TensorFlow
- `to_pytorch.py`: Conversion utilities for PyTorch
- `optimize.py`: Model optimization and compression tools
- `genomics_io.py`: Genomic-specific I/O adapters
