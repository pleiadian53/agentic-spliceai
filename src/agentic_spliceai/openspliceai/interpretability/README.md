# Interpretability

Sequence-level model explanation tools for OpenSpliceAI predictions.

## Features

- Attribution methods for identifying important nucleotides
- Saliency maps for visualizing model focus
- Integrated Gradients implementation for deep networks
- Class activation mapping for convolutional architectures
- Counterfactual analysis for splice site models
- Motif extraction from model activations

## Usage

```python
from openspliceai.interpretability import generate_saliency_map, integrated_gradients

# Generate a saliency map for a sequence
sequence = "ACGTACGTACGT"
model_path = "models/splice_model.pt"
saliency_map = generate_saliency_map(model_path, sequence)

# Perform integrated gradients analysis
attribution = integrated_gradients(model_path, sequence, target_class=1)
```

## Components

- `saliency.py`: Saliency map generation
- `integrated_gradients.py`: Integrated Gradients implementation
- `cam.py`: Class activation mapping tools
- `counterfactual.py`: Counterfactual analysis
- `motif_extraction.py`: Motif extraction from model activations
- `visualization.py`: Visualization tools for attributions
