# Training Recipes

Preset training configurations and best practices for OpenSpliceAI models.

## Features

- Curated hyperparameter configurations for different scenarios
- Training pipelines optimized for various datasets
- Transfer learning recipes for different genomes
- Performance optimization strategies
- Early stopping and checkpoint management
- Learning rate scheduling recipes

## Usage

```python
from openspliceai.training_recipes import human_genome, mouse_genome

# Load a recipe for human genome training
config = human_genome.get_config(context_length=10000)

# Train a model using the recipe
from openspliceai.train import train_model
train_model(config)

# Or load a recipe for mouse genome
mouse_config = mouse_genome.get_config(donor_weight=2.0)
```

## Components

- `human_genome.py`: Recipes optimized for human genome
- `model_zoo.py`: Pre-configured model architectures
- `transfer_learning.py`: Transfer learning configurations
- `optimization.py`: Performance optimization strategies
- `scheduling.py`: Learning rate scheduling strategies
- `validation.py`: Cross-validation configurations
