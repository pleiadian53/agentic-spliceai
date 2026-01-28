# Benchmark Package

Performance evaluation and comparison tools for OpenSpliceAI models.

## Features

- Standardized test datasets for consistent evaluation
- Comprehensive metrics for splice site prediction assessment
- Tools for comparison against baseline methods and published approaches
- Performance visualization and reporting utilities

## Usage

```python
from openspliceai.benchmark import evaluate_model, compare_models

# Evaluate a single model
results = evaluate_model(model_path, test_dataset)

# Compare multiple models
comparison = compare_models([model1, model2, model3], test_dataset)
comparison.generate_report('comparison_report.pdf')
```

## Components

- `metrics.py`: Implementation of standard and custom evaluation metrics
- `datasets.py`: Standard benchmark datasets for consistent evaluation
- `comparison.py`: Tools for comparing multiple models
- `visualization.py`: Visualization utilities for performance analysis
