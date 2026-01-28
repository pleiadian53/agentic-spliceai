# Error Analyzer

Automated false positive/negative error pattern analysis for splice site prediction models.

## Features

- Systematic error categorization for splice site predictions
- Pattern detection in false positive and false negative cases
- Sequence context analysis for prediction errors
- Error visualization and reporting tools
- Recommendation generation for model improvement

## Usage

```python
from openspliceai.error_analyzer import analyze_errors, generate_report

# Analyze errors from a model's predictions
model_path = "models/splice_model.pt"
test_data = "data/test_dataset.h5"
error_analysis = analyze_errors(model_path, test_data)

# Generate a comprehensive report
generate_report(error_analysis, "error_analysis_report.pdf")
```

## Components

- `categorize.py`: Error categorization and classification
- `pattern_detector.py`: Pattern detection in error cases
- `sequence_context.py`: Analysis of sequence context around errors
- `visualization.py`: Error visualization tools
- `report.py`: Report generation utilities
