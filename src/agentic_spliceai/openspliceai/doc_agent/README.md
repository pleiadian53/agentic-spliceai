# Documentation Agent

This package provides tools for automatically generating and maintaining documentation, as well as creating optimized prompts for model interaction with OpenSpliceAI.

## Features

- Automated documentation generation from code
- Documentation validation and update tools
- Prompt engineering utilities for genomic sequence analysis
- Integration with standard documentation formats and tools

## Usage

```python
from openspliceai.doc_agent import generate_module_docs

# Generate documentation for a module
generate_module_docs('openspliceai.create_data')
```

## Components

- `doc_generator.py`: Tools for generating documentation from code
- `doc_validator.py`: Validation tools for existing documentation
- `prompt_templates.py`: Template system for common genomic analysis prompts
- `api_reference.py`: API reference documentation utilities
