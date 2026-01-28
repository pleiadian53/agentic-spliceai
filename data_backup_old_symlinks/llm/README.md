# LLM Models Directory

This directory stores information about available LLM models from OpenAI and other providers.

## Files

- **`available_models.json`** - Current snapshot of available models with metadata and recommendations

## Usage

### Save Current Models List

```python
from openai import OpenAI
from chart_agent import save_models_list

client = OpenAI()

# Save with full metadata
filepath = save_models_list(
    client.models.list(),
    output_dir="data/llm",
    filename="available_models.json",
    include_metadata=True
)

print(f"Saved to: {filepath}")
```

### Load Saved Models List

```python
from chart_agent import load_models_list

# Load previously saved models
data = load_models_list("data/llm/available_models.json")

print(f"Timestamp: {data['timestamp']}")
print(f"Total models: {data['total_models']}")
print(f"Recommendations: {data['recommendations']}")
```

### Get Recommendations

The saved file includes intelligent recommendations by use case:

```python
from chart_agent import get_recommended_models
from openai import OpenAI

client = OpenAI()
recommendations = get_recommended_models(client.models.list())

# Use cases:
# - chart_generation: Best models for visualization code
# - code_generation: Best models for general code (including codex)
# - fast_prototyping: Quick/cheap models for testing
# - cost_effective: Cheapest models
# - vision: Models with image understanding
```

## Model Detection Strategy

The recommendation system uses **dynamic pattern matching** rather than hardcoded versions.

### Model Tiers (Verified as of Nov 2025)

OpenAI uses the following naming convention:
- **Full models**: `gpt-5.1`, `gpt-5`, `gpt-4o` - Most capable, most expensive
- **Pro models**: `gpt-5-pro` - Premium tier
- **Mini models**: `gpt-4o-mini`, `gpt-5-mini` - Balanced cost/performance
- **Nano models**: `gpt-5-nano`, `gpt-4.1-nano` - Cheapest, fastest
- **Codex models**: `gpt-5-codex`, `gpt-5.1-codex` - Code-specialized

### Detection Patterns

- **Chart generation**: `gpt-5.1`, `gpt-5-pro`, `gpt-5`, `gpt-4o` (excludes mini/nano)
- **Code generation**: Codex models first, then `gpt-5.1`, `gpt-5-pro`, `gpt-4o`
- **Fast prototyping**: `gpt-4o-mini` (default), then other mini variants
- **Cost effective**: Nano models first, then `gpt-4o-mini`, then `gpt-3.5-turbo`
- **Vision**: Models with vision/image capabilities

This makes the system **future-proof** - it will automatically detect:
- `gpt-5.1`, `gpt-5.2`, `gpt-6`, etc.
- `gpt-5-codex`, `gpt-5.1-codex-mini`, etc.
- Any new model families following OpenAI naming conventions

### Default Model: gpt-4o-mini

For development, testing, and fast iteration, **gpt-4o-mini is the recommended default**:
- ✅ Proven performance for chart generation
- ✅ Cost-effective (~10x cheaper than GPT-5)
- ✅ Fast response times
- ✅ Good balance of quality and speed

Use GPT-5 series for production or when quality is critical.

## File Format

```json
{
  "timestamp": "2025-11-18T13:26:39.759101",
  "total_models": 102,
  "models": [
    {
      "id": "gpt-4o",
      "created": 1715367049,
      "owner": "openai",
      "status": "active",
      "description": "GPT-4o model description..."
    },
    ...
  ],
  "recommendations": {
    "chart_generation": ["gpt-5.1", "gpt-4o", "gpt-4-turbo"],
    "code_generation": ["gpt-5-codex", "gpt-5.1", "gpt-4o"],
    "fast_prototyping": ["gpt-4o-mini", "gpt-3.5-turbo"],
    "cost_effective": ["gpt-4o-mini", "gpt-3.5-turbo"],
    "vision": ["gpt-5.1", "gpt-4o", "gpt-4-turbo"]
  }
}
```

## Updating Models List

It's recommended to update the models list periodically:

```bash
# Quick update
mamba run -n agentic-ai python -c "from openai import OpenAI; from chart_agent import save_models_list; from dotenv import load_dotenv; load_dotenv(); save_models_list(OpenAI().models.list())"

# Or run the demo
mamba run -n agentic-ai python -m chart_agent.examples.utils_demo
```

## Integration with Analysis Scripts

The splice site analysis script can use these recommendations:

```bash
# Load recommendations and use best model
mamba run -n agentic-ai python -m chart_agent.examples.analyze_splice_sites \
    --data data/splice_sites_enhanced.tsv \
    --analysis all \
    --model gpt-5.1 \
    --reflect \
    --reflection-model gpt-5.1
```

## Version History

- **2025-11-18**: Initial version with dynamic pattern matching
- Models are updated automatically when `save_models_list()` is called
