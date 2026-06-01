"""System-wide agentic infrastructure for agentic-spliceai.

Reusable, splice-agnostic components: LLM clients, code-as-plan generators,
sandboxed code execution, and generic tabular data adapters used by all
agentic workflows in the project. Splice-domain agents live under
`agentic_spliceai.splice_engine.agentic_layer` and depend on this package
one-way.
"""
