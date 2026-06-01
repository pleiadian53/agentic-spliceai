"""Splice-domain agentic layer.

Agents that consume base-layer + meta-layer outputs and apply
splice-specific knowledge (analysis templates, MANE/transcript context,
domain workflows). Depends one-way on the system-wide
`agentic_spliceai.agentic_layer` for LLM clients, planners, and executors.
"""
