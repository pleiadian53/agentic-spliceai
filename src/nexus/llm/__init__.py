"""LLM client infrastructure for nexus.

Wraps aisuite + raw openai to support both Chat Completions and
Responses APIs, with a tool-use loop for agents that need tool calls.
Splice-agnostic, splittable away from agents/ if nexus needs it elsewhere.
"""
