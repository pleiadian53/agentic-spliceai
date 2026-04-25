"""Cross-application utilities (experiment tracking, shared types).

These helpers are shared across the ``base_layer``, ``data_preparation``,
and ``multimodal_features`` applications. They are deliberately minimal
and have no side effects on import — each import has an optional
dependency fallback so the core package works without wandb (or any
other optional tool) installed.
"""
