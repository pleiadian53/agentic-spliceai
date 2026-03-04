#!/usr/bin/env python3
"""
Example: Check Hardware Feasibility for Foundation Model Workflows

Detects current hardware (or simulates a target) and prints a feasibility
table showing which tasks can run: embedding extraction, classifier training,
LoRA fine-tuning.

Usage:
    # Auto-detect current hardware
    python examples/foundation_models/01_resource_check.py

    # Simulate a specific hardware profile
    python examples/foundation_models/01_resource_check.py --hardware a40-48gb
    python examples/foundation_models/01_resource_check.py --hardware a100-80gb

    # List available hardware profiles
    python examples/foundation_models/01_resource_check.py --list-hardware
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check hardware feasibility for foundation model workflows.",
    )
    parser.add_argument(
        "--hardware", type=str, default=None,
        help="Hardware profile to simulate (e.g., a40-48gb, a100-80gb). "
             "If omitted, auto-detects current hardware.",
    )
    parser.add_argument(
        "--list-hardware", action="store_true",
        help="List all available hardware profiles and exit.",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        choices=["embedding", "training", "lora"],
        help="Show detailed estimate for a specific task.",
    )
    parser.add_argument(
        "--model-size", type=str, default="7b",
        choices=["7b", "40b"],
        help="Evo2 model size for estimation (default: 7b).",
    )
    parser.add_argument(
        "--n-genes", type=int, default=100,
        help="Number of genes for embedding estimation (default: 100).",
    )
    parser.add_argument(
        "--n-windows", type=int, default=500,
        help="Number of training windows for classifier estimation (default: 500).",
    )

    args = parser.parse_args()

    from foundation_models.utils.resources import (
        HARDWARE_PROFILES,
        estimate_classifier_training,
        estimate_embedding_extraction,
        estimate_lora_finetuning,
        print_feasibility_report,
    )

    # List hardware profiles
    if args.list_hardware:
        print("Available hardware profiles:")
        print()
        for name, spec in HARDWARE_PROFILES.items():
            print(f"  {name:<16} {spec['label']:<30} ({spec['device']})")
        print()
        print("Usage: python examples/foundation_models/01_resource_check.py --hardware <name>")
        return

    # Detailed single-task estimate
    if args.task:
        if args.task == "embedding":
            result = estimate_embedding_extraction(
                model_size=args.model_size,
                n_genes=args.n_genes,
                hardware=args.hardware,
            )
            print(f"Embedding Extraction (Evo2 {args.model_size}, {args.n_genes} genes)")
        elif args.task == "training":
            result = estimate_classifier_training(
                n_windows=args.n_windows,
                hardware=args.hardware,
            )
            print(f"Classifier Training ({args.n_windows} windows)")
        elif args.task == "lora":
            result = estimate_lora_finetuning(
                model_size=args.model_size,
                hardware=args.hardware,
            )
            print(f"LoRA Fine-Tuning (Evo2 {args.model_size})")

        print(f"  Hardware: {result['hardware']}")
        print(f"  Feasible: {'YES' if result['feasible'] else 'NO'}")
        for note in result["notes"]:
            print(f"  {note}")
        print()
        return

    # Full feasibility report
    print()
    print("=" * 70)
    print("Foundation Model Resource Feasibility Report")
    print("=" * 70)
    print()
    print_feasibility_report(hardware=args.hardware)

    # Suggest next steps
    print("Next steps:")
    print("  If all tasks are feasible:")
    print("    python examples/foundation_models/02_synthetic_training_pipeline.py")
    print()
    print("  For tasks that require remote GPU:")
    print("    sky launch foundation_models/configs/skypilot/<config>.yaml")
    print()


if __name__ == "__main__":
    main()
