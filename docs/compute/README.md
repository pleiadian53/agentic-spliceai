# Compute Resources

Guides for managing compute resources across local development and remote GPU training.

---

## Documents

| Document | Description |
|---|---|
| [remote_training_options.md](remote_training_options.md) | Modal vs SkyPilot vs RunPods — programmatic remote training comparison |

## Related Resources

| Location | Description |
|---|---|
| [getting_started/RUNPODS_SETUP.md](../getting_started/RUNPODS_SETUP.md) | RunPods first-time setup guide |
| `runpods.example/docs/` | RunPod-specific reference (SSH, rsync, pod env setup) |
| `foundation_models/docs/training/deepspeed_training.md` | DeepSpeed configs for A40/A100 training |

## Development Principle

**Local-first, remote-when-proven.** Always develop and validate the full training workflow locally on small/synthetic data before offloading to remote GPU. Remote compute should only be used when the pipeline is known to work end-to-end. This minimizes wasted GPU cost and debugging-over-SSH friction.
