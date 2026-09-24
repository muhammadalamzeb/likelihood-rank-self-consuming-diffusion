# Environment freeze

**Recorded:** 2026-09-23  
**Purpose:** Path 2 intake — reproducible dependency snapshot before an external direction.

## Runtime

| Item | Value |
|------|-------|
| OS | Windows 11 (10.0.22000) |
| Python | 3.14.6 |
| Executable | `D:\Machine Learning Paper\.venv\Scripts\python.exe` |
| Device | CPU (`torch 2.14.0+cpu`) |

## Core packages (imported by probes)

| Package | Version |
|---------|---------|
| torch | 2.14.0+cpu |
| torchvision | 0.29.0+cpu |
| numpy | 2.5.2 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| networkx | 3.6.1 |
| pillow | 12.3.0 |
| pandas | 3.0.6 |
| diffusers | 0.40.0 |
| transformers | 5.17.0 |
| datasets | 5.0.1 |
| huggingface_hub | 1.32.0 |
| accelerate | 1.15.0 |

## Full freeze

Exact `pip freeze` snapshot: [`docs/requirements-freeze.txt`](requirements-freeze.txt)

Recreate (approximate):

```powershell
py -3.14 -m venv "D:\Machine Learning Paper\.venv"
& "D:\Machine Learning Paper\.venv\Scripts\python.exe" -m pip install -r "D:\Machine Learning Paper\docs\requirements-freeze.txt"
```

Note: torch/torchvision are CPU wheels; GPU installs need matching CUDA indexes if an external direction requires GPU.
