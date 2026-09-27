# Environment freeze

**Purpose:** Reproducible dependency snapshot for the experiments in this repository.

## Runtime (reference machine)

| Item | Value |
|------|-------|
| OS | Windows 11 |
| Python | 3.14.x |
| Device | CPU (`torch` CPU wheel) |

Create a local virtualenv in the repo root (path may differ on your machine):

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\python.exe -m pip install -r docs/requirements-freeze.txt
# Unix:    .venv/bin/python -m pip install -r docs/requirements-freeze.txt
```

## Core packages

| Package | Version (freeze) |
|---------|------------------|
| torch | 2.14.0+cpu |
| torchvision | 0.29.0+cpu |
| numpy | 2.5.2 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| matplotlib | (pulled by reproduce script as needed) |

## Full freeze

Exact `pip freeze` snapshot: [`requirements-freeze.txt`](requirements-freeze.txt)
