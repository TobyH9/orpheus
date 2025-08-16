# Orpheus

## Getting Started

First, clone this repository:

```bash
git clone https://github.com/TobyH9/orpheus.git
cd orpheus
```
# Orpheus

This is an implementation of Andrej Kaparthy's nanogpt.

## Environment Setup (using uv)

1. Install [uv](https://github.com/astral-sh/uv) if you don’t have it:
	```bash
	pip install uv
	```
2. Create and activate a virtual environment:
	```bash
	uv venv .venv
	source .venv/bin/activate
	```
3. Install dependencies from `pyproject.toml`:
	```bash
	uv sync
	```

Optional: install the package in editable mode (so local changes are picked up):
```bash
uv pip install -e .
```

You can now run the training script:
```bash
python train.py
```

Or using uv to ensure the right environment:
```bash
uv run python train.py
```

To resume from a checkpoint:
```bash
python train.py --resume-from checkpoints/orpheus_final.pt
```
