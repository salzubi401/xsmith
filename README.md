# xsmith

Python workspace managed with [uv](https://docs.astral.sh/uv/).

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on your machine.

## Setup

Create the virtual environment and install dependencies:

```bash
uv venv
uv pip install ipykernel
```

Activate the environment (optional; you can also use `uv run …`):

```bash
source .venv/bin/activate
```

### Environment variables

```bash
cp .env.example .env
```

Edit `.env` with your local secrets and configuration. Never commit `.env`; tracked secrets belong only in `.env.example` as blank or dummy placeholders.

### Jupyter kernel

Register this environment as a Jupyter kernel (once per machine, after creating `.venv`):

```bash
.venv/bin/python -m ipykernel install --user --name=xsmith --display-name="Python (xsmith)"
```

In Jupyter or your editor, choose the kernel **Python (xsmith)**.
