# Optimized COLLAPSE Inference Pipeline

This repository is focused on **optimized embedding/inference** for COLLAPSE.

## Kept core implementation
- `collapse/data.py` (core data/model utilities)
- `collapse/embedding_utils.py` (optimized graph preparation path)
- `gen_embed.py` (optimized batch embedding CLI)
- `gen_embed/` (module entrypoint wrapper)

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Input data
Place PDB files under a directory (nested folders allowed). Example:
```text
input_pdbs/
  1abc.pdb
  2xyz.pdb.gz
```

## Run optimized embedding pipeline
```bash
python gen_embed.py INPUT_DIR OUTPUT_DIR --filetype pdb --k_atoms 32
```
Or module mode:
```bash
python -m gen_embed INPUT_DIR OUTPUT_DIR --filetype pdb --k_atoms 32
```

Useful flags:
- `--checkpoint data/checkpoints/collapse_base.pt`
- `--num_splits N --split_id K` for chunked runs

## Outputs
The output is an LMDB dataset. Each entry includes:
- `embeddings`: residue embeddings (N x 512)
- `resids`: residue identifiers
- `chains`: chain identifiers
- `confidence`: per-residue confidence values
- original ATOM3D item fields (`id`, `atoms`)

## Tests
```bash
pytest -q tests/test_repo_smoke.py
```

## Quick benchmark
```bash
python tests/benchmark_transform.py
```
This reports transform throughput (`per_graph_ms`) on CPU.
