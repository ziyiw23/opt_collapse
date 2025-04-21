import os
import gzip
import pickle
import shutil
import lmdb
from pathlib import Path

def is_human_pdb(pdb_path):
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("SOURCE") and ("HOMO SAPIENS" in line.upper() or "TAXID: 9606" in line.upper()):
                return True
    return False

def get_lmdb_pdb_id(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            key_str = key.decode('utf-8', errors='replace')
            if key_str in ['id_to_idx', 'num_examples', 'serialization_format']:
                continue
            try:
                decompressed = gzip.decompress(value)
                data = pickle.loads(decompressed)
                return key_str  # assume key is the PDB ID
            except Exception:
                continue
    return None

def filter_human_lmdbs(lmdb_root_dir, pdb_dir, output_dir):
    lmdb_root_dir = Path(lmdb_root_dir)
    pdb_dir = Path(pdb_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for lmdb_subdir in lmdb_root_dir.iterdir():
        if not lmdb_subdir.is_dir():
            continue

        pdb_id = get_lmdb_pdb_id(str(lmdb_subdir))
        if not pdb_id:
            print(f"[SKIP] Could not determine ID from {lmdb_subdir}")
            continue

        matching_pdb = pdb_dir / f"{pdb_id}.pdb"
        if not matching_pdb.exists():
            print(f"[SKIP] No matching PDB file for {pdb_id}")
            continue

        if is_human_pdb(matching_pdb):
            dest_path = output_dir / lmdb_subdir.name
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(lmdb_subdir, dest_path)
            print(f"[COPY] {pdb_id} is human. Copied to {dest_path}")
        else:
            print(f"[SKIP] {pdb_id} is not human.")

filter_human_lmdbs(
    lmdb_root_dir="/scratch/groups/rbaltman/ziyiw23/clps_embed/ori_res",
    pdb_dir="/scratch/groups/rbaltman/ziyiw23/clps_pdbs",
    output_dir="/scratch/groups/rbaltman/ziyiw23/clps_embed/human"
)