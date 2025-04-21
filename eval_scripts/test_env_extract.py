import numpy as np
import os
import argparse
import torch
from collapse.data import EmbedTransform
from atom3d.datasets import load_dataset, make_lmdb_dataset
import atom3d.util.file as fi
from collapse import initialize_model
import collections as col
from collapse.data import atom_info
from collapse.data import sample_functional_center
import scipy.spatial

def compare_environment_extraction(df, ch_resid, env_radius=10.0, ca_center=False):
    # Original method using KDTree
    import scipy.spatial
    df_orig = df.copy()
    kd_tree = scipy.spatial.cKDTree(df_orig[['x','y','z']].to_numpy().astype(np.float32))
    if ca_center:
        try:
            center = df_orig[df_orig['name'] == 'CA'][['x','y','z']].to_numpy()[0]
        except Exception:
            return None
    else:
        center = sample_functional_center(df_orig, ch_resid[1], False)
    idxs_kdtree = kd_tree.query_ball_point(center, r=env_radius)
    
    # Optimized method using vectorized distance computation
    device = torch.device('cpu')
    coords = torch.as_tensor(df[['x','y','z']].to_numpy().astype(np.float32), device=device)
    center_tensor = torch.as_tensor(center, dtype=torch.float32, device=device)
    dists = torch.norm(coords - center_tensor, dim=1)
    neighbor_mask = dists < env_radius
    idxs_vectorized = torch.where(neighbor_mask)[0].cpu().numpy().tolist()
    
    print("KDTree found " + str(len(idxs_kdtree)) + " neighbors")
    print("Vectorized found " + str(len(idxs_vectorized)) + " neighbors")

    return idxs_kdtree, idxs_vectorized

# Use this function on a single example:
import os
import argparse
import numpy as np
import torch
from atom3d.datasets import load_dataset
from collapse import initialize_model
import sys

# Import the two embedding functions.
# The original embedding function:
from collapse.data import embed_protein
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from test_embed import opt_embed_protein

def main():
    parser = argparse.ArgumentParser(
        description="Compare original vs. optimized embeddings for the same proteins."
    )
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing the input dataset")
    parser.add_argument("--out_file", default="debug_comparison.txt",
                        help="Path to the output txt file for comparison results")
    parser.add_argument("--checkpoint", default="data/checkpoints/collapse_base.pt",
                        help="Path to the model checkpoint")
    parser.add_argument("--filetype", default="pdb",
                        help="Input file type (default: pdb)")
    parser.add_argument("--num_examples", type=int, default=10,
                        help="Number of protein examples to compare")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(args.checkpoint, device=device)
    
    # Load the dataset without applying any transform
    dataset = load_dataset(args.data_dir, args.filetype, transform=None)
    
    print("Loaded dataset with {} examples.".format(len(dataset)))
    count = 0
    
    with open(args.out_file, "w") as f:
        # Write header line.
        f.write("ProteinID\tSequence\tOriginalEmbedding\tOptimizedEmbedding\tL2Difference\n")
        for elem in dataset:
            if elem is None:
                continue
            # Use element 'id' if available; otherwise, generate one.
            prot_id = elem.get("id", "protein_{}".format(count))
            atoms_df = elem.get("atoms")
            if atoms_df is None:
                continue
            
            # Create a simple sequence string from the atom dataframe.
            # Here we take the unique residue names (in order) as a proxy for sequence.
            if "resname" in atoms_df.columns and "residue" in atoms_df.columns:
                unique_res = atoms_df.drop_duplicates(subset=["residue"])["resname"].tolist()
                sequence = "".join(unique_res)
            else:
                sequence = "N/A"
            
            # Compute original embedding.
            orig_data = embed_protein(atoms_df, model, device=device,
                                      include_hets=True, env_radius=10.0)
            if orig_data is None or "embeddings" not in orig_data:
                print("Skipping {}: original embedding not generated".format(prot_id))
                continue
            orig_emb = orig_data["embeddings"]
            
            # Compute optimized embedding.
            opt_data = opt_embed_protein(atoms_df, model, device=device,
                                         include_hets=True, env_radius=10.0)
            if opt_data is None or "embeddings" not in opt_data:
                print("Skipping {}: optimized embedding not generated".format(prot_id))
                continue
            opt_emb = opt_data["embeddings"]
            
            # Average the embeddings over all residues to obtain one vector per protein.
            orig_vec = orig_emb.mean(axis=0)
            opt_vec = opt_emb.mean(axis=0)
            
            # Compute L2 norm difference.
            l2_diff = np.linalg.norm(orig_vec - opt_vec)
            
            # Write the results to the file.
            f.write("{}\t{}\t{}\t{}\t{}\n".format(prot_id, sequence, orig_vec.tolist(), opt_vec.tolist(), l2_diff))
            print("Processed {}: L2 difference = {:.4f}".format(prot_id, l2_diff))
            count += 1
            if count >= args.num_examples:
                break

    print("Comparison results written to {}".format(args.out_file))

if __name__ == "__main__":
    main()
