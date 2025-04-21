import re
from collections import defaultdict
import pandas as pd

def parse_annotation_log(log_path: str):
    """
    Parses an annotation log file into a structured dictionary:
    {
      'PDB_ID': {
          'Motif description': ['CHAIN_RESID', ...],
           ...
      },
      ...
    }
    """
    results = defaultdict(lambda: defaultdict(list))
    current_pdb = None
    current_motif = None

    # Regex patterns:
    pdb_pattern = re.compile(r"^Input PDB:\s*(\S+)")
    # Match lines that contain a motif description ending with (PROSITE) or (M-CSA).
    motif_pattern = re.compile(r"^\s*(.+signature.*\((PROSITE|M-CSA)\))\s*$", re.IGNORECASE)
    # Match residue annotation lines: dash followed by residue, e.g. "    - A_Y28: 1 PDBs"
    residue_pattern = re.compile(r"^\s*-\s*([A-Z]_[A-Z0-9]+):")

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Check for PDB ID line
            pdb_match = pdb_pattern.match(line)
            if pdb_match:
                current_pdb = pdb_match.group(1).strip()
                current_motif = None  # reset current motif
                continue

            # Check if the line is a motif description
            motif_match = motif_pattern.match(line)
            if motif_match:
                # Remove trailing dot if any
                current_motif = motif_match.group(1).strip().rstrip('.')
                continue

            # Check for residue annotation lines
            residue_match = residue_pattern.match(line)
            if residue_match and current_pdb and current_motif:
                residue = residue_match.group(1).strip()
                results[current_pdb][current_motif].append(residue)

    return results

def compare_annotations(ori_results, opt_results):
    pdb_ids = set(ori_results.keys()) | set(opt_results.keys())
    summary = []

    for pdb in sorted(pdb_ids):
        motifs_ori = set(ori_results[pdb].keys())
        motifs_opt = set(opt_results[pdb].keys())
        all_motifs = motifs_ori | motifs_opt

        motif_jaccards = []
        for motif in all_motifs:
            r1 = set(ori_results[pdb].get(motif, []))
            r2 = set(opt_results[pdb].get(motif, []))
            if r1 or r2:
                jaccard = len(r1 & r2) / len(r1 | r2)
                motif_jaccards.append(jaccard)

        exact_match = ori_results[pdb] == opt_results[pdb]
        summary.append({
            'PDB_ID': pdb,
            'ExactMatch': exact_match,
            'NumMotifs_Ori': len(motifs_ori),
            'NumMotifs_Opt': len(motifs_opt),
            'SharedMotifs': len(motifs_ori & motifs_opt),
            'JaccardAvg': sum(motif_jaccards) / len(motif_jaccards) if motif_jaccards else None,
        })

    return pd.DataFrame(summary)

ori_log = "/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/annotation_63429904.log"
opt_log = "/scratch/groups/rbaltman/ziyiw23/COLLAPSE/logs/annotation_63429902.log"

ori_results = parse_annotation_log(ori_log)
opt_results = parse_annotation_log(opt_log)
summary_df = compare_annotations(ori_results, opt_results)

# Save and print summary
summary_df.to_csv("annotation_comparison_summary.csv", index=False)
unmatched = summary_df[summary_df['ExactMatch'] == False]
unmatched.to_csv("annotation_comparison_unmatched.csv", index=False)
print(unmatched)
print(f'{len(unmatched)} out of {len(summary_df)} PDBs do not match ({len(unmatched)/len(summary_df)}).')
