#!/usr/bin/env python3
"""
Ring statistics script.
Input: SMILES file path.
Output: frequency statistics for carbon rings appearing in the dataset.

Based on the count_ring_types function from JT-Tree.ipynb.
"""

import os
import sys
from collections import defaultdict
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    print("Error: rdkit is required.")
    print("Install via: pip install rdkit")
    sys.exit(1)


def count_ring_types(smiles_list):
    """
    Count all ring types and frequencies in a list of SMILES strings.
    
    Args:
        smiles_list (list of str): molecular SMILES strings.
    
    Returns:
        dict: ring SMILES mapped to occurrence counts.
    """
    ring_counts = defaultdict(int)
    
    print(f"Processing {len(smiles_list)} molecules...")
    
    for smi in tqdm(smiles_list, desc="Counting ring types"):
        # Parse SMILES into an RDKit molecule
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Warning: failed to parse SMILES: {smi}")
            continue
        
        # Extract ring information
        ri = mol.GetRingInfo()
        bond_rings = ri.BondRings()
        
        for bond_ring in bond_rings:
            # Build sub-molecule corresponding to the ring
            submol = Chem.PathToSubmol(mol, bond_ring)
            
            # Generate canonical SMILES
            ring_smiles = Chem.MolToSmiles(submol, canonical=True)
            
            # Update counts
            ring_counts[ring_smiles] += 1
    
    return dict(ring_counts)


def load_smiles_from_file(file_path):
    """
    Load a list of SMILES strings from a file.
    
    Args:
        file_path (str): path to the SMILES file
    
    Returns:
        list: SMILES strings
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    smiles_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comment lines
                smiles_list.append(line)
    
    print(f"Loaded {len(smiles_list)} SMILES strings from {file_path}")
    return smiles_list


def save_ring_statistics(ring_counts, output_file):
    """
    Save ring statistics to a text file
    
    Args:
        ring_counts (dict): ring statistics
        output_file (str): output file path
    """
    # Sort by occurrence frequency
    sorted_rings = sorted(ring_counts.items(), key=lambda x: x[1], reverse=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Ring statistics\n")
        f.write("# Format: ring_SMILES: occurrences\n")
        f.write(f"# Number of unique rings: {len(ring_counts)}  unique rings\n")
        f.write(f"# Total rings: {sum(ring_counts.values())}\n\n")
        
        for ring_smiles, count in sorted_rings:
            f.write(f"{ring_smiles}: {count}\n")
    
    print(f"Ring statistics saved to: {output_file}")


def print_ring_statistics(ring_counts, top_n=20):
    """
    Print ring statistics
    
    Args:
        ring_counts (dict): ring statistics
        top_n (int): number of most frequent rings to show
    """
    print(f"\n=== Ring Statistics ===")
    print(f"Number of unique rings: {len(ring_counts)}  unique rings")
    print(f"Total rings: {sum(ring_counts.values())}")
    
    # Sort by occurrence frequency
    sorted_rings = sorted(ring_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {min(top_n, len(sorted_rings))}  most frequent rings:")
    print("-" * 50)
    for i, (ring_smiles, count) in enumerate(sorted_rings[:top_n], 1):
        percentage = (count / sum(ring_counts.values())) * 100
        print(f"{i:2d}. {ring_smiles:20s}: {count:6d} occurrences ({percentage:5.2f}%)")


def main():
    """CLI entry point"""
    if len(sys.argv) != 2:
        print("Usage: python ring_statistics.py <SMILES file path>")
        print("Example: python ring_statistics.py test.smiles")
        sys.exit(1)
    
    smiles_file = sys.argv[1]
    
    try:
        # Load SMILES file
        smiles_list = load_smiles_from_file(smiles_file)
        
        if not smiles_list:
            print("Error: no valid SMILES strings found in the file")
            sys.exit(1)
        
        # Count ring types
        ring_counts = count_ring_types(smiles_list)
        
        if not ring_counts:
            print("Warning: no rings detected")
            sys.exit(0)
        
        # Print statistics
        print_ring_statistics(ring_counts)
        
        # Save results to file
        base_name = os.path.splitext(os.path.basename(smiles_file))[0]
        output_file = f"{base_name}_ring_statistics.txt"
        save_ring_statistics(ring_counts, output_file)
        
        print(f"\n✅ Ring statistics completed！")
        print(f"📊 Found {len(ring_counts)}  unique rings")
        print(f"📁 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
