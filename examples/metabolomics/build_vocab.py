#!/usr/bin/env python3
"""
Standalone vocabulary builder for Mountcastle deployment.
Builds vocabulary from MetaboLights studies and saves in BMFM format.

Usage:
    python build_vocab.py

Output:
    ./bmfm_metabolomics_vocab/  - Vocabulary directory with BMFM-compatible files
"""

import sys
from pathlib import Path

from streamlined_metabolite_vocab import build_metabolite_vocabulary
from vocabulary_saver import save_metabolite_vocabulary_bmfm_format

def main():
    print("=" * 70)
    print("🧬 BMFM-NVP: Metabolite Vocabulary Builder")
    print("=" * 70)
    
    # Define studies for vocabulary construction
    study_ids = ['MTBLS1', 'MTBLS2', 'MTBLS10']
    
    print(f"\n📚 Building vocabulary from {len(study_ids)} MetaboLights studies...")
    print(f"   Studies: {', '.join(study_ids)}")
    
    # Build vocabulary from MetaboLights
    vocab_config = build_metabolite_vocabulary(
        study_ids=study_ids,
        output_dir="./data"
    )
    
    print(f"\n✅ Vocabulary built: {len(vocab_config['metabolites'])} metabolites")
    
    # Save in BMFM-RNA compatible format
    print("\n💾 Saving vocabulary in BMFM-RNA compatible format...")
    save_path, multifield_vocab, stats = save_metabolite_vocabulary_bmfm_format(
        vocab_config,
        save_dir="./bmfm_metabolomics_vocab"
    )
    
    print(f"\n🎯 Vocabulary saved to: {save_path}")
    print(f"   Ready for training!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
