from typing import List, Dict, Optional
# BMFM-RNA Compatible Metabolite Vocabulary Saver
# Mirrors BMFM-RNA vocabulary format exactly

import json
from pathlib import Path

def save_metabolite_vocabulary_bmfm_format(vocab_config, save_dir="./metabolite_tokenizer"):
    """
    Save metabolite vocabulary in BMFM-RNA compatible format.
    
    Mirrors the exact structure used by BMFM-RNA:
    - multifield_vocab.json (primary format)
    - individual field_vocab.txt files (secondary format)
    - 50 concentration bins (0-49) matching expression bins
    """
    
    # Create multifield vocabulary structure (exact BMFM-RNA format)
    multifield_vocab = {
        "metabolites": vocab_config['special_tokens'] + vocab_config['metabolites'],
        "concentrations": vocab_config['special_tokens'] + [str(i) for i in range(50)]  # 0-49 like expressions
    }
    
    # Create output directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON (primary format - mirrors MultiFieldVocabulary.save_to_json)
    json_path = save_path / "multifield_vocab.json"
    with open(json_path, "w") as f:
        json.dump(multifield_vocab, f, indent=4, sort_keys=True)
    
    # Save as individual text files (secondary format - mirrors MultiFieldVocabulary.save)
    for field, tokens in multifield_vocab.items():
        txt_path = save_path / f"{field}_vocab.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(tokens) + "\n")
    
    # Create summary stats
    stats = {
        "metabolites_vocab_size": len(multifield_vocab["metabolites"]),
        "concentrations_vocab_size": len(multifield_vocab["concentrations"]),
        "total_metabolites": len(vocab_config['metabolites']),
        "special_tokens": vocab_config['special_tokens'],
        "source_studies": vocab_config.get('source_studies', []),
        "concentration_bins": 50
    }
    
    # Save vocabulary stats
    stats_path = save_path / "vocab_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Saved BMFM-RNA compatible vocabulary to {save_dir}")
    print(f"   📁 Files created:")
    print(f"      - multifield_vocab.json ({len(multifield_vocab)} fields)")
    print(f"      - metabolites_vocab.txt ({len(multifield_vocab['metabolites'])} tokens)")
    print(f"      - concentrations_vocab.txt ({len(multifield_vocab['concentrations'])} tokens)")
    print(f"      - vocab_stats.json (metadata)")
    print(f"   🧪 Metabolites: {len(vocab_config['metabolites'])}")
    print(f"   📊 Concentration bins: 50 (0-49, matching BMFM-RNA expressions)")
    
    return save_path, multifield_vocab, stats

def load_metabolite_vocabulary_bmfm_format(vocab_dir):
    """Load vocabulary in BMFM-RNA format (for testing/validation)"""
    
    vocab_path = Path(vocab_dir)
    
    # Load JSON format
    json_path = vocab_path / "multifield_vocab.json"
    with open(json_path, "r") as f:
        multifield_vocab = json.load(f)
    
    # Load stats
    stats_path = vocab_path / "vocab_stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        stats = {}
    
    print(f"✅ Loaded BMFM-RNA compatible vocabulary from {vocab_dir}")
    print(f"   Fields: {list(multifield_vocab.keys())}")
    print(f"   Metabolites: {len(multifield_vocab.get('metabolites', [])) - len(stats.get('special_tokens', []))}")
    
    return multifield_vocab, stats

# Usage with your existing vocabulary
def save_current_vocabulary(vocab_config):
    """Save the vocabulary you just built"""
    
    print("=== Saving Metabolite Vocabulary in BMFM-RNA Format ===")
    
    # Save vocabulary
    save_path, multifield_vocab, stats = save_metabolite_vocabulary_bmfm_format(
        vocab_config, 
        save_dir="./bmfm_metabolomics_vocab"
    )
    
    # Validate it can be loaded
    print("\n=== Validation: Testing Load ===")
    loaded_vocab, loaded_stats = load_metabolite_vocabulary_bmfm_format(save_path)
    
    # Show sample content
    print(f"\n=== Sample Vocabulary Content ===")
    print(f"First 10 metabolites: {multifield_vocab['metabolites'][:10]}")
    print(f"First 10 concentrations: {multifield_vocab['concentrations'][:10]}")
    print(f"Last 5 concentrations: {multifield_vocab['concentrations'][-5:]}")
    
    print(f"\n🎯 Ready for BMFM-RNA integration!")
    print(f"   Use: MultiFieldVocabulary.load_from_json('{save_path}/multifield_vocab.json')")
    
    return save_path

# Add caching check to your streamlined vocabulary builder
def build_metabolite_vocabulary_with_save(study_ids: List[str], output_dir: str = "./data", vocab_save_dir: str = "./bmfm_metabolomics_vocab"):
    """Enhanced version that saves vocabulary in BMFM format"""
    
    # Check if vocabulary already exists
    vocab_path = Path(vocab_save_dir)
    if (vocab_path / "multifield_vocab.json").exists():
        print(f"✅ Found existing vocabulary at {vocab_save_dir}")
        return load_metabolite_vocabulary_bmfm_format(vocab_save_dir)
    
    # Build vocabulary (using your existing streamlined code)
    from streamlined_metabolite_vocab import build_metabolite_vocabulary
    vocab_config = build_metabolite_vocabulary(study_ids, output_dir)
    
    # Save in BMFM-RNA format
    save_current_vocabulary(vocab_config)
    
    return vocab_config

if __name__ == "__main__":
    # Assuming you have vocab_config from your successful run
    print("Ready to save your 455-metabolite vocabulary in BMFM-RNA format!")
    print("Run: save_current_vocabulary(vocab_config)")