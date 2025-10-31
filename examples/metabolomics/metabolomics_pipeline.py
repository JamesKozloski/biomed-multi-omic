#!/usr/bin/env python3
"""
Integrated MetaboLights to BMFM-RNA Pipeline
Corrected to match BMFM-RNA's exact tokenizer format
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import anndata as ad
import numpy as np
from ftplib import FTP

class MetabolomicsPipeline:
    """Pipeline for processing MetaboLights data into BMFM-RNA format"""
    
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.merged_dir = self.base_dir / "merged"
        self.vocab_dir = self.base_dir / "vocab"
        
        # Create directories
        for dir_path in [self.raw_dir, self.merged_dir, self.vocab_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        self.stats = {
            'studies_processed': 0,
            'studies_failed': 0,
            'total_samples': 0,
            'total_metabolites': set()
        }
    
    def download_study(self, study_id: str) -> Path:
        """Download MAF files from MetaboLights FTP"""
        print(f"\nDownloading {study_id}...")
        study_dir = self.raw_dir / study_id
        study_dir.mkdir(exist_ok=True)
        
        try:
            ftp = FTP('ftp.ebi.ac.uk')
            ftp.login()
            ftp.cwd(f'/pub/databases/metabolights/studies/public/{study_id}')
            
            files = ftp.nlst()
            maf_files = [f for f in files if f.startswith('m_') and f.endswith('.txt')]
            
            if not maf_files:
                print(f"  ⚠ No MAF files found")
                return None
            
            for maf_file in maf_files:
                local_path = study_dir / maf_file
                if not local_path.exists():
                    with open(local_path, 'wb') as f:
                        ftp.retrbinary(f'RETR {maf_file}', f.write)
                    print(f"  ✓ Downloaded: {maf_file}")
            
            ftp.quit()
            return study_dir
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None
    
    def extract_metabolites_from_maf(self, maf_path: Path) -> List[str]:
        """Extract metabolite identifiers from MAF file"""
        try:
            df = pd.read_csv(maf_path, sep='\t', low_memory=False)
            
            # Look for metabolite identifier columns
            id_columns = [
                'database_identifier', 'metabolite_identification',
                'chemical_name', 'metabolite_name', 'compound_name'
            ]
            
            metabolites = set()
            for col in id_columns:
                if col in df.columns:
                    valid_ids = df[col].dropna().astype(str)
                    valid_ids = valid_ids[valid_ids != 'nan']
                    valid_ids = valid_ids[valid_ids != '']
                    metabolites.update(valid_ids)
            
            return sorted(list(metabolites))
            
        except Exception as e:
            print(f"  ⚠ Error parsing {maf_path.name}: {e}")
            return []
    
    def process_study(self, study_id: str) -> ad.AnnData:
        """Process a single study into AnnData format"""
        study_dir = self.raw_dir / study_id
        
        # Download if needed
        if not study_dir.exists():
            study_dir = self.download_study(study_id)
            if study_dir is None:
                return None
        
        # Find MAF files
        maf_files = list(study_dir.glob('m_*.txt'))
        if not maf_files:
            print(f"  ⚠ No MAF files in {study_dir}")
            return None
        
        # Extract metabolites from all MAF files
        all_metabolites = set()
        for maf_file in maf_files:
            metabolites = self.extract_metabolites_from_maf(maf_file)
            all_metabolites.update(metabolites)
        
        if not all_metabolites:
            print(f"  ⚠ No metabolites extracted")
            return None
        
        metabolites_list = sorted(list(all_metabolites))
        n_metabolites = len(metabolites_list)
        n_samples = len(maf_files)  # Simplified: one sample per MAF file
        
        # Create sparse random data for demonstration
        # In production, extract actual concentration values
        X = np.random.rand(n_samples, n_metabolites)
        
        # Create AnnData
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=[f"sample_{i}" for i in range(n_samples)]),
            var=pd.DataFrame(index=metabolites_list)
        )
        
        adata.obs['study_id'] = study_id
        
        # Update stats
        self.stats['total_metabolites'].update(metabolites_list)
        self.stats['total_samples'] += n_samples
        
        print(f"  ✓ Processed: {n_samples} samples, {n_metabolites} metabolites")
        return adata
    
    def build_vocabulary(self, adatas: List[ad.AnnData]) -> Tuple[Dict, List[str], List[str]]:
        """
        Build BMFM-compatible vocabulary from datasets
        
        CRITICAL: Matches BMFM-RNA's create_gene2vec_tokenizer.py exactly:
        - special_tokens prepended to BOTH genes and expressions fields
        - expressions field has special_tokens + ["0", "1", ..., "49"]
        """
        # Collect all unique metabolites
        all_metabolites = set()
        for adata in adatas:
            all_metabolites.update(adata.var_names)
        
        # Sort alphabetically
        metabolites = sorted(list(all_metabolites))
        
        # Special tokens (BMFM-RNA standard order)
        special_tokens = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
        
        # Create vocabulary config matching BMFM-RNA exactly
        vocab_config = {
            'genes': special_tokens + metabolites,
            'expressions': special_tokens + [str(i) for i in range(50)]  # 5 special + 50 bins = 55 total
        }
        
        return vocab_config, metabolites, special_tokens
    
    def save_bmfm_vocabulary(self, vocab_config: Dict, metabolites: List[str], 
                            special_tokens: List[str]):
        """
        Save vocabulary in BMFM-RNA format (3 root files only)
        
        BMFM-RNA's from_old_multifield_tokenizer() will automatically:
        1. Read multifield_vocab.json
        2. Create BertTokenizerFast for each field
        3. Save to tokenizers/{field}/ with proper HuggingFace format
        """
        print("\n[Saving BMFM-RNA Compatible Vocabulary]")
        
        vocab_base = self.vocab_dir / "all_genes_vocab"
        vocab_base.mkdir(exist_ok=True, parents=True)
        
        # 1. Save multifield_vocab.json (PRIMARY FILE)
        multifield_path = vocab_base / "multifield_vocab.json"
        with open(multifield_path, 'w') as f:
            json.dump(vocab_config, f, indent=2)
        print(f"✓ Saved: {multifield_path}")
        
        # 2. Save special_tokens_map.json
        special_tokens_map = {
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
            "pad_token": "[PAD]",
            "sep_token": "[SEP]",
            "unk_token": "[UNK]"
        }
        special_tokens_path = vocab_base / "special_tokens_map.json"
        with open(special_tokens_path, 'w') as f:
            json.dump(special_tokens_map, f, indent=2)
        print(f"✓ Saved: {special_tokens_path}")
        
        # 3. Save tokenizer_config.json
        tokenizer_config = {
            "clean_up_tokenization_spaces": True,
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
            "model_max_length": 1000000000000000019884624838656,
            "pad_token": "[PAD]",
            "sep_token": "[SEP]",
            "strip_accents": None,
            "tokenizer_class": "MultiFieldTokenizer",
            "unk_token": "[UNK]"
        }
        tokenizer_config_path = vocab_base / "tokenizer_config.json"
        with open(tokenizer_config_path, 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        print(f"✓ Saved: {tokenizer_config_path}")
        
        # Print vocabulary statistics
        print(f"\n📊 Vocabulary Statistics:")
        print(f"  Special tokens: {len(special_tokens)}")
        print(f"  Metabolites: {len(metabolites)}")
        print(f"  Total 'genes' field: {len(vocab_config['genes'])}")
        print(f"  Total 'expressions' field: {len(vocab_config['expressions'])} (5 special + 50 bins)")
        print(f"\n✓ BMFM-RNA will auto-convert to HuggingFace format on first load")
    
    def merge_datasets(self, adatas: List[ad.AnnData]) -> ad.AnnData:
        """Merge all datasets into single AnnData"""
        print("Merging datasets...")
        
        if len(adatas) == 0:
            raise ValueError("No datasets to merge")
        
        if len(adatas) == 1:
            return adatas[0]
        
        # Merge with outer join to keep all metabolites
        merged = ad.concat(adatas, join='outer', merge='same')
        
        # Fill NaN with zeros
        if np.isnan(merged.X).any():
            merged.X = np.nan_to_num(merged.X, nan=0.0)
        
        print(f"  ✓ Merged shape: {merged.shape}")
        return merged
    
    def run_pipeline(self, study_ids: List[str]):
        """Run complete pipeline"""
        start_time = time.time()
        
        print("="*70)
        print("METABOLOMICS PIPELINE - BMFM-RNA FORMAT")
        print("="*70)
        
        # Process studies
        print("\n[Step 1/4] Processing studies...")
        all_adatas = []
        
        for study_id in study_ids:
            try:
                adata = self.process_study(study_id)
                if adata is not None:
                    all_adatas.append(adata)
                    self.stats['studies_processed'] += 1
                else:
                    self.stats['studies_failed'] += 1
            except Exception as e:
                print(f"  ✗ Failed {study_id}: {e}")
                self.stats['studies_failed'] += 1
        
        if not all_adatas:
            raise ValueError("No studies successfully processed")
        
        # Build vocabulary
        print("\n[Step 2/4] Building BMFM-RNA vocabulary...")
        vocab_config, metabolites, special_tokens = self.build_vocabulary(all_adatas)
        
        # Save vocabulary
        print("\n[Step 3/4] Saving vocabulary...")
        self.save_bmfm_vocabulary(vocab_config, metabolites, special_tokens)
        
        # Merge datasets
        print("\n[Step 4/4] Merging datasets...")
        merged = self.merge_datasets(all_adatas)
        
        # Save merged dataset
        output_file = self.merged_dir / "metabolomics_training.h5ad"
        merged.write_h5ad(output_file)
        print(f"✓ Saved merged dataset: {output_file}")
        
        # Save statistics
        self.stats['processing_time'] = time.time() - start_time
        stats_to_save = self.stats.copy()
        stats_to_save['total_metabolites'] = len(self.stats['total_metabolites'])
        
        stats_file = self.vocab_dir / "pipeline_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        
        # Print summary
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"Studies processed: {self.stats['studies_processed']}")
        print(f"Studies failed: {self.stats['studies_failed']}")
        print(f"Total samples: {self.stats['total_samples']}")
        print(f"Unique metabolites: {len(self.stats['total_metabolites'])}")
        print(f"Processing time: {self.stats['processing_time']/60:.1f} minutes")
        print(f"\n📁 Outputs:")
        print(f"  - Merged dataset: {output_file}")
        print(f"  - Vocabulary: {self.vocab_dir / 'all_genes_vocab'}")
        print(f"  - Statistics: {stats_file}")
        print(f"\n🚀 Next: Run BMFM-RNA training with vocabulary at {self.vocab_dir / 'all_genes_vocab'}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Integrated MetaboLights to BMFM-RNA Pipeline (Corrected)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 3 studies
  python metabolomics_pipeline.py --studies MTBLS1 MTBLS2 MTBLS3
  
  # Full processing
  python metabolomics_pipeline.py --studies MTBLS1 MTBLS2 ... MTBLS50 --base-dir ./bmfm-metabolomics
        """
    )
    
    parser.add_argument(
        '--studies',
        nargs='+',
        required=True,
        help='MetaboLights study IDs (e.g., MTBLS1 MTBLS2)'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        default='./data',
        help='Base directory for all outputs (default: ./data)'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = MetabolomicsPipeline(base_dir=args.base_dir)
    pipeline.run_pipeline(args.studies)


if __name__ == '__main__':
    main()