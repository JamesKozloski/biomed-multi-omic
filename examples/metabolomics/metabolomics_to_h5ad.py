# metabolomics_to_h5ad.py (FIXED VERSION - Handles encoding and type errors)
"""
Convert MetaboLights metabolomics data to h5ad format for BMFM-RNA training.
Stores raw concentrations - transformations handled in training config.
Handles encoding errors and non-string metabolite identifiers.
"""

import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetabolomicsToH5ADConverter:
    """Convert MetaboLights MAF data to h5ad using 10K vocabulary"""
    
    def __init__(self, vocab_dir: str = "./bmfm_metabolomics_vocab_10k"):
        """Load the 10K vocabulary"""
        self.vocab_dir = Path(vocab_dir)
        self.vocab = self._load_vocabulary()
        logger.info(f"Loaded vocabulary with {len(self.vocab['metabolites'])} metabolites")
    
    def _load_vocabulary(self) -> Dict:
        """Load BMFM-RNA format vocabulary"""
        vocab_file = self.vocab_dir / "multifield_vocab.json"
        with open(vocab_file, 'r') as f:
            vocab_config = json.load(f)
        
        metabolite_tokens = vocab_config.get('metabolites', [])
        metabolite_to_idx = {name: idx for idx, name in enumerate(metabolite_tokens)}
        
        special_tokens = {}
        for token in ['[PAD]', '[CLS]', '[MASK]', '[UNK]']:
            if token in metabolite_to_idx:
                special_tokens[token] = metabolite_to_idx[token]
        
        return {
            'metabolites': metabolite_tokens,
            'metabolite_to_idx': metabolite_to_idx,
            'vocab_size': len(metabolite_tokens),
            'special_tokens': special_tokens
        }
    
    def convert_maf_to_h5ad(
        self, 
        maf_file: Path, 
        study_id: str,
        output_file: Optional[Path] = None
    ) -> Optional[ad.AnnData]:
        """
        Convert MAF file to h5ad format with raw concentrations.
        Returns None if data is pre-transformed or invalid.
        """
        logger.info(f"Converting {maf_file} to h5ad format")
        
        # Load MAF file with encoding fallback
        try:
            df = pd.read_csv(maf_file, sep='\t', low_memory=False, encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decode failed, trying latin-1 encoding")
            try:
                df = pd.read_csv(maf_file, sep='\t', low_memory=False, encoding='latin-1')
            except Exception as e:
                logger.error(f"Failed to read file with any encoding: {e}")
                return None
        except Exception as e:
            logger.error(f"Failed to read MAF file: {e}")
            return None
        
        logger.info(f"Loaded MAF with shape: {df.shape}")
        
        # Identify MAF structure
        try:
            metabolite_col, sample_cols = self._identify_maf_structure(df)
        except Exception as e:
            logger.error(f"Failed to identify MAF structure: {e}")
            return None
        
        logger.info(f"Found {len(sample_cols)} samples")
        
        # Clean and aggregate data
        df_clean = self._clean_and_aggregate_maf(df, metabolite_col, sample_cols)
        
        if df_clean is None or len(df_clean) == 0:
            return None
            
        logger.info(f"After cleaning: {len(df_clean)} unique metabolites")
        
        # Validate vocabulary coverage
        self._validate_coverage(df_clean, metabolite_col)
        
        # Build concentration matrix: samples × metabolites
        conc_matrix = df_clean[sample_cols].T.values.astype(np.float32)
        
        # Final check: reject if negative values present
        if (conc_matrix < 0).any():
            logger.warning(f"REJECTED: Contains negative values (pre-transformed data)")
            return None
        
        # Create AnnData object with RAW concentrations
        adata = ad.AnnData(
            X=conc_matrix,
            obs=pd.DataFrame(index=sample_cols),
            var=pd.DataFrame(index=df_clean[metabolite_col].values)
        )
        
        # Add metadata
        adata.obs['study_id'] = study_id
        adata.obs['sample_type'] = 'metabolomics'
        adata.obs['source'] = 'MetaboLights'
        
        adata.var['metabolite_name'] = df_clean[metabolite_col].values
        adata.var['in_vocabulary'] = adata.var['metabolite_name'].isin(self.vocab['metabolites'])
        adata.var['vocab_idx'] = [
            self.vocab['metabolite_to_idx'].get(m, self.vocab['special_tokens']['[UNK]'])
            for m in adata.var['metabolite_name']
        ]
        
        # Store concentration statistics
        conc_valid = conc_matrix[~np.isnan(conc_matrix)]
        adata.uns['concentration_stats'] = {
            'min': float(conc_valid.min()) if len(conc_valid) > 0 else 0.0,
            'max': float(conc_valid.max()) if len(conc_valid) > 0 else 0.0,
            'mean': float(conc_valid.mean()) if len(conc_valid) > 0 else 0.0,
            'median': float(np.median(conc_valid)) if len(conc_valid) > 0 else 0.0,
            'zeros': int((conc_matrix == 0).sum()),
            'nans': int(np.isnan(conc_matrix).sum()),
            'negatives': 0
        }
        
        logger.info(f"Created AnnData: {adata.shape} (samples × metabolites)")
        logger.info(f"Concentration range: [{adata.uns['concentration_stats']['min']:.2e}, "
                   f"{adata.uns['concentration_stats']['max']:.2e}]")
        logger.info(f"Valid values: {len(conc_valid)}/{conc_matrix.size} "
                   f"({100*len(conc_valid)/conc_matrix.size:.1f}%)")
        
        if output_file:
            adata.write_h5ad(output_file)
            logger.info(f"Saved h5ad to {output_file}")
        
        return adata
    
    def _clean_and_aggregate_maf(
        self, 
        df: pd.DataFrame, 
        metabolite_col: str, 
        sample_cols: List[str]
    ) -> Optional[pd.DataFrame]:
        """
        Clean MAF data and detect pre-transformed data.
        Returns None if data should be filtered out.
        """
        df_work = df.copy()
        
        # Convert metabolite column to string first (handles numeric/NaN values)
        df_work[metabolite_col] = df_work[metabolite_col].astype(str)
        
        # Remove rows with missing metabolite names
        df_work = df_work[df_work[metabolite_col].notna()].copy()
        df_work = df_work[df_work[metabolite_col].str.strip() != ''].copy()
        df_work = df_work[df_work[metabolite_col] != 'nan'].copy()  # Remove string 'nan'
        df_work = df_work[df_work[metabolite_col] != 'None'].copy()  # Remove string 'None'
        
        if len(df_work) == 0:
            logger.warning("REJECTED: No valid metabolite identifiers")
            return None
        
        # Convert sample columns to numeric, coercing errors to NaN
        for col in sample_cols:
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
        
        # Early detection: check if data is pre-transformed
        sample_check = df_work[sample_cols[:min(5, len(sample_cols))]].values.flatten()
        sample_check_valid = sample_check[~np.isnan(sample_check)]
        
        if len(sample_check_valid) > 0 and (sample_check_valid < 0).any():
            logger.warning("REJECTED: Contains negative values (pre-transformed data)")
            return None
        
        # Aggregate duplicates by taking mean across samples
        n_before = len(df_work)
        df_agg = df_work.groupby(metabolite_col)[sample_cols].mean().reset_index()
        n_after = len(df_agg)
        
        if n_before != n_after:
            logger.info(f"Aggregated {n_before} rows → {n_after} unique metabolites")
        
        return df_agg
    
    def _identify_maf_structure(self, df: pd.DataFrame) -> Tuple[str, List[str]]:
        """Identify metabolite column and sample concentration columns"""
        metabolite_col_patterns = [
            'database_identifier', 'metabolite_identification',
            'chemical_name', 'metabolite_name', 'compound_name'
        ]
        
        metabolite_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if any(pattern in col_lower for pattern in metabolite_col_patterns):
                metabolite_col = col
                break
        
        if not metabolite_col:
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    metabolite_col = col
                    break
            
            if not metabolite_col:
                raise ValueError("Could not identify metabolite name column")
            
            logger.warning(f"Using {metabolite_col} as metabolite column (fallback)")
        
        # Sample columns: potentially numeric columns
        sample_cols = []
        for col in df.columns:
            if col == metabolite_col:
                continue
            # Skip obvious metadata columns
            if col.lower() in ['chemical_formula', 'smiles', 'inchi', 'mass_to_charge', 
                              'retention_time', 'fragmentation']:
                continue
            # Try to convert to numeric
            try:
                pd.to_numeric(df[col], errors='coerce')
                sample_cols.append(col)
            except:
                pass
        
        if not sample_cols:
            raise ValueError("No numeric sample columns found")
        
        logger.info(f"Metabolite column: {metabolite_col}")
        logger.info(f"Sample columns: {len(sample_cols)}")
        
        return metabolite_col, sample_cols
    
    def _validate_coverage(self, df: pd.DataFrame, metabolite_col: str) -> None:
        """Validate vocabulary coverage"""
        maf_metabolites = set(df[metabolite_col].dropna())
        vocab_metabolites = set(self.vocab['metabolites']) - set(self.vocab['special_tokens'].keys())
        
        in_both = maf_metabolites & vocab_metabolites
        in_maf_only = maf_metabolites - vocab_metabolites
        
        coverage = len(in_both) / len(maf_metabolites) * 100 if maf_metabolites else 0
        
        logger.info(f"Vocabulary coverage: {len(in_both)}/{len(maf_metabolites)} ({coverage:.1f}%)")
        
        if in_maf_only and len(in_maf_only) <= 10:
            logger.warning(f"Metabolites not in vocabulary: {in_maf_only}")
    
    def convert_study_to_h5ad(
        self,
        study_id: str,
        data_dir: Path = Path("./data"),
        output_dir: Path = Path("./h5ad_data")
    ) -> List[ad.AnnData]:
        """Convert all MAF files from a study to h5ad"""
        study_dir = data_dir / study_id
        maf_files = list(study_dir.glob("m_*.tsv"))
        
        if not maf_files:
            logger.warning(f"No MAF files found for {study_id}")
            return []
        
        output_dir.mkdir(parents=True, exist_ok=True)
        adatas = []
        
        for maf_file in maf_files:
            output_file = output_dir / f"{study_id}_{maf_file.stem}.h5ad"
            try:
                adata = self.convert_maf_to_h5ad(maf_file, study_id, output_file)
                if adata is not None:
                    adatas.append(adata)
            except Exception as e:
                logger.error(f"Failed to convert {maf_file}: {e}")
                import traceback
                traceback.print_exc()
        
        return adatas
    
    def merge_studies(
        self,
        adatas: List[ad.AnnData],
        output_file: Path
    ) -> ad.AnnData:
        """Merge multiple h5ad files into single training dataset"""
        logger.info(f"Merging {len(adatas)} h5ad files")
        
        # Make variable names unique before merging
        for adata in adatas:
            adata.var_names_make_unique()
        
        # Concatenate
        merged = ad.concat(adatas, axis=0, join='outer', fill_value=0.0)
        
        # Update metadata
        merged.uns['n_studies'] = len(set(merged.obs['study_id']))
        merged.uns['n_samples'] = merged.n_obs
        merged.uns['n_metabolites'] = merged.n_vars
        
        logger.info(f"Merged dataset: {merged.shape} (samples × metabolites)")
        logger.info(f"Studies: {merged.uns['n_studies']}")
        
        merged.write_h5ad(output_file)
        logger.info(f"Saved merged dataset to {output_file}")
        
        return merged


def main():
    """Convert MetaboLights studies to h5ad format - CLEAN DATA ONLY"""
    print("="*70)
    print("METABOLOMICS TO H5AD CONVERTER - CLEAN RAW DATA ONLY")
    print("="*70)
    print("Filtering criteria:")
    print("  - Reject studies with negative values (pre-transformed)")
    print("  - Accept missing data (NaNs)")
    print("  - Handle encoding errors gracefully")
    print("="*70)
    
    converter = MetabolomicsToH5ADConverter()
    
    # Only process studies with clean raw concentration data
    studies = [
        'MTBLS1',   # NMR concentrations
        'MTBLS2',   # MS peak areas
        'MTBLS10',  # MS peak intensities
        'MTBLS25',  # NMR concentrations
        'MTBLS100', # NMR concentrations
    ]
    
    all_adatas = []
    for study_id in studies:
        print(f"\n{'='*70}")
        print(f"Converting {study_id}")
        print(f"{'='*70}")
        adatas = converter.convert_study_to_h5ad(study_id)
        all_adatas.extend(adatas)
    
    if all_adatas:
        print(f"\n{'='*70}")
        print("Creating merged training dataset")
        print(f"{'='*70}")
        merged = converter.merge_studies(
            all_adatas,
            output_file=Path("./h5ad_data/metabolomics_training_clean.h5ad")
        )
    
    print(f"\n{'='*70}")
    print("CONVERSION COMPLETE - CLEAN DATA ONLY")
    print(f"{'='*70}")
    print(f"Individual h5ad files: {len(all_adatas)}")
    if all_adatas:
        print(f"Total samples: {sum(a.n_obs for a in all_adatas)}")
        print(f"Unique metabolites: {len(set.union(*[set(a.var_names) for a in all_adatas]))}")
        total_values = sum(a.n_obs * a.n_vars for a in all_adatas)
        total_valid = sum((~np.isnan(a.X)).sum() for a in all_adatas)
        print(f"Data completeness: {total_valid}/{total_values} ({100*total_valid/total_values:.1f}% non-NaN)")
    print(f"Output directory: ./h5ad_data/")
    print(f"Merged file: ./h5ad_data/metabolomics_training_clean.h5ad")


if __name__ == "__main__":
    main()