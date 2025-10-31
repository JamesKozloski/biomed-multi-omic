# Streamlined MetaboLights Metabolite Vocabulary Builder
# Simplified version - discovery code removed

import pandas as pd
import requests
from pathlib import Path
import re
from typing import List, Dict, Set
import json

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

def get_study_files(study_id: str) -> List[str]:
    """Get file list for a MetaboLights study"""
    url = f"https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{study_id}/"
    
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        return []
    
    if BS4_AVAILABLE:
        soup = BeautifulSoup(response.content, 'html.parser')
        files = [link.get('href') for link in soup.find_all('a') 
                if link.get('href') and not link.get('href').startswith('/') and link.get('href') != '../']
    else:
        # Fallback regex parsing
        files = re.findall(r'<a href="([^"]+\.txt)"', response.text)
    
    return files

def download_file(study_id: str, filename: str, output_dir: Path) -> Path:
    """Download a file from MetaboLights"""
    url = f"https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{study_id}/{filename}"
    output_path = output_dir / study_id / filename
        
    # Check if file already exists
    if output_path.exists():
        print(f"    ✓ Using cached {filename}")
        return output_path

    else:
        # Only download if missing
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
    return output_path

def find_maf_files(study_id: str, output_dir: Path) -> List[str]:
    """Find MAF (Metabolite Assignment File) references for a study"""
    files = get_study_files(study_id)
    
    # Download assay files to find MAF references
    assay_files = [f for f in files if f.startswith('a_') and f.endswith('.txt')]
    maf_files = []
    
    for assay_file in assay_files:
        try:
            assay_path = download_file(study_id, assay_file, output_dir)
            df = pd.read_csv(assay_path, sep='\t', low_memory=False)
            
            # Find MAF file references
            for col in df.columns:
                if 'metabolite assignment' in col.lower():
                    maf_refs = df[col].dropna().unique()
                    maf_files.extend([str(ref) for ref in maf_refs if str(ref) != 'nan'])
        except:
            continue
    
    return list(set(maf_files))

def extract_metabolites_from_maf(maf_df: pd.DataFrame) -> Set[str]:
    """Extract metabolite identifiers from MAF dataframe"""
    metabolites = set()
    
    # Known MAF columns containing metabolite identifiers
    metabolite_columns = ['database_identifier', 'chemical_formula', 'smiles', 
                         'inchi', 'metabolite_identification']
    
    for col in metabolite_columns:
        if col in maf_df.columns:
            values = maf_df[col].dropna().astype(str).unique()
            
            # Filter valid metabolite identifiers
            filtered = [v.strip() for v in values 
                       if len(v.strip()) > 2 
                       and v.strip().lower() not in ['nan', 'na', 'null', 'unknown', 'n/a']
                       and not v.strip().startswith('http')
                       and not v.strip().endswith(('.txt', '.xlsx', '.csv'))]
            
            metabolites.update(filtered)
    
    return metabolites

def build_metabolite_vocabulary(study_ids: List[str], output_dir: str = "./data") -> Dict:
    """Build metabolite vocabulary from MetaboLights studies"""
    
    print(f"Building vocabulary from {len(study_ids)} studies...")
    
    all_metabolites = set()
    output_path = Path(output_dir)
    study_details = {}
    
    for study_id in study_ids:
        print(f"Processing {study_id}...")
        
        try:
            # Find and download MAF files
            maf_files = find_maf_files(study_id, output_path)
            
            study_metabolites = set()
            for maf_file in maf_files:
                maf_path = download_file(study_id, maf_file, output_path)
                maf_df = pd.read_csv(maf_path, sep='\t', low_memory=False)
                
                metabolites = extract_metabolites_from_maf(maf_df)
                study_metabolites.update(metabolites)
            
            all_metabolites.update(study_metabolites)
            study_details[study_id] = len(study_metabolites)
            
            print(f"  ✓ {len(study_metabolites)} metabolites")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    vocabulary = sorted(list(all_metabolites))
    
    print(f"\n🎯 Final vocabulary: {len(vocabulary)} metabolites")
    print(f"Sample: {vocabulary[:5]}")
    
    return {
        'metabolites': vocabulary,
        'vocab_size': len(vocabulary) + 4,  # +4 for [PAD], [CLS], [MASK], [UNK]
        'special_tokens': ['[PAD]', '[CLS]', '[MASK]', '[UNK]'],
        'source_studies': list(study_details.keys()),
        'study_counts': study_details
    }
