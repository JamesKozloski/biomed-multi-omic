# Standalone Real Metabolite Vocabulary Builder
# Completely self-contained - no module dependencies

import pandas as pd
import numpy as np
import requests
from pathlib import Path
import re
from typing import List, Dict, Optional, Tuple, Set

# You'll need to install BeautifulSoup if not already available:
# pip install beautifulsoup4

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    print("BeautifulSoup4 not available. Install with: pip install beautifulsoup4")
    BS4_AVAILABLE = False

def get_study_files_via_ftp(study_id: str) -> List[str]:
    """Get list of files for a study using FTP directory listing"""
    
    ftp_url = f"https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{study_id}/"
    
    try:
        response = requests.get(ftp_url, timeout=10)
        
        if response.status_code == 200:
            if BS4_AVAILABLE:
                # Parse the HTML directory listing
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract file links
                files = []
                for link in soup.find_all('a'):
                    href = link.get('href')
                    if href and not href.startswith('/') and href != '../':
                        files.append(href)
                
                print(f"✓ Found {len(files)} files for {study_id}")
                return files
            else:
                # Fallback: parse HTML manually (less robust)
                content = response.text
                # Simple regex to find file links
                file_pattern = r'<a href="([^"]+\.txt)"'
                files = re.findall(file_pattern, content)
                print(f"✓ Found {len(files)} .txt files for {study_id} (fallback parsing)")
                return files
        else:
            print(f"✗ Could not access FTP directory for {study_id}")
            return []
            
    except Exception as e:
        print(f"✗ Error listing files for {study_id}: {e}")
        return []

def download_file(study_id: str, filename: str, output_dir: Path) -> Optional[Path]:
    """Download a specific file using FTP URL pattern"""
    
    ftp_url = f"https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{study_id}/{filename}"
    output_path = output_dir / study_id / filename
    
    try:
        # Create directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download file
        response = requests.get(ftp_url, timeout=30)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"    ✓ Downloaded {filename} ({len(response.content)} bytes)")
            return output_path
        else:
            print(f"    ✗ Failed to download {filename}: status {response.status_code}")
            return None
            
    except Exception as e:
        print(f"    ✗ Error downloading {filename}: {e}")
        return None

def parse_file(file_path: Path) -> Optional[pd.DataFrame]:
    """Parse a file with robust error handling"""
    
    try:
        # Try TSV first (most common in MetaboLights)
        df = pd.read_csv(file_path, sep='\t', low_memory=False)
        
        if df.shape[1] > 1:  # Successfully parsed with multiple columns
            print(f"    ✓ Parsed {file_path.name}: {df.shape}")
            return df
        
        # Fallback to CSV
        df = pd.read_csv(file_path, sep=',', low_memory=False)
        print(f"    ✓ Parsed {file_path.name} as CSV: {df.shape}")
        return df
        
    except Exception as e:
        print(f"    ✗ Could not parse {file_path.name}: {e}")
        return None

def find_maf_references(df: pd.DataFrame) -> List[str]:
    """Find Metabolite Assignment File (MAF) references in a dataframe"""
    
    maf_files = []
    
    # Look for MAF file references in columns
    for col in df.columns:
        if 'metabolite assignment' in col.lower() or 'maf' in col.lower():
            # Get unique MAF file names
            maf_refs = df[col].dropna().unique()
            maf_files.extend([str(ref) for ref in maf_refs if str(ref) != 'nan'])
    
    return list(set(maf_files))  # Remove duplicates

def extract_metabolites_from_maf(maf_df: pd.DataFrame, maf_filename: str) -> Set[str]:
    """Extract actual metabolite identifiers from MAF dataframe"""
    
    print(f"    Analyzing MAF: {maf_filename}")
    print(f"      Shape: {maf_df.shape}")
    print(f"      Columns: {list(maf_df.columns)[:10]}...")
    
    metabolites = set()
    
    # Common MAF column names for metabolite identifiers
    metabolite_id_patterns = [
        r'.*database.*identifier.*',
        r'.*metabolite.*identification.*', 
        r'.*chemical.*name.*',
        r'.*metabolite.*name.*',
        r'.*compound.*name.*',
        r'.*hmdb.*',
        r'.*kegg.*',
        r'.*chebi.*',
        r'.*pubchem.*',
        r'.*inchi.*',
        r'.*smiles.*',
        r'.*formula.*',
        r'.*identifier.*',
        r'.*name.*',
        r'.*compound.*'
    ]
    
    # Find metabolite columns using pattern matching
    found_columns = []
    for col in maf_df.columns:
        col_lower = str(col).lower()
        for pattern in metabolite_id_patterns:
            if re.search(pattern, col_lower):
                found_columns.append(col)
                break
    
    print(f"      Potential metabolite columns: {found_columns}")
    
    # Extract metabolites from found columns
    for col in found_columns:
        if col in maf_df.columns:
            values = maf_df[col].dropna().astype(str).unique()
            
            # Filter to get actual metabolite identifiers
            filtered_values = []
            for v in values:
                v_clean = v.strip()
                
                # Keep if it looks like a real metabolite identifier
                if (len(v_clean) > 2 and 
                    not v_clean.lower() in ['nan', 'na', 'null', 'unknown', 'n/a', 
                                          'unidentified', 'not available', 'not applicable'] and
                    not v_clean.startswith('http') and  # Skip URLs
                    not v_clean.endswith(('.txt', '.xlsx', '.csv')) and  # Skip file names
                    not re.match(r'^[A-Z]+\d+[A-Z]*_\d+', v_clean) and  # Skip sample IDs
                    not re.match(r'^\d{4}-\d{2}-\d{2}', v_clean)):  # Skip dates
                    
                    filtered_values.append(v_clean)
            
            if filtered_values:
                print(f"        {col}: {len(filtered_values)} metabolites")
                print(f"          Examples: {filtered_values[:3]}")
                metabolites.update(filtered_values)
    
    return metabolites

def build_real_metabolite_vocabulary(study_ids: List[str] = None, output_dir: str = "./data") -> Tuple[List[str], Dict]:
    """Build real metabolite vocabulary from MetaboLights MAF files"""
    
    if study_ids is None:
        study_ids = ['MTBLS1', 'MTBLS2']
    
    print(f"=== Building Real Metabolite Vocabulary ===")
    print(f"Processing studies: {study_ids}")
    
    all_metabolites = set()
    study_details = {}
    output_path = Path(output_dir)
    
    for study_id in study_ids:
        print(f"\n--- Processing {study_id} ---")
        
        try:
            # Get file list for this study
            files = get_study_files_via_ftp(study_id)
            
            if not files:
                print(f"  ✗ No files found for {study_id}")
                continue
            
            # Download assay files (these contain MAF references)
            assay_files = [f for f in files if f.startswith('a_') and f.endswith('.txt')]
            print(f"  Found assay files: {assay_files}")
            
            study_metabolites = set()
            maf_files_processed = []
            
            # Process assay files to find MAF references
            for assay_file in assay_files[:2]:  # Process up to 2 assay files
                print(f"  Processing assay file: {assay_file}")
                
                # Download assay file
                assay_path = download_file(study_id, assay_file, output_path)
                
                if assay_path:
                    # Parse assay file
                    assay_df = parse_file(assay_path)
                    
                    if assay_df is not None:
                        # Find MAF references
                        maf_refs = find_maf_references(assay_df)
                        print(f"    MAF references found: {maf_refs}")
                        
                        # Download and process each MAF file
                        for maf_ref in maf_refs:
                            print(f"  Downloading MAF: {maf_ref}")
                            
                            maf_path = download_file(study_id, maf_ref, output_path)
                            
                            if maf_path:
                                maf_df = parse_file(maf_path)
                                
                                if maf_df is not None:
                                    maf_metabolites = extract_metabolites_from_maf(maf_df, maf_ref)
                                    study_metabolites.update(maf_metabolites)
                                    maf_files_processed.append(maf_ref)
                                    print(f"    ✓ Extracted {len(maf_metabolites)} metabolites from {maf_ref}")
            
            all_metabolites.update(study_metabolites)
            
            study_details[study_id] = {
                'assay_files': assay_files,
                'maf_files': maf_files_processed,
                'metabolite_count': len(study_metabolites),
                'sample_metabolites': sorted(list(study_metabolites))[:10]
            }
            
            print(f"  ✓ {study_id} total: {len(study_metabolites)} real metabolites")
            
        except Exception as e:
            print(f"  ✗ Error processing {study_id}: {e}")
            continue
    
    # Final vocabulary
    vocabulary = sorted(list(all_metabolites))
    
    print(f"\n🎯 REAL METABOLITE VOCABULARY RESULTS:")
    print(f"   Total unique metabolites: {len(vocabulary)}")
    print(f"   From {len(study_details)} studies")
    
    if vocabulary:
        print(f"   Sample metabolites:")
        for i, metabolite in enumerate(vocabulary[:20]):
            print(f"     {i+1:2d}. {metabolite}")
        if len(vocabulary) > 20:
            print(f"     ... and {len(vocabulary) - 20} more")
    else:
        print("   ⚠️  No metabolites found!")
        print("   This could mean:")
        print("     - MAF files use different column naming")
        print("     - Studies don't have MAF files")
        print("     - Different file format than expected")
    
    # Create vocabulary config for BMFM-RNA
    vocab_config = {
        'metabolites': vocabulary,
        'vocab_size': len(vocabulary) + 4,  # +4 for special tokens
        'special_tokens': ['[PAD]', '[CLS]', '[MASK]', '[UNK]'],
        'source_studies': list(study_details.keys()),
        'study_details': study_details,
        'total_maf_files': sum(len(details['maf_files']) for details in study_details.values())
    }
    
    return vocabulary, vocab_config

def analyze_vocabulary_quality(vocabulary: List[str]) -> Dict[str, int]:
    """Analyze the quality and composition of the metabolite vocabulary"""
    
    if not vocabulary:
        return {}
    
    print(f"\n=== Vocabulary Quality Analysis ===")
    
    categories = {
        'hmdb_ids': 0,          # HMDB0000123
        'kegg_ids': 0,          # C00022
        'chebi_ids': 0,         # CHEBI:15903
        'pubchem_ids': 0,       # Numeric
        'chemical_names': 0,    # glucose, alanine
        'chemical_formulas': 0, # C6H12O6
        'inchi_identifiers': 0, # InChI strings
        'other_identifiers': 0
    }
    
    for metabolite in vocabulary:
        metabolite_upper = metabolite.upper()
        
        if re.match(r'^HMDB\d+$', metabolite_upper):
            categories['hmdb_ids'] += 1
        elif re.match(r'^C\d+$', metabolite) or metabolite.startswith('KEGG:'):
            categories['kegg_ids'] += 1  
        elif metabolite.startswith('CHEBI:'):
            categories['chebi_ids'] += 1
        elif re.match(r'^\d+$', metabolite) and len(metabolite) > 3:
            categories['pubchem_ids'] += 1
        elif re.match(r'^[A-Za-z][a-z]+$', metabolite) and len(metabolite) > 3:
            categories['chemical_names'] += 1
        elif re.match(r'^C\d+H\d+', metabolite):
            categories['chemical_formulas'] += 1
        elif metabolite.startswith('InChI'):
            categories['inchi_identifiers'] += 1
        else:
            categories['other_identifiers'] += 1
    
    print("Vocabulary composition:")
    total = len(vocabulary)
    for category, count in categories.items():
        if count > 0:
            percentage = (count / total) * 100
            print(f"  {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    return categories

# Main execution
def main():
    """Main function to build and analyze real metabolite vocabulary"""
    
    print("=== MetaboLights Real Metabolite Vocabulary Builder ===")
    
    # Check dependencies
    if not BS4_AVAILABLE:
        print("⚠️  BeautifulSoup4 not available - using fallback HTML parsing")
    
    # Build vocabulary from real MAF files
    vocabulary, vocab_config = build_real_metabolite_vocabulary(
        study_ids=['MTBLS1', 'MTBLS2', 'MTBLS10'],  # Try multiple studies
        output_dir="./data"
    )
    
    if vocabulary:
        # Analyze vocabulary quality
        categories = analyze_vocabulary_quality(vocabulary)
        
        print(f"\n✅ SUCCESS - REAL METABOLITE VOCABULARY BUILT!")
        print(f"   Final vocabulary size: {len(vocabulary)} metabolites")
        print(f"   BMFM-compatible size: {vocab_config['vocab_size']} (with special tokens)")
        print(f"   Source studies: {len(vocab_config['source_studies'])}")
        print(f"   MAF files processed: {vocab_config['total_maf_files']}")
        print(f"   Data saved to: ./data/")
        
        # Vocabulary is ready for BMFM-RNA integration
        print(f"\n🎯 READY FOR BMFM-RNA VOCABULARY REPLACEMENT!")
        print(f"   Use vocab_config['metabolites'] as vocabulary")
        print(f"   Replace gene vocab with metabolite vocab")
        print(f"   Vocabulary covers real experimental metabolites")
        
        return vocab_config
    else:
        print("\n❌ NO METABOLITES EXTRACTED")
        print("Debugging suggestions:")
        print("1. Check if MAF files exist in the studies")
        print("2. Examine downloaded files manually")
        print("3. Try different study IDs") 
        print("4. Check MAF file column naming conventions")
        
        return None

if __name__ == "__main__":
    vocab_config = main()