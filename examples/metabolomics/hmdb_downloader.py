"""
HMDB Database Downloader - TRUE Database Discovery Pattern

Architecture: Download → Parse → Cache → Query
- Downloads HMDB metabolite database (XML format, ~220K metabolites)
- Parses XML to extract metabolite information
- Caches in SQLite for fast querying
- Provides browsing interface for vocabulary expansion

Pattern: Database Dump Integration (vs Query Validation)
"""

import requests
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import sqlite3
import json
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HMDBMetabolite:
    """Standardized HMDB metabolite record"""
    hmdb_id: str
    name: str
    synonyms: List[str]
    chemical_formula: str
    smiles: str
    inchi: str
    inchi_key: str
    molecular_weight: float
    kegg_id: Optional[str] = None
    chebi_id: Optional[str] = None
    pubchem_cid: Optional[str] = None
    metabolite_class: Optional[str] = None
    kingdom: Optional[str] = None
    super_class: Optional[str] = None
    sub_class: Optional[str] = None


class HMDBDatabaseDownloader:
    """
    Download and parse HMDB metabolite database.
    
    True database discovery pattern:
    - Downloads complete HMDB metabolite database
    - Extracts all metabolite entries
    - Enables browsing by chemical class, kingdom, etc.
    - No query validation - pure discovery
    """
    
    def __init__(self, cache_dir: str = "./hmdb_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.cache_dir / "hmdb_metabolites.db"
        self.xml_path = self.cache_dir / "hmdb_metabolites.xml"
        self.zip_path = self.cache_dir / "hmdb_metabolites.zip"
        
        # HMDB download URL (use "serum" subset for faster testing, or full for production)
        self.hmdb_urls = {
            'serum': 'https://hmdb.ca/system/downloads/current/serum_metabolites.zip',
            'urine': 'https://hmdb.ca/system/downloads/current/urine_metabolites.zip',
            'full': 'https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip'
        }
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for caching HMDB data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS hmdb_metabolites (
            hmdb_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            synonyms TEXT,  -- JSON array
            chemical_formula TEXT,
            smiles TEXT,
            inchi TEXT,
            inchi_key TEXT,
            molecular_weight REAL,
            kegg_id TEXT,
            chebi_id TEXT,
            pubchem_cid TEXT,
            metabolite_class TEXT,
            kingdom TEXT,
            super_class TEXT,
            sub_class TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create indices for fast querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON hmdb_metabolites(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kingdom ON hmdb_metabolites(kingdom)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_class ON hmdb_metabolites(metabolite_class)")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Initialized HMDB database at {self.db_path}")
    
    def download_hmdb_database(self, subset: str = 'serum') -> Path:
        """
        Download HMDB metabolite database.
        
        Args:
            subset: 'serum' (~5K metabolites, fast), 
                   'urine' (~3K metabolites),
                   'full' (~220K metabolites, slow)
        
        Returns:
            Path to downloaded XML file
        """
        if subset not in self.hmdb_urls:
            raise ValueError(f"Unknown subset: {subset}. Choose from {list(self.hmdb_urls.keys())}")
        
        url = self.hmdb_urls[subset]
        
        # Check if already downloaded
        if self.xml_path.exists():
            logger.info(f"HMDB database already downloaded at {self.xml_path}")
            return self.xml_path
        
        logger.info(f"Downloading HMDB {subset} database from {url}")
        logger.info("This may take several minutes depending on subset size...")
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(self.zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Downloaded to {self.zip_path}")
        
        # Extract XML from zip
        logger.info("Extracting XML from zip file...")
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            # Find the XML file in the zip
            xml_files = [f for f in zip_ref.namelist() if f.endswith('.xml')]
            if not xml_files:
                raise ValueError("No XML file found in downloaded zip")
            
            # Extract the first XML file
            zip_ref.extract(xml_files[0], self.cache_dir)
            extracted_path = self.cache_dir / xml_files[0]
            extracted_path.rename(self.xml_path)
        
        logger.info(f"Extracted XML to {self.xml_path}")
        
        # Clean up zip file
        self.zip_path.unlink()
        
        return self.xml_path
    
    def parse_hmdb_xml(self, xml_path: Optional[Path] = None) -> int:
        """
        Parse HMDB XML and load into SQLite database.
        
        Returns:
            Number of metabolites parsed and cached
        """
        if xml_path is None:
            xml_path = self.xml_path
        
        if not xml_path.exists():
            raise FileNotFoundError(f"HMDB XML not found at {xml_path}. Run download_hmdb_database() first.")
        
        logger.info(f"Parsing HMDB XML from {xml_path}")
        logger.info("This may take several minutes for large databases...")
        
        # Parse XML iteratively to handle large files
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        count = 0
        
        # Use iterparse for memory efficiency with large XML files
        context = ET.iterparse(xml_path, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        
        for event, elem in context:
            if event == 'end' and elem.tag.endswith('metabolite'):
                try:
                    metabolite = self._parse_metabolite_element(elem)
                    self._save_metabolite_to_db(cursor, metabolite)
                    count += 1
                    
                    if count % 1000 == 0:
                        conn.commit()
                        logger.info(f"Parsed {count} metabolites...")
                    
                except Exception as e:
                    logger.warning(f"Error parsing metabolite: {e}")
                
                # Clear element to free memory
                elem.clear()
                root.clear()
        
        conn.commit()
        conn.close()
        
        logger.info(f"Successfully parsed and cached {count} metabolites")
        return count
    
    def _parse_metabolite_element(self, elem: ET.Element) -> HMDBMetabolite:
        """Parse a single metabolite XML element"""
        
        # Extract namespace from element tag
        namespace = ''
        if elem.tag.startswith('{'):
            namespace = elem.tag.split('}')[0] + '}'
        
        # Helper to get text from element
        def get_text(tag: str, default: str = None) -> Optional[str]:
            child = elem.find(f".//{namespace}{tag}")
            return child.text if child is not None and child.text else default
        
        # Parse synonyms
        synonyms = []
        synonyms_elem = elem.find(f".//{namespace}synonyms")
        if synonyms_elem is not None:
            for syn in synonyms_elem.findall(f".//{namespace}synonym"):
                if syn.text:
                    synonyms.append(syn.text)
        
        # Parse taxonomy
        taxonomy = elem.find(f".//{namespace}taxonomy")
        kingdom = None
        super_class = None
        metabolite_class = None
        sub_class = None
        
        if taxonomy is not None:
            kingdom = get_text('kingdom')
            super_class = get_text('super_class')
            metabolite_class = get_text('class')
            sub_class = get_text('sub_class')
        
        # Create metabolite object
        return HMDBMetabolite(
            hmdb_id=get_text('accession'),
            name=get_text('name'),
            synonyms=synonyms,
            chemical_formula=get_text('chemical_formula'),
            smiles=get_text('smiles'),
            inchi=get_text('inchi'),
            inchi_key=get_text('inchikey'),
            molecular_weight=float(get_text('monisotopic_molecular_weight', 0) or 0),
            kegg_id=get_text('kegg_id'),
            chebi_id=get_text('chebi_id'),
            pubchem_cid=get_text('pubchem_compound_id'),
            metabolite_class=metabolite_class,
            kingdom=kingdom,
            super_class=super_class,
            sub_class=sub_class
        )
    
    def _save_metabolite_to_db(self, cursor, metabolite: HMDBMetabolite):
        """Save metabolite to SQLite database"""
        cursor.execute("""
        INSERT OR REPLACE INTO hmdb_metabolites 
        (hmdb_id, name, synonyms, chemical_formula, smiles, inchi, inchi_key,
         molecular_weight, kegg_id, chebi_id, pubchem_cid, metabolite_class,
         kingdom, super_class, sub_class)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metabolite.hmdb_id,
            metabolite.name,
            json.dumps(metabolite.synonyms),
            metabolite.chemical_formula,
            metabolite.smiles,
            metabolite.inchi,
            metabolite.inchi_key,
            metabolite.molecular_weight,
            metabolite.kegg_id,
            metabolite.chebi_id,
            metabolite.pubchem_cid,
            metabolite.metabolite_class,
            metabolite.kingdom,
            metabolite.super_class,
            metabolite.sub_class
        ))
    
    def get_metabolite_count(self) -> int:
        """Get total number of cached metabolites"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM hmdb_metabolites")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def browse_metabolites(
        self, 
        limit: int = 100, 
        offset: int = 0,
        kingdom: Optional[str] = None,
        metabolite_class: Optional[str] = None
    ) -> List[Dict]:
        """
        Browse metabolites with optional filtering.
        
        TRUE database discovery: Query what the database has, not what we think it should have.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT hmdb_id, name, chemical_formula, molecular_weight, kingdom, metabolite_class FROM hmdb_metabolites WHERE 1=1"
        params = []
        
        if kingdom:
            query += " AND kingdom = ?"
            params.append(kingdom)
        
        if metabolite_class:
            query += " AND metabolite_class = ?"
            params.append(metabolite_class)
        
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'hmdb_id': row[0],
                'name': row[1],
                'chemical_formula': row[2],
                'molecular_weight': row[3],
                'kingdom': row[4],
                'metabolite_class': row[5]
            })
        
        conn.close()
        return results
    
    def get_random_metabolites(self, count: int) -> List[str]:
        """
        Get random metabolite names for vocabulary expansion.
        
        Pattern: Random sampling for diversity
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT name FROM hmdb_metabolites 
        WHERE name IS NOT NULL 
        ORDER BY RANDOM() 
        LIMIT ?
        """, (count,))
        
        metabolites = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return metabolites
    
    def get_metabolites_by_class(self, metabolite_class: str, limit: int = 1000) -> List[str]:
        """Get metabolites filtered by chemical class"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT name FROM hmdb_metabolites 
        WHERE metabolite_class = ? AND name IS NOT NULL
        LIMIT ?
        """, (metabolite_class, limit))
        
        metabolites = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return metabolites
    
    def get_all_metabolite_names(self, limit: Optional[int] = None) -> List[str]:
        """
        Get all metabolite names from database.
        
        TRUE database discovery: Return what HMDB has, not what we think it should have.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if limit:
            cursor.execute("SELECT name FROM hmdb_metabolites WHERE name IS NOT NULL LIMIT ?", (limit,))
        else:
            cursor.execute("SELECT name FROM hmdb_metabolites WHERE name IS NOT NULL")
        
        metabolites = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        logger.info(f"Retrieved {len(metabolites)} metabolite names from HMDB")
        return metabolites
    
    def get_metabolite_classes(self) -> Dict[str, int]:
        """Get distribution of metabolites by class"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT metabolite_class, COUNT(*) 
        FROM hmdb_metabolites 
        WHERE metabolite_class IS NOT NULL
        GROUP BY metabolite_class
        ORDER BY COUNT(*) DESC
        """)
        
        classes = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        
        return classes


# Integration with existing MetaboliteDatabaseIntegrator
def integrate_hmdb_with_vocabulary_builder(
    base_metabolites: Set[str],
    target_size: int,
    hmdb_downloader: HMDBDatabaseDownloader
) -> List[str]:
    """
    Use HMDB database for TRUE vocabulary expansion.
    
    Architecture: Database Discovery Pattern
    - Query HMDB for metabolites we DON'T already have
    - Random sampling for diversity
    - Class-based sampling for coverage
    """
    
    expanded_metabolites = list(base_metabolites)
    base_set = set(base_metabolites)
    
    needed = target_size - len(expanded_metabolites)
    logger.info(f"Need {needed} metabolites from HMDB to reach target of {target_size}")
    
    # Strategy 1: Get random metabolites for diversity (50% of needed)
    random_count = needed // 2
    random_metabolites = hmdb_downloader.get_random_metabolites(random_count * 2)  # 2x buffer
    
    added_random = 0
    for metabolite in random_metabolites:
        if metabolite not in base_set and len(expanded_metabolites) < target_size:
            expanded_metabolites.append(metabolite)
            base_set.add(metabolite)
            added_random += 1
    
    logger.info(f"Added {added_random} random metabolites from HMDB")
    
    # Strategy 2: Sample by class for coverage (remaining needed)
    if len(expanded_metabolites) < target_size:
        remaining = target_size - len(expanded_metabolites)
        classes = hmdb_downloader.get_metabolite_classes()
        
        # Sample from top classes proportionally
        samples_per_class = max(1, remaining // len(classes))
        
        for metabolite_class, count in classes.items():
            if len(expanded_metabolites) >= target_size:
                break
            
            class_metabolites = hmdb_downloader.get_metabolites_by_class(
                metabolite_class, 
                limit=samples_per_class * 2
            )
            
            for metabolite in class_metabolites:
                if metabolite not in base_set and len(expanded_metabolites) < target_size:
                    expanded_metabolites.append(metabolite)
                    base_set.add(metabolite)
    
    logger.info(f"Final vocabulary size: {len(expanded_metabolites)}")
    
    return expanded_metabolites


# Example usage
if __name__ == "__main__":
    # Initialize downloader
    downloader = HMDBDatabaseDownloader()
    
    # Download and parse HMDB database (one-time setup)
    # Use 'serum' for testing (~5K metabolites), 'full' for production (~220K)
    xml_path = downloader.download_hmdb_database(subset='serum')
    count = downloader.parse_hmdb_xml(xml_path)
    
    print(f"\n✅ HMDB database ready with {count} metabolites")
    
    # Example: Browse metabolites
    metabolites = downloader.browse_metabolites(limit=10)
    print("\nSample metabolites:")
    for m in metabolites:
        print(f"  {m['name']} ({m['hmdb_id']}) - {m['metabolite_class']}")
    
    # Example: Get metabolite classes
    classes = downloader.get_metabolite_classes()
    print(f"\nMetabolite classes available: {len(classes)}")
    print("Top 5 classes:")
    for cls, count in list(classes.items())[:5]:
        print(f"  {cls}: {count} metabolites")
