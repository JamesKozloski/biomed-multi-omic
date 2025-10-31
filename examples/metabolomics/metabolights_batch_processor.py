# metabolights_batch_processor.py (FIXED - No redundant downloads)
"""
Batch processing pipeline with automatic study downloading.
Downloads MAF files on-demand and tracks download attempts to avoid redundant work.
"""

import pandas as pd
import numpy as np
import requests
from ftplib import FTP
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import logging
from datetime import datetime
from metabolomics_to_h5ad import MetabolomicsToH5ADConverter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MetaboLightsBatchProcessor:
    """Discover and process MetaboLights studies with auto-download and caching"""
    
    def __init__(
        self, 
        vocab_dir: str = "./bmfm_metabolomics_vocab_10k",
        output_dir: str = "./h5ad_data_batch",
        cache_dir: str = "./metabolights_cache",
        data_dir: str = "./data"
    ):
        self.converter = MetabolomicsToH5ADConverter(vocab_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.data_dir = Path(data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.ftp_host = "ftp.ebi.ac.uk"
        self.ftp_path = "/pub/databases/metabolights/studies/public"
        
        # Processing statistics
        self.stats = {
            'discovered': 0,
            'attempted': 0,
            'successful': 0,
            'rejected_negative': 0,
            'rejected_no_maf': 0,
            'rejected_error': 0,
            'downloaded': 0,
            'download_failed': 0,
            'download_skipped': 0,
            'total_samples': 0,
            'total_metabolites': set(),
            'study_details': []
        }
    
    def discover_studies_from_ftp(self) -> List[str]:
        """Discover all public studies from FTP"""
        logger.info(f"Discovering studies from FTP: {self.ftp_host}{self.ftp_path}")
        
        try:
            ftp = FTP(self.ftp_host)
            ftp.login()
            ftp.cwd(self.ftp_path)
            
            studies = []
            items = ftp.nlst()
            
            for item in items:
                if item.startswith('MTBLS') and item[5:].isdigit():
                    studies.append(item)
            
            ftp.quit()
            
            logger.info(f"Found {len(studies)} public studies on FTP")
            self.stats['discovered'] = len(studies)
            
            # Save discovered studies
            discovery_file = self.cache_dir / f"discovered_studies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(discovery_file, 'w') as f:
                json.dump(sorted(studies), f, indent=2)
            
            return sorted(studies)
            
        except Exception as e:
            logger.error(f"Error discovering studies from FTP: {e}")
            return []
    
    def _has_local_maf_files(self, study_id: str) -> bool:
        """Check if study has MAF files locally"""
        study_path = self.data_dir / study_id
        if not study_path.exists():
            return False
        maf_files = list(study_path.glob("m_*.tsv"))
        return len(maf_files) > 0
    
    def _download_attempted(self, study_id: str) -> bool:
        """Check if download was already attempted for this study"""
        study_path = self.data_dir / study_id
        download_marker = study_path / ".download_attempted"
        return download_marker.exists()
    
    def _mark_download_attempted(self, study_id: str, success: bool) -> None:
        """Mark that download was attempted"""
        study_path = self.data_dir / study_id
        study_path.mkdir(parents=True, exist_ok=True)
        download_marker = study_path / ".download_attempted"
        
        # Write status to marker file
        with open(download_marker, 'w') as f:
            f.write(f"success={success}\n")
            f.write(f"timestamp={datetime.now().isoformat()}\n")
    
    def _download_study_maf_files(self, study_id: str) -> bool:
        """
        Download MAF files for a study from FTP.
        Returns True if successful, False otherwise.
        """
        study_path = self.data_dir / study_id
        study_path.mkdir(parents=True, exist_ok=True)
        
        try:
            ftp = FTP(self.ftp_host)
            ftp.login()
            ftp.cwd(f"{self.ftp_path}/{study_id}")
            
            # List all files
            files = ftp.nlst()
            
            # Download MAF files (m_*.tsv) and assay files (a_*.txt)
            maf_files = [f for f in files if f.startswith('m_') and f.endswith('.tsv')]
            assay_files = [f for f in files if f.startswith('a_') and f.endswith('.txt')]
            
            if not maf_files:
                logger.debug(f"{study_id}: No MAF files found on FTP")
                ftp.quit()
                self._mark_download_attempted(study_id, False)
                return False
            
            # Download MAF files
            downloaded_count = 0
            for filename in maf_files:
                local_file = study_path / filename
                if local_file.exists():
                    downloaded_count += 1
                    continue  # Skip if already downloaded
                
                try:
                    with open(local_file, 'wb') as f:
                        ftp.retrbinary(f'RETR {filename}', f.write)
                    downloaded_count += 1
                    logger.debug(f"Downloaded {filename}")
                except Exception as e:
                    logger.debug(f"Error downloading {filename}: {e}")
            
            # Download assay files (needed for MAF reference)
            for filename in assay_files:
                local_file = study_path / filename
                if local_file.exists():
                    continue
                
                try:
                    with open(local_file, 'wb') as f:
                        ftp.retrbinary(f'RETR {filename}', f.write)
                    logger.debug(f"Downloaded {filename}")
                except Exception as e:
                    logger.debug(f"Error downloading {filename}: {e}")
            
            ftp.quit()
            
            if downloaded_count > 0:
                self.stats['downloaded'] += 1
                self._mark_download_attempted(study_id, True)
                return True
            else:
                self._mark_download_attempted(study_id, False)
                return False
            
        except Exception as e:
            logger.debug(f"Error downloading {study_id}: {e}")
            self.stats['download_failed'] += 1
            self._mark_download_attempted(study_id, False)
            return False
    
    def process_studies_batch(
        self,
        study_ids: List[str],
        target_samples: int = 20000,
        max_studies: int = 300
    ) -> List[str]:
        """Process studies with automatic downloading and caching"""
        logger.info(f"="*70)
        logger.info(f"BATCH PROCESSING: Target {target_samples} samples from max {max_studies} studies")
        logger.info(f"="*70)
        
        successful_h5ads = []
        
        for i, study_id in enumerate(study_ids[:max_studies]):
            if self.stats['total_samples'] >= target_samples:
                logger.info(f"Target samples reached: {self.stats['total_samples']}")
                break
            
            logger.info(f"\n[{i+1}/{min(len(study_ids), max_studies)}] Processing {study_id}")
            self.stats['attempted'] += 1
            
            try:
                # Check if we have local MAF files
                has_local_files = self._has_local_maf_files(study_id)
                
                if not has_local_files:
                    # Check if we already tried downloading this study
                    if self._download_attempted(study_id):
                        logger.info(f"  Download previously attempted and failed - skipping")
                        self.stats['download_skipped'] += 1
                        self.stats['rejected_no_maf'] += 1
                        continue
                    
                    # Try downloading
                    logger.info(f"  Downloading MAF files...")
                    if not self._download_study_maf_files(study_id):
                        self.stats['rejected_no_maf'] += 1
                        logger.warning(f"  No MAF files available")
                        continue
                
                # Convert study
                adatas = self.converter.convert_study_to_h5ad(
                    study_id=study_id,
                    data_dir=self.data_dir,
                    output_dir=self.output_dir
                )
                
                if not adatas:
                    self.stats['rejected_no_maf'] += 1
                    logger.warning(f"  Rejected (no valid data or pre-transformed)")
                    continue
                
                # Update statistics
                study_samples = sum(a.n_obs for a in adatas)
                study_metabolites = set()
                for a in adatas:
                    study_metabolites.update(a.var_names)
                
                self.stats['successful'] += 1
                self.stats['total_samples'] += study_samples
                self.stats['total_metabolites'].update(study_metabolites)
                
                # Record study details
                self.stats['study_details'].append({
                    'study_id': study_id,
                    'n_samples': study_samples,
                    'n_metabolites': len(study_metabolites),
                    'n_h5ad_files': len(adatas)
                })
                
                logger.info(f"  ✓ {study_samples} samples, {len(study_metabolites)} metabolites")
                logger.info(f"  Progress: {self.stats['total_samples']}/{target_samples} samples "
                          f"({self.stats['successful']}/{self.stats['attempted']} studies)")
                
            except Exception as e:
                self.stats['rejected_error'] += 1
                logger.error(f"  ✗ Error - {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return successful_h5ads
    
    def generate_report(self, output_file: str = "batch_processing_report.txt"):
        """Generate processing statistics report"""
        report = []
        report.append("="*70)
        report.append("METABOLIGHTS BATCH PROCESSING REPORT")
        report.append("="*70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("PROCESSING STATISTICS")
        report.append("-"*70)
        report.append(f"Studies discovered: {self.stats['discovered']}")
        report.append(f"Studies attempted: {self.stats['attempted']}")
        report.append(f"Studies downloaded (new): {self.stats['downloaded']}")
        report.append(f"Downloads skipped (cached): {self.stats['download_skipped']}")
        report.append(f"Studies successful: {self.stats['successful']}")
        report.append(f"Download failures: {self.stats['download_failed']}")
        report.append(f"Rejected (no MAF/pre-transformed): {self.stats['rejected_no_maf']}")
        report.append(f"Rejected (errors): {self.stats['rejected_error']}")
        report.append(f"Success rate: {100*self.stats['successful']/max(self.stats['attempted'],1):.1f}%")
        report.append("")
        
        report.append("DATA STATISTICS")
        report.append("-"*70)
        report.append(f"Total samples: {self.stats['total_samples']}")
        report.append(f"Unique metabolites: {len(self.stats['total_metabolites'])}")
        if self.stats['successful'] > 0:
            report.append(f"Average samples/study: {self.stats['total_samples']/self.stats['successful']:.1f}")
        report.append("")
        
        if self.stats['study_details']:
            report.append("TOP 20 STUDIES BY SAMPLE COUNT")
            report.append("-"*70)
            sorted_studies = sorted(
                self.stats['study_details'], 
                key=lambda x: x['n_samples'], 
                reverse=True
            )[:20]
            for study in sorted_studies:
                report.append(f"{study['study_id']}: {study['n_samples']} samples, "
                            f"{study['n_metabolites']} metabolites")
        
        report_text = "\n".join(report)
        print("\n" + report_text)
        
        report_file = self.output_dir / output_file
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {report_file}")
        return report_text
    
    def merge_all_h5ad_files(self, output_filename: str = "metabolomics_training_large.h5ad"):
        """Merge all successfully processed h5ad files"""
        logger.info("Merging all h5ad files...")
        
        h5ad_files = list(self.output_dir.glob("MTBLS*.h5ad"))
        
        if not h5ad_files:
            logger.error("No h5ad files found to merge")
            return None
        
        logger.info(f"Found {len(h5ad_files)} h5ad files to merge")
        
        import anndata as ad
        adatas = []
        for h5ad_file in h5ad_files:
            try:
                adata = ad.read_h5ad(h5ad_file)
                adata.var_names_make_unique()
                adatas.append(adata)
            except Exception as e:
                logger.error(f"Error loading {h5ad_file}: {e}")
        
        merged = self.converter.merge_studies(
            adatas,
            output_file=self.output_dir / output_filename
        )
        
        return merged


def main():
    """Main batch processing pipeline with auto-download and caching"""
    print("="*70)
    print("METABOLIGHTS BATCH PROCESSING PIPELINE WITH AUTO-DOWNLOAD")
    print("="*70)
    print("Features:")
    print("  - Automatic MAF file downloading from FTP")
    print("  - Download caching to avoid redundant work")
    print("  - Filters pre-transformed data")
    print("  - Handles encoding errors")
    print("="*70)
    print("Target: 15,000-20,000 samples from ~250-300 studies")
    print("="*70)
    
    processor = MetaboLightsBatchProcessor()
    
    # Step 1: Discover studies from FTP
    print("\n[STEP 1] Discovering studies from FTP...")
    discovered = processor.discover_studies_from_ftp()
    
    print(f"\nDiscovered {len(discovered)} public studies")
    
    if len(discovered) == 0:
        print("No studies found. Exiting.")
        return
    
    # Step 2: Process studies with auto-download
    print("\n[STEP 2] Processing studies (with auto-download and caching)...")
    print("Progress logged to batch_processing.log")
    
    processor.process_studies_batch(
        study_ids=discovered,
        target_samples=20000,
        max_studies=300
    )
    
    # Step 3: Generate report
    print("\n[STEP 3] Generating report...")
    processor.generate_report()
    
    # Step 4: Merge all data
    print("\n[STEP 4] Merging all h5ad files...")
    merged = processor.merge_all_h5ad_files()
    
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    if merged:
        print(f"Final dataset: {merged.shape}")
    print(f"Output directory: {processor.output_dir}")


if __name__ == "__main__":
    main()